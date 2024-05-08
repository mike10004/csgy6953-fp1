#!/usr/bin/env python3

import csv
import queue
import sys
from pathlib import Path
from queue import Queue
from typing import Any
from typing import Callable
from typing import Collection
from typing import Iterator
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from threading import Thread
from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch
import tabulate
from torch.optim import Optimizer
from torch.nn import Module
from tqdm import tqdm

import dlfp.utils
from dlfp.models import Seq2SeqTransformer
from dlfp.translate import NodeNavigator
from dlfp.translate import Suggestion
from dlfp.utils import Bilinguist
from dlfp.train import TrainLoaders
from dlfp.train import Trainer
from dlfp.train import create_model
from dlfp.translate import Translator
from dlfp.utils import Checkpointer
from dlfp.utils import EpochResult
from dlfp.utils import PhrasePairDataset
from dlfp.utils import Restored
from dlfp.utils import Split


StringTransform = Callable[[str], str]
DEFAULT_RANKS = (1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
ATTEMPTS_TOP_K = 10


class Attempt(NamedTuple):

    index: int
    source: str
    target: str
    rank: int
    suggestion_count: int
    top: tuple[str, ...]

    @staticmethod
    def headers(top_k: int) -> list[str]:
        return list(Attempt._fields[:-1]) + [f"top_{i+1}" for i in range(top_k)]

    def to_row(self) -> list[Any]:
        return [self.index, self.source, self.target, self.rank, self.suggestion_count] + list(self.top)


class AccuracyResult(NamedTuple):

    rank_acc_count: dict[int, int]
    attempt_count: int

    @staticmethod
    def table_headers() -> list[Any]:
        return ["rank", "count", "percent"]

    def to_table(self) -> list[list[Any]]:
        ranks = sorted(self.rank_acc_count.keys())
        table = []
        for rank in ranks:
            count = self.rank_acc_count[rank]
            proportion = count / self.attempt_count
            table.append([rank, count, proportion * 100.0])
        return table


def measure_accuracy(attempt_file: Path, ranks: Collection[int] = None) -> AccuracyResult:
    ranks = ranks or DEFAULT_RANKS
    ranks = list(sorted(ranks, reverse=True))
    rank_acc_count = defaultdict(int)
    attempt_count = 0
    with open(attempt_file, "r") as ifile:
        csv_reader = csv.DictReader(ifile)
        for row in csv_reader:
            attempt_count += 1
            row: dict[str, str]
            try:
                actual_rank = int(row["rank"])
            except ValueError:
                actual_rank = None
            if actual_rank is not None:
                for rank in ranks:
                    if actual_rank <= rank:
                        rank_acc_count[rank] += 1
                    else:
                        break
    return AccuracyResult(rank_acc_count, attempt_count)


class ModelManager:

    def __init__(self, model: Seq2SeqTransformer, bilinguist: Bilinguist, device):
        self.device = device
        self.model = model
        self.bilinguist = bilinguist
        self.device = device
        self.src_transform: StringTransform = dlfp.utils.identity
        self.tgt_transform: StringTransform = dlfp.utils.identity
        self.node_navigator: Optional[NodeNavigator] = None

    def evaluate(self,
                 dataset: PhrasePairDataset,
                 suggestion_count: int,
                 callback: Callable[[Attempt], None],
                 hide_progress: bool = False,
                 concurrency: int = None):
        progress_bar = tqdm(file=sys.stdout, total=len(dataset), disable=hide_progress)
        def perform(dataset_part: PhrasePairDataset):
            for index, (src_phrase, tgt_phrase, suggestions) in enumerate(self._iterate_guesses(dataset_part, limit=None, guesses_per_phrase=suggestion_count)):
                phrases = [s.phrase for s in suggestions]
                try:
                    actual_rank = phrases.index(tgt_phrase) + 1
                except ValueError:
                    actual_rank = float("nan")
                if callback is not None:
                    callback(Attempt(index, src_phrase, tgt_phrase, actual_rank, len(phrases), tuple(phrases[:ATTEMPTS_TOP_K])))
                progress_bar.update(1)
        if concurrency is not None:
            partitioning = dataset.partition(concurrency)
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                for partitioning_item in partitioning:
                    partitioning_item: PhrasePairDataset
                    executor.submit(perform, partitioning_item)
        else:
            perform(dataset)

    def print_translations(self, dataset: PhrasePairDataset, limit: int):
        for index, (src_phrase, tgt_phrase, suggestions) in enumerate(self._iterate_guesses(dataset, limit=limit)):
            if index > 0:
                print()
            print(f"{index: 2d} src: {src_phrase}")
            print(f"{index: 2d} tgt: {tgt_phrase}")
            print(f"{index: 2d} xxx: {suggestions[0].phrase}")

    def _iterate_guesses(self, dataset: PhrasePairDataset, limit: Optional[int] = None, guesses_per_phrase: Optional[int] = None) -> Iterator[Tuple[str, str, list[Suggestion]]]:
        translator = Translator(self.model, self.bilinguist, self.device)
        for index, (src_phrase, tgt_phrase) in enumerate(dataset):
            if limit is not None and index >= limit:
                break
            src_phrase = self.src_transform(src_phrase)
            tgt_phrase = self.tgt_transform(tgt_phrase)
            if guesses_per_phrase is None:
                translation = translator.translate(src_phrase).strip()
                translation = self.tgt_transform(translation)
                suggestions = [Suggestion(translation, float("nan"))]
            else:
                suggestions = translator.suggest(src_phrase, count=guesses_per_phrase, navigator=self.node_navigator)
                if not self.tgt_transform is dlfp.utils.identity:
                    suggestions = [Suggestion(self.tgt_transform(s.phrase), s.probability) for s in suggestions]
            yield src_phrase, tgt_phrase, suggestions


    def train(self, loaders: TrainLoaders, checkpoints_dir: Path, train_config: 'TrainConfig'):
        trainer = Trainer(self.model, pad_idx=self.bilinguist.source.specials.indexes.pad, device=self.device)
        trainer.optimizer_factory = train_config.create_optimizer
        print(f"writing checkpoints to {checkpoints_dir}")
        checkpointer = Checkpointer(checkpoints_dir, self.model)
        checkpointer.retain_all = train_config.retain_all_checkpoints
        checkpointer.extra = {
            "train_config": train_config.to_jsonable(),
        }
        results = trainer.train(loaders, train_config.epoch_count, callback=checkpointer.checkpoint)
        results_table = [
            (r.epoch_index, r.train_loss, r.valid_loss)
            for r in results
        ]
        print(tabulate.tabulate(results_table, headers=EpochResult._fields))


class DataSuperset(NamedTuple):

    train: PhrasePairDataset
    valid: PhrasePairDataset


class TrainConfig(NamedTuple):

    dataset_name: str
    checkpoints_dir: Path
    epoch_count: int = 10
    batch_size: int = 128
    retain_all_checkpoints: bool = False
    lr: float = 0.0001
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-9

    def to_jsonable(self) -> dict[str, Optional[str]]:
        return {k:(None if v is None else str(v)) for k, v in self._asdict().items()}

    def create_optimizer(self, model: Module) -> Optimizer:
        return torch.optim.Adam(model.parameters(), lr=self.lr, betas=self.betas, eps=self.eps)

    @staticmethod
    def argument_keys() -> list[str]:
        return [k for k in TrainConfig._fields if not k in {"dataset_name", "checkpoints_dir"}]

    @staticmethod
    def from_args(dataset_name: str, checkpoints_dir: Path, arguments: Optional[list[str]]) -> 'TrainConfig':
        # epoch_count=args.epoch_count, batch_size=args.batch_size, retain_all_checkpoints=args.retain
        types = {
            'epoch_count': int,
            'batch_size': int,
            'retain_all_checkpoints': bool,
            'betas': lambda s: tuple(float(b) for b in s.split(',')),
        }
        kwargs = {}
        for arg in (arguments or []):
            key, value = arg.split('=', maxsplit=1)
            if not key in TrainConfig._fields:
                raise ValueError(f'not a valid --train-config argument key: {key}; allowed keys are {TrainConfig.argument_keys()}')
            value_type = types.get(key, float)
            value = value_type(value)
            kwargs[key] = value
        return TrainConfig(dataset_name, checkpoints_dir, **kwargs)



class Runnable(NamedTuple):

    superset: DataSuperset
    bilinguist: Bilinguist
    manager: ModelManager


class Runner:

    def describe(self) -> str:
        raise NotImplementedError("abstract")

    def resolve_dataset(self, dataset_name: str = None) -> DataSuperset:
        raise NotImplementedError("abstract")

    def create_bilinguist(self, superset: DataSuperset) -> Bilinguist:
        raise NotImplementedError("abstract")

    def create_model(self, bilinguist: Bilinguist, device: str) -> Seq2SeqTransformer:
        model = create_model(
            src_vocab_size=len(bilinguist.source.vocab),
            tgt_vocab_size=len(bilinguist.target.vocab),
            DEVICE=device,
        )
        return model

    def run_train(self, train_config: TrainConfig, device: str):
        r = self.create_runnable(train_config.dataset_name, device)
        loaders = TrainLoaders.from_datasets(r.superset.train, r.superset.valid, collate_fn=r.bilinguist.collate, batch_size=train_config.batch_size)
        r.manager.train(loaders, train_config.checkpoints_dir, train_config)

    def create_runnable(self, dataset_name: str, device: str) -> Runnable:
        superset = self.resolve_dataset(dataset_name)
        bilinguist = self.create_bilinguist(superset)
        model = self.create_model(bilinguist, device)
        model_manager = ModelManager(model, bilinguist, device)
        return Runnable(superset, bilinguist, model_manager)

    def run_eval(self, restored: Restored, dataset_name: str, device: str, output_file: Path, split: Split = "valid", concurrency: Optional[int] = None):
        r = self.create_runnable(dataset_name, device)
        r.manager.model.load_state_dict(restored.model_state_dict)
        dataset = getattr(r.superset, split)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        completed = False
        attempt_q = Queue()
        def write_csv():
            with open(output_file, "w", newline="", encoding="utf-8") as ofile:
                csv_writer = csv.writer(ofile)
                csv_writer.writerow(Attempt.headers(ATTEMPTS_TOP_K))
                while (not completed) or attempt_q:
                    try:
                        attempt_ = attempt_q.get(timeout=0.25)
                        csv_writer.writerow(attempt_.to_row())
                    except queue.Empty:
                        pass
        write_csv_thread = Thread(target=write_csv)
        write_csv_thread.start()
        try:
            r.manager.evaluate(dataset, max(DEFAULT_RANKS), callback=attempt_q.put, concurrency=concurrency)
        finally:
            completed = True
        print("finishing csv writes...", end="")
        write_csv_thread.join()
        print("done")
        print("split:", split)
        result = measure_accuracy(output_file, DEFAULT_RANKS)
        table = result.to_table()
        print(tabulate.tabulate(table, headers=AccuracyResult.table_headers()))

    def run_demo(self, restored: Restored, dataset_name: str, device: str, limit: int = ...):
        r = self.create_runnable(dataset_name, device)
        if limit is ...:
            limit = 5
        if limit is None:
            limit = len(r.superset.valid)
        r.manager.model.load_state_dict(restored.model_state_dict)
        r.manager.print_translations(r.superset.valid, limit=limit)


def main(runner: Runner) -> int:
    parser = ArgumentParser(description=runner.describe(), epilog=f"Allowed --train-config keys are: {TrainConfig.argument_keys()}")
    parser.add_argument("-m", "--mode", metavar="MODE", choices=("train", "eval", "demo"), default="train", help="'train', 'eval', or 'demo' (print some outputs)")
    parser.add_argument("-o", "--output", metavar="DIR", help="output root directory")
    parser.add_argument("-f", "--file", metavar="FILE", help="checkpoint file for eval mode")
    parser.add_argument("-c", "--train-config", metavar="K=V", action='append', help="set training configuration parameter")
    parser.add_argument("--limit", type=int, default=..., metavar="N", help="demo mode phrase limit")
    parser.add_argument("-d", "--dataset", metavar="NAME", help="specify dataset name")
    split_choices = ("train", "valid", "test")
    parser.add_argument("-s", "--split", metavar="SPLIT", choices=split_choices, help="eval mode dataset split")
    parser.add_argument("--concurrency", type=int, help="eval mode concurrency")
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed = 0
    torch.manual_seed(seed)
    runner.dataset_name = args.dataset
    if args.mode == "demo":
        checkpoint_file = args.file
        if not checkpoint_file:
            parser.error("checkpoint file must be specified")
            return 1
        restored = Restored.from_file(checkpoint_file, device=device)
        runner.run_demo(restored, args.dataset, device, args.limit)
        return 0
    elif args.mode == "eval":
        checkpoint_file = args.file
        if not checkpoint_file:
            parser.error("checkpoint file must be specified")
            return 1
        checkpoint_file = Path(checkpoint_file)
        restored = Restored.from_file(checkpoint_file, device=device)
        split = args.split or "valid"
        output_file = Path(args.output or ".") / f"evaluations/{checkpoint_file.stem}_{split}_{dlfp.utils.timestamp()}.csv"
        runner.run_eval(restored, args.dataset, device, output_file, split=split, concurrency=args.concurrency)
        return 0
    elif args.mode == "train":
        checkpoints_dir = Path(args.output or ".") / f"checkpoints/{dlfp.utils.timestamp()}"
        train_config = TrainConfig.from_args(args.dataset, checkpoints_dir, args.train_config)
        runner.run_train(train_config, device)
        return 0
    else:
        parser.error("BUG unhandled mode")
        return 2
