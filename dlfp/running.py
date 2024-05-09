#!/usr/bin/env python3

import csv
import json
import sys
import queue
from pathlib import Path
from queue import Queue
from typing import Any
from typing import Callable
from typing import Iterator
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from threading import Thread
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.optim import Optimizer
from torch.nn import Module
from torchtext.vocab import Vocab
from tqdm import tqdm

import dlfp.utils
import dlfp.common
import dlfp.models
from dlfp.models import Seq2SeqTransformer
from dlfp.translate import Node
from dlfp.translate import NodeNavigator
from dlfp.translate import Suggestion
from dlfp.utils import Bilinguist
from dlfp.train import TrainLoaders
from dlfp.train import Trainer
from dlfp.models import ModelHyperparametry
from dlfp.translate import Translator
from dlfp.utils import Checkpointer
from dlfp.utils import EpochResult
from dlfp.utils import PhrasePairDataset
from dlfp.utils import Restored
from dlfp.utils import Split
from dlfp.common import Table
from dlfp.metrics import measure_accuracy
from dlfp.metrics import DEFAULT_RANKS

StringTransform = Callable[[str], str]
ATTEMPTS_TOP_K = 10


class Attempt(NamedTuple):

    index: int
    source: str
    target: str
    rank: int
    suggestion_count: int
    top: tuple[str, ...]
    nodes: Optional[list[Node]] = None

    @staticmethod
    def headers(top_k: int) -> list[str]:
        return list(Attempt._fields[:-1]) + [f"top_{i+1}" for i in range(top_k)]

    def to_row(self) -> list[Any]:
        return [self.index, self.source, self.target, self.rank, self.suggestion_count] + list(self.top)


class ModelManager:

    def __init__(self, model: Seq2SeqTransformer, bilinguist: Bilinguist, device: str):
        self.device = device
        self.model = model
        self.bilinguist = bilinguist
        self.device = device
        self.src_transform: StringTransform = dlfp.common.identity
        self.tgt_transform: StringTransform = dlfp.common.identity
        self.node_navigator: Optional[NodeNavigator] = None

    def evaluate(self,
                 dataset: PhrasePairDataset,
                 suggestion_count: int,
                 callback: Callable[[Attempt], None],
                 hide_progress: bool = False,
                 concurrency: int = None):
        progress_bar = tqdm(file=sys.stdout, total=len(dataset), disable=hide_progress)
        def perform(dataset_part: PhrasePairDataset):
            for index, (src_phrase, tgt_phrase, suggestions, nodes) in enumerate(self._iterate_guesses(dataset_part, limit=None, guesses_per_phrase=suggestion_count)):
                phrases = [s.phrase for s in suggestions]
                try:
                    actual_rank = phrases.index(tgt_phrase) + 1
                except ValueError:
                    actual_rank = float("nan")
                if callback is not None:
                    callback(Attempt(index, src_phrase, tgt_phrase, actual_rank, len(phrases), tuple(phrases[:ATTEMPTS_TOP_K]), nodes))
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
        for index, (src_phrase, tgt_phrase, suggestions, _) in enumerate(self._iterate_guesses(dataset, limit=limit)):
            if index > 0:
                print()
            print(f"{index: 2d} src: {src_phrase}")
            print(f"{index: 2d} tgt: {tgt_phrase}")
            print(f"{index: 2d} xxx: {suggestions[0].phrase}")

    def _iterate_guesses(self, dataset: PhrasePairDataset, limit: Optional[int] = None, guesses_per_phrase: Optional[int] = None) -> Iterator[Tuple[str, str, list[Suggestion], Optional[list[Node]]]]:
        translator = Translator(self.model, self.bilinguist, self.device)
        for index, (src_phrase, tgt_phrase) in enumerate(dataset):
            if limit is not None and index >= limit:
                break
            src_phrase = self.src_transform(src_phrase)
            tgt_phrase = self.tgt_transform(tgt_phrase)
            nodes = None
            if guesses_per_phrase is None:
                translation = translator.translate(src_phrase).strip()
                translation = self.tgt_transform(translation)
                suggestions = [Suggestion(translation, float("nan"))]
            else:
                def _set_nodes(nodes_):
                    nonlocal nodes
                    nodes = nodes_
                suggestions = translator.suggest(src_phrase, count=guesses_per_phrase, navigator=self.node_navigator, nodes_callback=_set_nodes)
                if not self.tgt_transform is dlfp.common.identity:
                    suggestions = [Suggestion(self.tgt_transform(s.phrase), s.probability) for s in suggestions]
            yield src_phrase, tgt_phrase, suggestions, nodes


    def train(self, loaders: TrainLoaders, checkpoints_dir: Path, train_config: 'TrainConfig'):
        trainer = Trainer(self.model, pad_idx=self.bilinguist.source.specials.indexes.pad, device=self.device)
        trainer.optimizer_factory = train_config.train_hp.create_optimizer
        print(f"writing checkpoints to {checkpoints_dir}")
        checkpointer = Checkpointer(checkpoints_dir)
        checkpointer.retain_all = train_config.retain_all_checkpoints
        train_config_dict = train_config.to_jsonable()
        checkpointer.extra = {
            "train_config": train_config_dict,
        }
        with dlfp.common.open_write(checkpoints_dir / "train-config.json") as ofile:
            json.dump(train_config_dict, ofile, indent=2)
        results = trainer.train(loaders, train_config.train_hp.epoch_count, callback=checkpointer.checkpoint)
        results_table = Table([r.to_row() for r in results], EpochResult.headers())
        results_table.write(fmt="simple_grid")


class DataSuperset(NamedTuple):

    train: PhrasePairDataset
    valid: PhrasePairDataset


class TrainHyperparametry(NamedTuple):

    epoch_count: int = 10
    batch_size: int = 128
    lr: float = 0.0001
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-9
    train_data_shuffle_disabled: bool = False

    def create_optimizer(self, model: Module) -> Optimizer:
        return torch.optim.Adam(model.parameters(), lr=self.lr, betas=self.betas, eps=self.eps)

    @staticmethod
    def from_args(arguments: Optional[list[str]]) -> 'TrainHyperparametry':
        types = {
            'epoch_count': int,
            'batch_size': int,
            'betas': lambda s: tuple(float(b) for b in s.split(',')),
            'train_data_shuffle_disabled': int,
        }
        return dlfp.common.nt_from_args(TrainHyperparametry, arguments, types)


class TrainConfig(NamedTuple):

    dataset_name: str
    checkpoints_dir: Path
    train_hp: TrainHyperparametry
    model_hp: ModelHyperparametry
    retain_all_checkpoints: bool = False
    save_optimizer: bool = False

    def to_jsonable(self) -> dict[str, Optional[str]]:
        def _xform(value):
            if hasattr(value, "_asdict"):
                # noinspection PyProtectedMember
                return value._asdict()
            if value is None:
                return value
            if isinstance(value, (int, float, bool, str)):
                return value
            return str(value)
        return {k:_xform(v) for k, v in self._asdict().items()}


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

    # noinspection PyMethodMayBeStatic
    def create_model(self, bilinguist: Bilinguist, h: ModelHyperparametry) -> Seq2SeqTransformer:
        model = dlfp.models.create_model(
            src_vocab_size=len(bilinguist.source.vocab),
            tgt_vocab_size=len(bilinguist.target.vocab),
            h=h,
        )
        return model

    def run_train(self, train_config: TrainConfig, device: str):
        r = self.create_runnable(train_config.dataset_name, train_config.model_hp, device)
        loaders = TrainLoaders.from_datasets(
            r.superset.train,
            r.superset.valid,
            collate_fn=r.bilinguist.collate,
            batch_size=train_config.train_hp.batch_size,
            train_shuffle=not train_config.train_hp.train_data_shuffle_disabled,
        )
        r.manager.train(loaders, train_config.checkpoints_dir, train_config)

    def create_runnable(self, dataset_name: str, h: ModelHyperparametry, device: str) -> Runnable:
        superset = self.resolve_dataset(dataset_name)
        bilinguist = self.create_bilinguist(superset)
        model = self.create_model(bilinguist, h)
        model = model.to(device)
        model_manager = ModelManager(model, bilinguist, device)
        return Runnable(superset, bilinguist, model_manager)

    def run_eval(self,
                 restored: Restored,
                 dataset_name: str,
                 h: ModelHyperparametry,
                 device: str,
                 output_file: Path,
                 split: Split = "valid",
                 concurrency: Optional[int] = None,
                 nodes_folder: Optional[Path] = None):
        r = self.create_runnable(dataset_name, h, device)
        r.manager.model.load_state_dict(restored.model_state_dict)
        dataset = getattr(r.superset, split)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        completed = False
        attempt_q = Queue()
        def write_csv():
            with open(output_file, "w", newline="", encoding="utf-8") as ofile:
                csv_writer = csv.writer(ofile)
                csv_writer.writerow(Attempt.headers(ATTEMPTS_TOP_K))
                attempt_index = 0
                while (not completed) or attempt_q:
                    try:
                        attempt_: Attempt = attempt_q.get(timeout=0.25)
                        csv_writer.writerow(attempt_.to_row())
                        if nodes_folder is not None:
                            self.write_nodes(nodes_folder, attempt_, attempt_index, r.bilinguist.target.vocab)
                        attempt_index += 1
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
        table.write()

    def write_nodes(self, nodes_folder: Path, attempt: Attempt, attempt_index: int, target_vocab: Vocab):
        answer = dlfp.utils.normalize_answer_upper(attempt.target)
        filename = f"{attempt_index:06d}-{answer}.csv"
        with dlfp.common.open_write(nodes_folder / filename, newline="") as ofile:
            csv_writer = csv.writer(ofile)
            csv_writer.writerow(["cumu_prob", "word", "prob"])
            for node in attempt.nodes:
                lineage = node.lineage()
                row = [node.cumulative_probability()]
                for n in lineage:
                    row.append(n.current_word_token(target_vocab))
                    row.append(n.prob)
            csv_writer.writerow(row)

    def run_demo(self, restored: Restored, dataset_name: str, h: ModelHyperparametry, device: str, limit: int = ...):
        r = self.create_runnable(dataset_name, h, device)
        if limit is ...:
            limit = 5
        if limit is None:
            limit = len(r.superset.valid)
        r.manager.model.load_state_dict(restored.model_state_dict)
        r.manager.print_translations(r.superset.valid, limit=limit)


def main(runner: Runner) -> int:
    parser = ArgumentParser(description=runner.describe(), epilog=f"""\
Allowed --train-param keys are: {TrainHyperparametry._fields}.
Allowed --model-param keys are: {ModelHyperparametry._fields}.\
""")
    parser.add_argument("-m", "--mode", metavar="MODE", choices=("train", "eval", "demo"), default="train", help="'train', 'eval', or 'demo' (print some outputs)")
    parser.add_argument("-o", "--output", metavar="DIR", type=Path, help="output root directory")
    parser.add_argument("-f", "--file", metavar="FILE", help="checkpoint file for eval mode")
    parser.add_argument("-t", "--train-param", metavar="K=V", action='append', help="set training hyperparameter")
    parser.add_argument("-p", "--model-param", metavar="K=V", action='append', help="set model hyperparameter")
    parser.add_argument("--limit", type=int, default=..., metavar="N", help="demo mode phrase limit")
    parser.add_argument("-d", "--dataset", metavar="NAME", help="specify dataset name")
    split_choices = ("train", "valid", "test")
    parser.add_argument("-s", "--split", metavar="SPLIT", choices=split_choices, help="eval mode dataset split")
    parser.add_argument("--concurrency", type=int, help="eval mode concurrency")
    parser.add_argument("--retain", action='store_true', help="train mode: retain all model checkpoints (instead of deleting obsolete)")
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    seed = 0
    torch.manual_seed(seed)
    runner.dataset_name = args.dataset
    timestamp = dlfp.common.timestamp()
    model_hp = ModelHyperparametry.from_args(args.model_param)
    if args.mode == "demo":
        checkpoint_file = args.file
        if not checkpoint_file:
            parser.error("checkpoint file must be specified")
            return 1
        restored = Restored.from_file(checkpoint_file, device=device)
        runner.run_demo(restored, args.dataset, model_hp, device, args.limit)
        return 0
    elif args.mode == "eval":
        checkpoint_file = args.file
        if not checkpoint_file:
            parser.error("checkpoint file must be specified")
            return 1
        checkpoint_file = Path(checkpoint_file)
        restored = Restored.from_file(checkpoint_file, device=device)
        split = args.split or "valid"
        output_file = (args.output or Path.cwd()) / "evaluations" / f"{checkpoint_file.stem}_{split}_{timestamp}.csv"
        runner.run_eval(restored, args.dataset, model_hp, device, output_file, split=split, concurrency=args.concurrency)
        return 0
    elif args.mode == "train":
        checkpoints_dir = Path(args.output or ".") / f"checkpoints/{timestamp}"
        train_hp = TrainHyperparametry.from_args(args.train_param)
        train_config = TrainConfig(args.dataset, checkpoints_dir, train_hp, model_hp, retain_all_checkpoints=args.retain)
        print(json.dumps(train_config.to_jsonable(), indent=2))
        runner.run_train(train_config, device)
        return 0
    else:
        parser.error("BUG unhandled mode")
        return 2
