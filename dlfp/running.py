#!/usr/bin/env python3
import contextlib
import csv
import sys
from argparse import ArgumentParser
from collections import defaultdict
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import RLock
from typing import Any
from typing import Callable
from typing import Collection
from typing import Iterator
from typing import NamedTuple
from typing import Optional
from typing import Tuple

import torch
import tabulate
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
                 ranks: Collection[int] = (1, 10, 100, 1000),
                 callback: Callable[[Attempt], None] = None,
                 hide_progress: bool = False,
                 concurrency: int = None,
                 top_k: int = 1):
        ranks = sorted(ranks, reverse=True)
        progress_bar = tqdm(file=sys.stdout, total=len(dataset), disable=hide_progress)
        def perform(dataset_part: PhrasePairDataset) -> dict[int, int]:
            rank_acc = defaultdict(int)
            for index, (src_phrase, tgt_phrase, suggestions) in enumerate(self._iterate_guesses(dataset_part, limit=None, guesses_per_phrase=max(ranks))):
                phrases = [s.phrase for s in suggestions]
                try:
                    actual_rank = phrases.index(tgt_phrase) + 1
                except ValueError:
                    actual_rank = float("nan")
                for rank in ranks:
                    if actual_rank <= rank:
                        rank_acc[rank] += 1
                if callback is not None:
                    callback(Attempt(index, src_phrase, tgt_phrase, actual_rank, len(phrases), tuple(phrases[:top_k])))
                progress_bar.update(1)
            return dict(rank_acc)
        if concurrency is not None:
            partitioning = dataset.partition(concurrency)
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures: list[Future] = []
                for partitioning_item in partitioning:
                    partitioning_item: PhrasePairDataset
                    future = executor.submit(perform, partitioning_item)
                    futures.append(future)
            rank_accs = [future.result() for future in futures]
            total_rank_acc = {rank: 0 for rank in ranks}
            for rank_acc_part in rank_accs:
                for rank in ranks:
                    total_rank_acc[rank] += rank_acc_part[rank]
        else:
            total_rank_acc = perform(dataset)
        return total_rank_acc

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


    def train(self, loaders: TrainLoaders, checkpoints_dir: Path, epoch_count: int = 10):
        trainer = Trainer(self.model, pad_idx=self.bilinguist.source.specials.indexes.pad, device=self.device)
        print(f"writing checkpoints to {checkpoints_dir}")
        checkpointer = Checkpointer(checkpoints_dir, self.model)
        results = trainer.train(loaders, epoch_count, callback=checkpointer.checkpoint)
        results_table = [
            (r.epoch_index, r.train_loss, r.valid_loss)
            for r in results
        ]
        print(tabulate.tabulate(results_table, headers=EpochResult._fields))


class DataSuperset(NamedTuple):

    train: PhrasePairDataset
    valid: PhrasePairDataset


class TrainConfig(NamedTuple):

    checkpoints_dir: Path
    epoch_count: int = 10
    batch_size: int = 128


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
        r = self.create_runnable(device)
        loaders = TrainLoaders.from_datasets(r.superset.train, r.superset.valid, collate_fn=r.bilinguist.collate, batch_size=train_config.batch_size)
        r.manager.train(loaders, train_config.checkpoints_dir, train_config.epoch_count)

    def create_runnable(self, device: str) -> Runnable:
        dataset_name = getattr(self, "dataset_name", None)
        superset = self.resolve_dataset(dataset_name)
        bilinguist = self.create_bilinguist(superset)
        model = self.create_model(bilinguist, device)
        model_manager = ModelManager(model, bilinguist, device)
        return Runnable(superset, bilinguist, model_manager)

    def run_eval(self, restored: Restored, device: str, output_file: Path, split: Split = "valid", concurrency: Optional[int] = None):
        r = self.create_runnable(device)
        r.manager.model.load_state_dict(restored.model_state_dict)
        dataset = getattr(r.superset, split)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        write_lock = RLock()
        @contextlib.contextmanager
        def maybe_lock():
            if concurrency is not None and concurrency > 1:
                with write_lock:
                    yield
            else:
                yield
        with open(output_file, "w", newline="", encoding="utf-8") as ofile:
            csv_writer = csv.writer(ofile)
            csv_writer.writerow(Attempt.headers(1))
            def callback(attempt: Attempt):
                with maybe_lock():
                    csv_writer.writerow(attempt.to_row())
            evaluation = r.manager.evaluate(dataset, callback=callback, concurrency=concurrency)
        print("split:", split)
        for rank, accuracy_count in evaluation.items():
            accuracy = accuracy_count / len(dataset)
            print(f"{rank:5d}: {accuracy*100:.4f}%")

    def run_demo(self, restored: Restored, device: str, limit: int = ...):
        r = self.create_runnable(device)
        if limit is ...:
            limit = 5
        if limit is None:
            limit = len(r.superset.valid)
        r.manager.model.load_state_dict(restored.model_state_dict)
        r.manager.print_translations(r.superset.valid, limit=limit)


def main(runner: Runner) -> int:
    parser = ArgumentParser(description=runner.describe())
    parser.add_argument("-m", "--mode", metavar="MODE", choices=("train", "eval", "demo"), default="train", help="'train', 'eval', or 'demo' (print some outputs)")
    parser.add_argument("-o", "--output", metavar="DIR", help="output root directory")
    parser.add_argument("-f", "--file", metavar="FILE", help="checkpoint file for eval mode")
    parser.add_argument("-e", "--epoch-count", type=int, default=10, metavar="N", help="epoch count")
    parser.add_argument("-b", "--batch-size", type=int, default=128, metavar="N", help="training batch size")
    parser.add_argument("--limit", type=int, default=..., metavar="N", help="demo mode phrase limit")
    parser.add_argument("-d", "--dataset", metavar="NAME", help="specify dataset name")
    split_choices = ("train", "valid", "test")
    parser.add_argument("-s", "--split", metavar="SPLIT", choices=split_choices, help="eval mode dataset split")
    parser.add_argument("--concurrency", type=int)
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
        runner.run_demo(restored, device, args.limit)
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
        runner.run_eval(restored, device, output_file, split=split, concurrency=args.concurrency)
        return 0
    elif args.mode == "train":
        checkpoints_dir = Path(args.output or ".") / f"checkpoints/{dlfp.utils.timestamp()}"
        train_config = TrainConfig(checkpoints_dir, epoch_count=args.epoch_count, batch_size=args.batch_size)
        runner.run_train(train_config, device)
        return 0
    else:
        parser.error("BUG unhandled mode")
        return 2
