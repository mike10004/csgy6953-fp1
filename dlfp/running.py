#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
from typing import Callable
from typing import Iterator
from typing import NamedTuple
from typing import Optional
from typing import Tuple

import torch
import tabulate

import dlfp.utils
from dlfp.models import Seq2SeqTransformer
from dlfp.utils import Bilinguist
from dlfp.train import TrainLoaders
from dlfp.train import Trainer
from dlfp.train import create_model
from dlfp.translate import Translator
from dlfp.utils import Checkpointer
from dlfp.utils import EpochResult
from dlfp.utils import PhrasePairDataset
from dlfp.utils import Restored


StringTransform = Callable[[str], str]

class ModelManager:

    def __init__(self, model: Seq2SeqTransformer, bilinguist: Bilinguist, device):
        self.device = device
        self.model = model
        self.bilinguist = bilinguist
        self.device = device
        self.src_transform: StringTransform = dlfp.utils.identity
        self.tgt_transform: StringTransform = dlfp.utils.identity

    # def evaluate_accuracy(self, dataset: PhrasePairDataset, ranks: Sequence[int] = (1, 10, 100, 1000)):
    #     ranks = sorted(ranks, reverse=True)
    #     rank_acc = defaultdict(int)
    #     count = 0
    #     for index, (src_phrase, tgt_phrase, translation) in enumerate(self._iterate_guesses(dataset)):
    #         count += 1
    #         for rank in ranks:

    def print_translations(self, dataset: PhrasePairDataset, limit: int):
        for index, (src_phrase, tgt_phrase, translation) in enumerate(self._iterate_guesses(dataset, limit=limit)):
            if index > 0:
                print()
            print(f"{index: 2d} src: {src_phrase}")
            print(f"{index: 2d} tgt: {tgt_phrase}")
            print(f"{index: 2d} xxx: {translation}")

    def _iterate_guesses(self, dataset: PhrasePairDataset, limit: Optional[int] = None) -> Iterator[Tuple[str, str, str]]:
        translator = Translator(self.model, self.bilinguist, self.device)
        for index, (src_phrase, tgt_phrase) in enumerate(dataset):
            if limit is not None and index >= limit:
                break
            src_phrase = self.src_transform(src_phrase)
            tgt_phrase = self.tgt_transform(tgt_phrase)
            translation = translator.translate(src_phrase).strip()
            translation = self.tgt_transform(translation)
            yield src_phrase, tgt_phrase, translation


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

    def resolve_dataset(self) -> DataSuperset:
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
        superset = self.resolve_dataset()
        bilinguist = self.create_bilinguist(superset)
        model = self.create_model(bilinguist, device)
        model_manager = ModelManager(model, bilinguist, device)
        return Runnable(superset, bilinguist, model_manager)

    def run_eval(self, restored: Restored, device: str, limit: int = ...):
        r = self.create_runnable(device)
        if limit is ...:
            limit = 5
        if limit is None:
            limit = len(r.superset.valid)
        r.manager.model.load_state_dict(restored.model_state_dict)
        r.manager.print_translations(r.superset.valid, limit=limit)


def main(runner: Runner) -> int:
    parser = ArgumentParser(description=runner.describe())
    parser.add_argument("-m", "--mode", metavar="MODE", choices=("train", "eval"), default="train", help="'train' or 'eval'")
    parser.add_argument("-o", "--output", metavar="DIR", help="output root directory")
    parser.add_argument("-f", "--file", metavar="FILE", help="checkpoint file for eval mode")
    parser.add_argument("-e", "--epoch-count", type=int, default=10, metavar="N", help="epoch count")
    parser.add_argument("-b", "--batch-size", type=int, default=128, metavar="N", help="training batch size")
    parser.add_argument("--limit", type=int, default=..., metavar="N", help="eval mode phrase limit")
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    seed = 0
    torch.manual_seed(seed)
    if args.mode == "eval":
        checkpoint_file = args.file
        if not checkpoint_file:
            parser.error("checkpoint file must be specified")
            return 1
        restored = Restored.from_file(checkpoint_file, device=device)
        runner.run_eval(restored, device, args.limit)
        return 0
    elif args.mode == "train":
        checkpoints_dir = Path(args.output or ".") / f"checkpoints/{dlfp.utils.timestamp()}"
        train_config = TrainConfig(checkpoints_dir, epoch_count=args.epoch_count, batch_size=args.batch_size)
        runner.run_train(train_config, device)
        return 0
    else:
        parser.error("BUG unhandled mode")
        return 2
