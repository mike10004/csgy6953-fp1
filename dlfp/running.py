#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
from typing import NamedTuple

import torch
import tabulate

import dlfp.utils
from dlfp.models import Seq2SeqTransformer
from dlfp.tokens import Biglot
from dlfp.train import TrainLoaders
from dlfp.train import Trainer
from dlfp.train import create_model
from dlfp.translate import Translator
from dlfp.utils import Checkpointer
from dlfp.utils import EpochResult
from dlfp.utils import PhrasePairDataset
from dlfp.utils import Restored


class ModelManager:

    def __init__(self, model: Seq2SeqTransformer, biglot: Biglot, device):
        self.device = device
        self.model = model
        self.biglot = biglot
        self.device = device

    def print_translations(self, dataset: PhrasePairDataset, limit):
        translator = Translator(self.model, self.biglot, self.device)
        for index, (de_phrase, en_phrase) in enumerate(dataset):
            if index >= limit:
                break
            if index > 0:
                print()
            print(f"{index: 2d} de: {de_phrase}")
            print(f"{index: 2d} en: {en_phrase}")
            translation = translator.translate(de_phrase).strip()
            print(f"{index: 2d} mx: {translation}")


    def train(self, loaders: TrainLoaders, checkpoints_dir: Path, epoch_count: int = 10):
        trainer = Trainer(self.model, pad_idx=self.biglot.source.language.specials.indexes.pad, device=self.device)
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
    biglot: Biglot
    manager: ModelManager


class Runner:

    def describe(self) -> str:
        raise NotImplementedError("abstract")

    def resolve_dataset(self) -> DataSuperset:
        raise NotImplementedError("abstract")

    def create_biglot(self, superset: DataSuperset):
        raise NotImplementedError("abstract")

    def create_model(self, biglot: Biglot, device: str):
        model = create_model(
            src_vocab_size=len(biglot.source.language.vocab),
            tgt_vocab_size=len(biglot.target.language.vocab),
            DEVICE=device,
        )
        return model

    def run_train(self, train_config: TrainConfig, device: str):
        r = self.create_runnable(device)
        loaders = TrainLoaders.from_datasets(r.superset.train, r.superset.valid, collate_fn=r.biglot.collate, batch_size=train_config.batch_size)
        r.manager.train(loaders, train_config.checkpoints_dir, train_config.epoch_count)

    def create_runnable(self, device: str) -> Runnable:
        superset = self.resolve_dataset()
        biglot = self.create_biglot(superset)
        model = self.create_model(biglot, device)
        model_manager = ModelManager(model, biglot, device)
        return Runnable(superset, biglot, model_manager)

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
    parser.add_argument("-m", "--mode", choices=("train", "eval"), default="train")
    parser.add_argument("-o", "--output", metavar="DIR", help="output root directory")
    parser.add_argument("-f", "--file", metavar="FILE", help="checkpoint file for eval mode")
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
        train_config = TrainConfig(checkpoints_dir)
        runner.run_train(train_config, device)
        return 0
    else:
        parser.error("BUG unhandled mode")
        return 2
