#!/usr/bin/env python3

import csv
import json
import sys
import queue
import logging
from pathlib import Path
from queue import Queue
from random import Random
from typing import Any
from typing import Callable
from typing import Iterator
from typing import NamedTuple
from typing import Optional
from threading import Thread
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

import torch
from tqdm import tqdm

import dlfp.utils
import dlfp.common
import dlfp.models
import dlfp.translate
from dlfp.datasets import DatasetResolver
from dlfp.models import Seq2SeqTransformer
from dlfp.results import AccuracyResult
from dlfp.translate import Node
from dlfp.translate import NodeNavigator
from dlfp.translate import Suggestion
from dlfp.utils import Bilinguist
from dlfp.train import TrainLoaders
from dlfp.train import Trainer
from dlfp.models import ModelHyperparametry
from dlfp.models import TrainHyperparametry
from dlfp.translate import Translator
from dlfp.utils import Checkpointer
from dlfp.utils import EpochResult
from dlfp.utils import PhrasePairDataset
from dlfp.utils import Restored
from dlfp.utils import Split
from dlfp.common import Table
from dlfp.results import measure_accuracy
from dlfp.results import DEFAULT_RANKS
from dlfp.translate import Attempt

StringTransform = Callable[[str], str]
ATTEMPTS_TOP_K = 10
_log = logging.getLogger(__name__)

class GuessResult(NamedTuple):

    src_phrase: str
    tgt_phrase: str
    suggestions: list[Suggestion]
    nodes: Optional[list[Node]]


class ModelManager:

    def __init__(self, model: Seq2SeqTransformer, bilinguist: Bilinguist, device: str):
        self.device = device
        self.model = model
        self.bilinguist = bilinguist
        self.device = device
        self.src_transform: StringTransform = dlfp.common.identity
        self.tgt_transform: StringTransform = dlfp.common.identity

    def evaluate(self,
                 dataset: PhrasePairDataset,
                 suggestion_count: int,
                 callback: Callable[[Attempt], None],
                 node_navigator: NodeNavigator,
                 hide_progress: bool = False,
                 concurrency: int = None):
        progress_bar = tqdm(file=sys.stdout, total=len(dataset), disable=hide_progress)
        def perform(dataset_part: PhrasePairDataset):
            _log.debug("evaluating dataset of size %s", len(dataset_part))
            for index, (src_phrase, tgt_phrase, suggestions, nodes) in enumerate(self._iterate_guesses(dataset_part, node_navigator=node_navigator, limit=None, guesses_per_phrase=suggestion_count)):
                phrases = [s.phrase for s in suggestions]
                try:
                    actual_rank = phrases.index(tgt_phrase) + 1
                except ValueError:
                    actual_rank = float("nan")
                if callback is not None:
                    _log.debug("invoking callback with attempt %s", index)
                    callback(Attempt(index, src_phrase, tgt_phrase, actual_rank, len(phrases), tuple(phrases[:ATTEMPTS_TOP_K]), nodes))
                progress_bar.update(1)
            _log.debug("evaluate: perform: finished")
        try:
            if concurrency is not None:
                partitioning = dataset.partition(concurrency)
                with ThreadPoolExecutor(max_workers=concurrency) as executor:
                    for partitioning_item in partitioning:
                        partitioning_item: PhrasePairDataset
                        executor.submit(perform, partitioning_item)
                    _log.debug("awaiting thread pool executor %s works to finish", len(partitioning))
            else:
                perform(dataset)
        finally:
            _log.debug("ModelManager.completed = True")

    def print_translations(self,
                           dataset: PhrasePairDataset,
                           limit: int,
                           node_navigator: NodeNavigator):
        for index, (src_phrase, tgt_phrase, suggestions, _) in enumerate(self._iterate_guesses(dataset, node_navigator=node_navigator, limit=limit)):
            if index > 0:
                print()
            print(f"{index: 2d} src: {src_phrase}")
            print(f"{index: 2d} tgt: {tgt_phrase}")
            print(f"{index: 2d} xxx: {suggestions[0].phrase}")

    def _iterate_guesses(self,
                         dataset: PhrasePairDataset,
                         node_navigator: NodeNavigator,
                         limit: Optional[int] = None,
                         guesses_per_phrase: Optional[int] = None) -> Iterator[GuessResult]:
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
                suggestions = translator.suggest(src_phrase, count=guesses_per_phrase, navigator=node_navigator, nodes_callback=_set_nodes)
                if not self.tgt_transform is dlfp.common.identity:
                    suggestions = [Suggestion(self.tgt_transform(s.phrase), s.probability) for s in suggestions]
            yield GuessResult(src_phrase, tgt_phrase, suggestions, nodes)


    def train(self, loaders: TrainLoaders, checkpoints_dir: Path, train_config: 'TrainConfig', metadata: dict[str, Any]):
        trainer = Trainer(self.model, pad_idx=self.bilinguist.source.specials.indexes.pad, device=self.device)
        trainer.optimizer_factory = train_config.train_hp.create_optimizer
        print(f"writing checkpoints to {checkpoints_dir}")
        checkpointer = Checkpointer(checkpoints_dir)
        checkpointer.retain_all = train_config.retain_all_checkpoints
        train_config_dict = train_config.to_jsonable()
        checkpointer.extra = {
            "train_config": train_config_dict,
            "metadata": metadata,
        }
        with dlfp.common.open_write(checkpoints_dir / "train-config.json") as ofile:
            json.dump(train_config_dict, ofile, indent=2)
        results = trainer.train(loaders, train_config.train_hp.epoch_count, callback=checkpointer.checkpoint)
        results_table = Table([r.to_row() for r in results], EpochResult.headers())
        results_table.write(fmt="simple_grid")


class DataSuperset(NamedTuple):

    train: PhrasePairDataset
    valid: PhrasePairDataset


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
    metadata: Optional[dict[str, Any]] = None


class EvalConfig(NamedTuple):

    split: Split = "valid"
    concurrency: Optional[int] = None
    nodes_folder: Optional[Path] = None
    limit: Optional[int] = None
    shuffle_seed: Optional[int] = None
    navigator: Optional[str] = None

    @staticmethod
    def from_args(arguments: Optional[list[str]]) -> 'EvalConfig':
        types = {
            'split': str,
            'nodes_folder': Path,
            'navigator': str,
        }
        return dlfp.common.nt_from_args(EvalConfig, arguments, types=types, default_type=int)


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
        r.manager.train(loaders, train_config.checkpoints_dir, train_config, r.metadata)

    def create_runnable(self, dataset_name: str, h: ModelHyperparametry, device: str) -> Runnable:
        superset = self.resolve_dataset(dataset_name)
        bilinguist = self.create_bilinguist(superset)
        model = self.create_model(bilinguist, h)
        model = model.to(device)
        model_manager = ModelManager(model, bilinguist, device)
        metadata = {
            "dataset_name": superset.train.name,
            "src_vocab_size": len(bilinguist.source.vocab),
            "tgt_vocab_size": len(bilinguist.target.vocab),
        }
        return Runnable(superset, bilinguist, model_manager, metadata)

    def create_navigator(self, navigator_spec: str, dataset_name: str) -> NodeNavigator:
        raise NotImplementedError("abstract")

    @staticmethod
    def _prepare_dataset(dataset: PhrasePairDataset,
                         limit: Optional[int],
                         shuffle_seed: Optional[int]) -> PhrasePairDataset:
        if limit is not None:
            rng = None
            if shuffle_seed is None or shuffle_seed >= 0:
                rng = Random(shuffle_seed)
            if rng is not None:
                dataset = dataset.shuffle(rng)
            dataset = dataset.slice(0, limit)
        return dataset

    def run_eval(self,
                 restored: Restored,
                 dataset_name: str,
                 h: ModelHyperparametry,
                 device: str,
                 output_file: Path,
                 eval_config: EvalConfig) -> AccuracyResult:
        r = self.create_runnable(dataset_name, h, device)
        r.manager.model.load_state_dict(restored.model_state_dict)
        dataset = DatasetResolver().by_name(dataset_name or r.superset.train.name, eval_config.split)
        dataset = self._prepare_dataset(dataset, limit=eval_config.limit, shuffle_seed=eval_config.shuffle_seed)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        nodes_folder = eval_config.nodes_folder
        if nodes_folder is not None:
            if str(nodes_folder) == "auto":
                nodes_folder = output_file.parent / f"{output_file.stem}-nodes"
        completed = False
        attempt_q = Queue()
        def write_csv():
            _log.debug("write attempts csv file starting: %s", output_file.as_posix())
            with open(output_file, "w", newline="", encoding="utf-8") as ofile:
                csv_writer = csv.writer(ofile)
                csv_writer.writerow(Attempt.headers(ATTEMPTS_TOP_K))
                def _process(a: Attempt):
                    nonlocal nodes_folder
                    csv_writer.writerow(a.to_row())
                    if nodes_folder is not None:
                        dlfp.translate.write_nodes(nodes_folder, a, r.bilinguist.target.vocab, r.bilinguist.target.specials)
                while not completed:
                    try:
                        attempt_: Attempt = attempt_q.get(timeout=0.25)
                        _process(attempt_)
                    except queue.Empty:
                        pass
                _log.debug("queue loop terminated: attempt_q.qsize=%s", attempt_q.qsize())
                # concurrent operation could result in completed=True but queue still has items
                while attempt_q.qsize() > 0:
                    try:
                        _process(attempt_q.get(block=False))
                    except queue.Empty:
                        break
        csv_thread = Thread(target=write_csv, daemon=True)
        csv_thread.start()
        print(f"split: {eval_config.split} (limit: {eval_config.limit})")
        try:
            r.manager.evaluate(dataset,
                               max(DEFAULT_RANKS),
                               callback=attempt_q.put,
                               concurrency=eval_config.concurrency,
                               node_navigator=self.create_navigator(eval_config.navigator, dataset.name))
        finally:
            completed = True
            print("awaiting csv writes complete...", end="")
            csv_thread.join()
            print("done")
        result = measure_accuracy(output_file, DEFAULT_RANKS)
        table = result.to_table()
        table.write()
        return result

    def run_demo(self, restored: Restored, dataset_name: str, h: ModelHyperparametry, device: str, limit: int = ...):
        r = self.create_runnable(dataset_name, h, device)
        if limit is ...:
            limit = 5
        if limit is None:
            limit = len(r.superset.valid)
        r.manager.model.load_state_dict(restored.model_state_dict)
        node_navigator = self.create_navigator("default", dataset_name)
        r.manager.print_translations(r.superset.valid, node_navigator=node_navigator, limit=limit)


def get_model_hyperparametry(restored: Restored) -> ModelHyperparametry:
    ok, _, model_hp = dlfp.models.get_hyperparameters(restored)
    if not ok:
        raise ValueError("model hyperparameters not available in restored checkpoint")
    return model_hp

def main(runner: Runner) -> int:
    parser = ArgumentParser(description=runner.describe(), epilog=f"""\
Allowed --train-param keys are: {TrainHyperparametry._fields}.
Allowed --model-param keys are: {ModelHyperparametry._fields}.\
""")
    parser.add_argument("-m", "--mode", metavar="MODE", choices=("train", "eval", "demo"), default="train", help="'train', 'eval', or 'demo' (print some outputs)")
    parser.add_argument("-o", "--output", metavar="PATH", type=Path, help="output file or root directory")
    parser.add_argument("-f", "--file", metavar="FILE", help="checkpoint file for eval mode")
    parser.add_argument("-t", "--train-param", metavar="K=V", action='append', help="set training hyperparameter")
    parser.add_argument("-p", "--model-param", metavar="K=V", action='append', help="set model hyperparameter")
    parser.add_argument("--limit", type=int, default=..., metavar="N", help="demo mode phrase limit")
    parser.add_argument("-d", "--dataset", metavar="NAME", help="specify dataset name")
    parser.add_argument("--retain", action='store_true', help="train mode: retain all model checkpoints (instead of deleting obsolete)")
    parser.add_argument("--optimizer", action='store_true', help="train mode: save optimizer state in checkpoint")
    parser.add_argument("-e", "--eval-config", metavar="K=V", action='append', help=f"set eval mode option; keys are {EvalConfig._fields}")
    args = parser.parse_args()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    seed = 0
    torch.manual_seed(seed)
    runner.dataset_name = args.dataset
    timestamp = dlfp.common.timestamp()
    try:
        if args.mode == "demo":
            checkpoint_file = args.file
            if not checkpoint_file:
                parser.error("checkpoint file must be specified")
                return 1
            restored = Restored.from_file(checkpoint_file, device=device)
            model_hp = get_model_hyperparametry(restored)
            runner.run_demo(restored, args.dataset, model_hp, device, args.limit)
            return 0
        elif args.mode == "eval":
            checkpoint_file = args.file
            if not checkpoint_file:
                parser.error("checkpoint file must be specified")
                return 1
            checkpoint_file = Path(checkpoint_file)
            restored = Restored.from_file(checkpoint_file, device=device)
            eval_config = EvalConfig.from_args(args.eval_config)
            output_file = args.output or (checkpoint_file.parent / "evaluations" / f"{checkpoint_file.stem}_{eval_config.split}_{timestamp}.csv")
            model_hp = get_model_hyperparametry(restored)
            runner.run_eval(restored,
                            args.dataset,
                            model_hp,
                            device,
                            output_file,
                            eval_config=eval_config)
            return 0
        elif args.mode == "train":
            model_hp = ModelHyperparametry.from_args(args.model_param)
            checkpoints_dir = Path(args.output or ".") / f"checkpoints/{timestamp}"
            train_hp = TrainHyperparametry.from_args(args.train_param)
            train_config = TrainConfig(args.dataset, checkpoints_dir, train_hp, model_hp, retain_all_checkpoints=args.retain, save_optimizer=args.optimizer)
            print(json.dumps(train_config.to_jsonable(), indent=2))
            runner.run_train(train_config, device)
            return 0
        else:
            parser.error("BUG unhandled mode")
            return 2
    except KeyboardInterrupt:
        print("dlfp: terminating due to keyboard interrupt", file=sys.stderr)
        return 3
