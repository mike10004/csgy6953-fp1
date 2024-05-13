#!/usr/bin/env python3

import os
import re
import sys
import csv
import json
import glob
import shutil
import fnmatch
import logging
from collections import defaultdict
from pathlib import Path
from argparse import ArgumentParser
from typing import Collection
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import NamedTuple
from typing import Optional
from typing import Sequence

import numpy as np

import dlfp.common
from dlfp.common import Table
from dlfp.models import ModelHyperparametry
from dlfp.models import TrainHyperparametry
from dlfp.utils import EpochResult
from dlfp.utils import Restored
from dlfp.utils import EvalConfig

_log = logging.getLogger(__name__)
DEFAULT_RANKS = (1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
HYPERPARAMETER_ABBREVIATIONS = {
    "transformer_dropout_rate": "tdr",
    "input_dropout_rate": "idr",
    "dataset_name": "dataset",
    "src_tok_emb.embedding.weight": "s_emb_sz",
    "tgt_tok_emb.embedding.weight": "t_emb_sz",
    "dim_feedforward": "ffnd",
    "tgt_pos_enc_disabled": "tped",
}


class AccuracyResult(NamedTuple):

    rank_acc_count: dict[int, int]
    attempt_count: int

    @staticmethod
    def table_headers() -> list[Any]:
        return ["rank", "count", "percent"]

    def to_table(self) -> Table:
        ranks = sorted(self.rank_acc_count.keys())
        table = []
        for rank in ranks:
            count = self.rank_acc_count[rank]
            proportion = count / self.attempt_count
            table.append([rank, count, proportion * 100.0])
        return Table(table, self.table_headers())


def measure_accuracy(attempt_file: Path, ranks: Collection[int] = None) -> AccuracyResult:
    ranks = ranks or DEFAULT_RANKS
    ranks = list(sorted(ranks, reverse=True))
    rank_acc_count = defaultdict(int)
    for rank in ranks:
        rank_acc_count[rank] = 0
    attempt_count = 0
    with open(attempt_file, "r") as ifile:
        csv_reader = csv.DictReader(ifile)
        for row in csv_reader:
            attempt_count += 1
            row: dict[str, str]
            try:
                actual_rank = int(row["rank"])
            except (ValueError, KeyError) as e:
                if isinstance(e, ValueError):
                    actual_rank = None
                else:
                    raise KeyError(f"{e} not present in {csv_reader.fieldnames}")
            if actual_rank is not None:
                for rank in ranks:
                    if actual_rank <= rank:
                        rank_acc_count[rank] += 1
                    else:
                        break
    return AccuracyResult(rank_acc_count, attempt_count)


def create_accuracy_table(attempts_file: Path) -> Table:
    result = measure_accuracy(attempts_file)
    return result.to_table()


def collect_checkpoint_files(checkpoints_dir: Path, filename_pattern: Optional[str] = None) -> Iterator[Path]:
    root_entries = list(Path(checkpoints_dir).iterdir())
    def _is_checkpoint_file(p: Path) -> bool:
        return p.name.endswith(".pt") and (not filename_pattern or fnmatch.fnmatch(p.name, filename_pattern))
    root_checkpoint_files = [f for f in root_entries if _is_checkpoint_file(f)]
    yield from iter(root_checkpoint_files)
    checkpoint_file_dirs = [d for d in root_entries if d.is_dir()]
    for checkpoint_dir in checkpoint_file_dirs:
        checkpoint_files = [f for f in checkpoint_dir.iterdir() if _is_checkpoint_file(f)]
        if checkpoint_files:
            yield sorted(checkpoint_files, reverse=True)[0]  # assumes latest checkpoint is best


DEFAULT_COLUMNS = (
    "dataset_name",
    "src_tok_emb.embedding.weight",
    "tgt_tok_emb.embedding.weight",
    "emb_size",
    "dim_feedforward",
    "lr",
    "transformer_dropout_rate",
    "tgt_pos_enc_disabled",
)


class EvalInfo(NamedTuple):

    eval_files: Sequence[Path]
    node_dirs: Sequence[Path]

    @staticmethod
    def headers() -> list[str]:
        return ["evfs", "ndds"]

    def latest_eval_filename(self) -> Optional[str]:
        if self.eval_files:
            return sorted(self.eval_files)[-1].name

    def data_dict(self) -> dict[str, Any]:
        return {k:v for k, v in zip(self.headers(), self.cells())}

    def cells(self) -> list[Any]:
        return [len(self.eval_files), len(self.node_dirs)]

    @staticmethod
    def find(checkpoint_file: Path) -> 'EvalInfo':
        evals_dir = checkpoint_file.parent / "evaluations"
        if not evals_dir.exists():
            return EvalInfo([], [])
        entries = list(evals_dir.iterdir())
        eval_files = list(filter(lambda p: p.is_file() and p.name.endswith(".csv"), entries))
        node_dirs = list(filter(lambda p: p.is_dir() and p.name.endswith("-nodes"), entries))
        return EvalInfo(eval_files, node_dirs)


def to_loss_dict(epoch_results: list[EpochResult]) -> dict[str, list[float]]:
    return {
        "train": [r.train_loss for r in epoch_results],
        "valid": [r.valid_loss for r in epoch_results],
    }


class Exported(NamedTuple):

    checkpoint_file: Path
    checkpoint_index: int
    min_valid_loss: float
    min_valid_loss_epoch: int
    final_valid_loss: float

    @staticmethod
    def summarize(exporteds: Sequence['Exported'], export_dir: Path, checkpoints_dir: Optional[Path] = None):
        table_rows = []
        headers = ["index", "checkpoint file", "min vl", "min vl epoch", "final vl"]
        valid_losses = [exported.min_valid_loss for exported in exporteds]
        min_min_valid_loss_idx = np.argmin(valid_losses)
        for exported_idx, exported in enumerate(exporteds):
            pathname = exported.checkpoint_file
            if checkpoints_dir is not None:
                pathname = pathname.relative_to(checkpoints_dir)
            pathname = str(pathname)
            if exported_idx == min_min_valid_loss_idx:
                pathname = "*" + pathname
            table_rows.append([exported.checkpoint_index, pathname, exported.min_valid_loss, exported.min_valid_loss_epoch, exported.final_valid_loss])
        table = Table(table_rows, headers)
        out_filename_stem = f"summary-{dlfp.common.timestamp()}"
        for tablefmt in ["csv", "simple_grid"]:
            suffix = ".csv" if tablefmt == "csv" else ".txt"
            out_pathname = export_dir / f"{out_filename_stem}{suffix}"
            table.write_file(out_pathname, fmt=tablefmt)


def export(checkpoint_file: Path, index: int, restored: Restored, eval_info: Optional[EvalInfo], export_dir: Path) -> Exported:
    def _prefix(filename: str, subindex: int) -> Path:
        return export_dir / f"{index}-{subindex}-{checkpoint_file.parent.name}-{filename}"
    info = {
        "loss": to_loss_dict(restored.epoch_results),
        "extra": restored.extra or {},
    }
    valid_losses = [r.valid_loss for r in restored.epoch_results]
    try:
        min_valid_loss_epoch = np.argmin(valid_losses)
        min_valid_loss = valid_losses[min_valid_loss_epoch]
        final_valid_loss = valid_losses[-1]
    except (ValueError, IndexError):
        min_valid_loss = float("inf")
        final_valid_loss = float("inf")
        min_valid_loss_epoch = -1
    with dlfp.common.open_write(_prefix("info.json", 0)) as ofile:
        json.dump(info, ofile, indent=2)
    if eval_info is not None:
        for e_index, eval_file in enumerate(eval_info.eval_files):
            dst_file = _prefix(f"attempts-{eval_file.name}", e_index + 1)
            shutil.copyfile(eval_file, dst_file)
            args_file = eval_file.parent / f"{eval_file.name}.args.txt"
            if args_file.is_file():
                shutil.copyfile(args_file, dst_file.parent / f"{dst_file.name}.args.txt")
    return Exported(checkpoint_file, index, min_valid_loss, min_valid_loss_epoch, final_valid_loss)


def create_params_table(checkpoints_dir: Path,
                        *,
                        filename_pattern: Optional[str] = None,
                        dataset: Optional[str] = None,
                        find_eval_info: bool = False,
                        export_dir: Optional[Path] = None,
                        columns: Sequence[str] = None) -> Table:
    columns = columns or DEFAULT_COLUMNS
    if find_eval_info:
        columns = list(columns) + list(EvalInfo.headers())
    table_rows = []
    exporteds = []
    for checkpoint_index, checkpoint_file in enumerate(sorted(collect_checkpoint_files(checkpoints_dir, filename_pattern=filename_pattern))):
        rel_file = checkpoint_file.relative_to(checkpoints_dir).as_posix()
        try:
            eval_info = None
            restored = Restored.from_file(checkpoint_file, device="cpu")
            metadata = (restored.extra or {}).get("metadata", {})
            dataset_name = metadata.get("dataset_name", None)
            if dataset_name and dataset and (dataset != dataset_name):
                continue
            ok, train_hp, model_hp = dlfp.models.get_hyperparameters(restored)
            if not ok:
                _log.warning("model/train hyperparameters not found in %s", checkpoint_file.as_posix())
                continue
            param_sizes = {k:v[0] for k, v in restored.model_param_shapes().items()}
            train_hp: TrainHyperparametry
            model_hp: ModelHyperparametry
            merged = {}
            merged.update(param_sizes)
            merged.update(metadata)
            merged.update(train_hp._asdict())
            merged.update(model_hp._asdict())
            if find_eval_info:
                eval_info = EvalInfo.find(checkpoint_file)
                merged.update(eval_info.data_dict())
            table_rows.append([rel_file] + [merged.get(k, None) for k in columns])
            if export_dir:
                exporteds.append(export(checkpoint_file, checkpoint_index, restored, eval_info, export_dir))
        except Exception as e:
            _log.warning(f"failed to process {rel_file} due to {type(e).__name__}: {e}")
    all_columns = ["file"] + list(columns)
    all_columns = [HYPERPARAMETER_ABBREVIATIONS.get(c, c) for c in all_columns]
    if export_dir and exporteds:
        Exported.summarize(exporteds, export_dir, checkpoints_dir)
    return Table(table_rows, headers=all_columns)


def create_loss_table(checkpoint_file: Path) -> Table:
    rows = []
    checkpoint = Restored.from_file(checkpoint_file, device='cpu')
    for epoch_result in checkpoint.epoch_results:
        rows.append(epoch_result.to_row())
    return Table(rows, headers=EpochResult.headers())


class Evaluation(NamedTuple):

    split: str
    max_ranks: tuple[int, ...]
    accuracy_result: AccuracyResult
    probnorm: Optional[str] = None

    @staticmethod
    def from_attempts_file(attempts_file: Path) -> 'Evaluation':
        args_file = Path(str(attempts_file) + ".args.txt")
        accuracy_result = measure_accuracy(attempts_file)
        if not args_file.is_file():
            raise ValueError(f"no args file for {attempts_file.name}")
        eval_config_dict = json.loads(args_file.read_text().split("===")[-1])
        eval_config = EvalConfig.from_args([f"{k}={v}" for k, v in eval_config_dict.items() if not v is None])
        node_strat = eval_config.parse_node_strategy()
        max_ranks = node_strat.get('max_ranks', None)
        if max_ranks is None:
            raise ValueError(f"no max ranks identified in args file for {attempts_file}")
        probnorm = node_strat.get('probnorm', None)
        return Evaluation(
            split=eval_config.split or _extract_dataset(attempts_file.name),
            max_ranks=tuple(max_ranks),
            accuracy_result=accuracy_result,
            probnorm=probnorm,
        )

    def suggest_title(self) -> str:
        title = "-".join(map(str, self.max_ranks))
        if self.probnorm:
            title = f"{title}_{self.probnorm}"
        return title

    @staticmethod
    def to_table(evaluations: Iterable['Evaluation']) -> Table:
        splits = set(evaluation.split for evaluation in evaluations)
        if len(splits) != 1:
            raise ValueError(f"evaluations performed on {len(splits)} splits: {splits}")
        rows = []
        ranks_superset = set()
        for evaluation in evaluations:
            ranks_superset.update(evaluation.accuracy_result.rank_acc_count.keys())
        ranks = sorted(ranks_superset)
        headers = ["params/ranks"] + ranks
        for evaluation in evaluations:
            row = [evaluation.suggest_title()]
            for rank in ranks:
                acc = evaluation.accuracy_result.rank_acc_count.get(rank, 0) / evaluation.accuracy_result.attempt_count
                row.append(acc)
            rows.append(row)
        return Table(rows, headers)


class Collated(NamedTuple):

    dataset: str
    model_hp: ModelHyperparametry
    train_hp: TrainHyperparametry
    train_loss: list[float]
    valid_loss: list[float]
    evaluations: list[Evaluation]

    @staticmethod
    def from_info_file(info_file: Path) -> 'Collated':
        timestamp = _extract_timestamp(info_file.name)
        attempts_files = map(Path, glob.glob(os.path.join(info_file.parent, f"*-{timestamp}-attempts-*.csv")))
        evaluations = []
        for attempts_file in attempts_files:
            try:
                evaluation = Evaluation.from_attempts_file(attempts_file)
                evaluations.append(evaluation)
            except Exception as e:
                if isinstance(e, ValueError) and str(e).startswith("no args file"):
                    pass
                else:
                    _log.warning(f"failed to read from {attempts_file} due to {type(e)}: {e}")
        info = json.loads(info_file.read_text())
        train_config = info['extra']['train_config']
        train_hp = TrainHyperparametry(**(train_config['train_hp']))
        model_hp = ModelHyperparametry(**(train_config['model_hp']))
        loss = info['loss']
        train_loss, valid_loss = loss['train'], loss['valid']
        metadata = info['extra']['metadata']
        dataset = metadata.get('dataset_name', '<unidentified>')
        return Collated(
            dataset,
            model_hp,
            train_hp,
            train_loss,
            valid_loss,
            evaluations,
        )

    def nondefault_hyperparameters(self) -> dict[str, Any]:
        d = {}
        d.update(dlfp.common.namedtuple_diff(ModelHyperparametry(), self.model_hp))
        d.update(dlfp.common.namedtuple_diff(TrainHyperparametry(), self.train_hp))
        return d

    def suggest_stem(self) -> str:
        nondefaults = self.nondefault_hyperparameters()
        def _abbreviate(key: str) -> str:
            abbr = HYPERPARAMETER_ABBREVIATIONS.get(key, key)
            return abbr.replace("_", "")
        if nondefaults:
            hp_infix = "_".join(f"{_abbreviate(k)}{v}" for k, v in nondefaults.items())
        else:
            hp_infix = "default"
        return f"{self.dataset}-{hp_infix}"

    def write(self, output_dir: Path, index: int):
        stem = self.suggest_stem()
        output_dir.mkdir(parents=True, exist_ok=True)
        losses_filename = f"{index}-{stem}-losses.json"
        (output_dir / losses_filename).write_text(json.dumps({'train': self.train_loss, 'valid': self.valid_loss}, indent=2))
        acc_filename = f"{index}-{stem}-accuracy.csv"
        if self.evaluations:
            eval_table = Evaluation.to_table(self.evaluations)
            eval_table.write_file(output_dir / acc_filename, fmt="csv")

    def epoch_count(self) -> int:
        return len(self.train_loss)


def _extract_timestamp(info_filename: str) -> str:
    m = re.fullmatch(r'^(\d+-\d+-)?(\d{8}-\d{4,6})-info\.json$', info_filename)
    if m is None:
        raise ValueError(f"unexpected filename convention: {repr(info_filename)}")
    return m.group(2)


def _extract_dataset(attempts_filename: str) -> str:
    m = re.fullmatch(r'^.*-epoch\d+_(\w+)_\d{8}-\d{4,6}\.csv$', attempts_filename)
    if m is None:
        raise ValueError(f"unexpected filename convention: {repr(attempts_filename)}")
    return m.group(1)


class UsageException(Exception):
    pass


def collate_all(export_dir: Path, collated_dir: Optional[Path]) -> Table:
    if not export_dir:
        raise UsageException("--export argument required in collate mode")
    collated_dir = collated_dir or (export_dir / "collated")
    info_files = sorted(map(Path, glob.glob(os.path.join(export_dir, "*-info.json"))))
    if not info_files:
        raise UsageException(f"no info files in {export_dir.absolute()}")
    collateds = [Collated.from_info_file(info_file) for info_file in info_files]
    by_stem = defaultdict(list)
    headers = ["idx", "identity", "epochs", "evals"]
    rows = []
    for index, (collated, info_file) in enumerate(zip(collateds, info_files)):
        by_stem[collated.suggest_stem()].append(info_file)
        collated.write(collated_dir, index)
        rows.append([index, collated.suggest_stem(), collated.epoch_count(), len(collated.evaluations)])
    for stem, info_files in by_stem.items():
        if len(info_files) > 1:
            _log.warning(f"multiple files identified by {repr(stem)}: {[f.name for f in info_files]}")
    return Table(rows, headers)


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", metavar="PATH", type=Path, help="file or directory pathname, depending on mode")
    mode_choices = ("checkpoint", "accuracy", "params", "collate")
    parser.add_argument("-m", "--mode", metavar="MODE", choices=mode_choices, required=True, help=f"one of {mode_choices}")
    format_choices = ("csv", "json", "table")
    parser.add_argument("-t", "--format", metavar="FMT", choices=format_choices, help=f"one of {format_choices}")
    parser.add_argument("-o", "--output", metavar="PATH", type=Path)
    parser.add_argument("--eval", action='store_true', help="in params mode, find eval info")
    parser.add_argument("--export", metavar="DIR", type=Path, help="in params mode, export data to DIR; in collate mode, read from DIR")
    parser.add_argument("-p", "--pattern", help="in params mode, filter checkpoint files with filename pattern")
    parser.add_argument("-d", "--dataset", metavar="NAME", help="in params mode, filter checkpoint files by dataset")
    args = parser.parse_args()
    logging.basicConfig(level="INFO")
    output_format = {
        "table": "simple_grid",
    }.get(args.format, args.format)
    try:
        if args.mode == "accuracy":
            table = create_accuracy_table(args.file)
        elif args.mode == "checkpoint":
            table = create_loss_table(args.file)
        elif args.mode == "params":
            checkpoints_dir = args.file or (dlfp.common.get_repo_root() / "checkpoints")
            table = create_params_table(checkpoints_dir, dataset=args.dataset, filename_pattern=args.pattern, find_eval_info=args.eval, export_dir=args.export)
        elif args.mode == "collate":
            table = collate_all(args.export, args.output)
        else:
            raise NotImplementedError(f"bug: unhandled mode {repr(args.mode)}")
        table.write_file(args.output, fmt=output_format)
    except UsageException as e:
        _log.error("usage: %s", e)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
