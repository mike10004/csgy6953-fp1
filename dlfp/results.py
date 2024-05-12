#!/usr/bin/env python3
import fnmatch
import sys
import csv
import json
import shutil
import logging
from collections import defaultdict
from pathlib import Path
from argparse import ArgumentParser
from typing import Collection
from typing import Any
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

_log = logging.getLogger(__name__)
DEFAULT_RANKS = (1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)


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
            except ValueError:
                actual_rank = None
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

    @staticmethod
    def summarize(exporteds: Sequence['Exported'], export_dir: Path, checkpoints_dir: Optional[Path] = None):
        table_rows = []
        headers = ["index", "checkpoint file", "min valid loss", "min valid loss epoch"]
        valid_losses = [exported.min_valid_loss for exported in exporteds]
        min_min_valid_loss_idx = np.argmin(valid_losses)
        for exported_idx, exported in enumerate(exporteds):
            pathname = exported.checkpoint_file
            if checkpoints_dir is not None:
                pathname = pathname.relative_to(checkpoints_dir)
            pathname = str(pathname)
            if exported_idx == min_min_valid_loss_idx:
                pathname = "*" + pathname
            table_rows.append([exported.checkpoint_index, pathname, exported.min_valid_loss, exported.min_valid_loss_epoch])
        table = Table(table_rows, headers)
        out_filename_stem = f"summary-{dlfp.common.timestamp()}"
        for tablefmt in ["csv", "grid"]:
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
    except (ValueError, IndexError):
        min_valid_loss = float("inf")
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
    return Exported(checkpoint_file, index, min_valid_loss, min_valid_loss_epoch)


def create_params_table(checkpoints_dir: Path,
                        *,
                        filename_pattern: Optional[str] = None,
                        find_eval_info: bool = False,
                        export_dir: Optional[Path] = None,
                        columns: Sequence[str] = None) -> Table:
    columns = columns or DEFAULT_COLUMNS
    if find_eval_info:
        columns = list(columns) + list(EvalInfo.headers())
    table_rows = []
    short_names = {
        "transformer_dropout_rate": "tdr",
        "input_dropout_rate": "idr",
        "dataset_name": "dataset",
        "src_tok_emb.embedding.weight": "s_emb_sz",
        "tgt_tok_emb.embedding.weight": "t_emb_sz",
        "dim_feedforward": "ffnd",
    }
    exporteds = []
    for checkpoint_index, checkpoint_file in enumerate(sorted(collect_checkpoint_files(checkpoints_dir, filename_pattern=filename_pattern))):
        rel_file = checkpoint_file.relative_to(checkpoints_dir).as_posix()
        try:
            eval_info = None
            restored = Restored.from_file(checkpoint_file, device="cpu")
            metadata = (restored.extra or {}).get("metadata", {})
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
            _log.warning(f"failed to process {rel_file} due to {type(e)}: {e}")
    all_columns = ["file"] + list(columns)
    all_columns = [short_names.get(c, c) for c in all_columns]
    if export_dir and exporteds:
        Exported.summarize(exporteds, export_dir, checkpoints_dir)
    return Table(table_rows, headers=all_columns)


def create_loss_table(checkpoint_file: Path) -> Table:
    rows = []
    checkpoint = Restored.from_file(checkpoint_file, device='cpu')
    for epoch_result in checkpoint.epoch_results:
        rows.append(epoch_result.to_row())
    return Table(rows, headers=EpochResult.headers())


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", metavar="PATH", type=Path, help="file or directory pathname, depending on mode")
    mode_choices = ("checkpoint", "accuracy", "params")
    parser.add_argument("-m", "--mode", metavar="MODE", choices=mode_choices, required=True, help=f"one of {mode_choices}")
    format_choices = ("csv", "json", "table")
    parser.add_argument("-t", "--format", metavar="FMT", choices=format_choices, help=f"one of {format_choices}")
    parser.add_argument("-o", "--output", metavar="FILE", type=Path)
    parser.add_argument("--eval", action='store_true', help="in params mode, find eval info")
    parser.add_argument("--export", metavar="DIR", type=Path, help="export data to DIR")
    parser.add_argument("-p", "--pattern", help="in params mode, filter checkpoint files with filename pattern")
    args = parser.parse_args()
    logging.basicConfig(level="INFO")
    output_format = {
        "table": "simple_grid",
    }.get(args.format, args.format)
    if args.mode == "accuracy":
        table = create_accuracy_table(args.file)
    elif args.mode == "checkpoint":
        table = create_loss_table(args.file)
    elif args.mode == "params":
        checkpoints_dir = args.file or (dlfp.common.get_repo_root() / "checkpoints")
        table = create_params_table(checkpoints_dir, filename_pattern=args.pattern, find_eval_info=args.eval, export_dir=args.export)
    else:
        raise NotImplementedError(f"bug: unhandled mode {repr(args.mode)}")
    table.write_file(args.output, fmt=output_format)
    return 0


if __name__ == '__main__':
    sys.exit(main())
