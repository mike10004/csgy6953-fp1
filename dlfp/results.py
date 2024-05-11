#!/usr/bin/env python3

import os
import sys
import csv
import logging
from collections import defaultdict
from pathlib import Path
from argparse import ArgumentParser
from typing import Collection
from typing import Any
from typing import Iterator
from typing import NamedTuple
from typing import Sequence

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


def collect_checkpoint_files(checkpoints_dir: Path) -> Iterator[Path]:
    root_entries = list(Path(checkpoints_dir).iterdir())
    def _is_checkpoint_file(p: Path) -> bool:
        return p.name.endswith(".pt")
    root_checkpoint_files = [f for f in root_entries if _is_checkpoint_file(f)]
    yield from iter(root_checkpoint_files)
    checkpoint_file_dirs = [d for d in root_entries if d.is_dir()]
    for checkpoint_dir in checkpoint_file_dirs:
        checkpoint_files = [f for f in checkpoint_dir.iterdir() if _is_checkpoint_file(f)]
        if checkpoint_files:
            yield sorted(checkpoint_files, reverse=True)[0]  # assumes latest checkpoint is best


DEFAULT_COLUMNS = ("dataset_name", "src_tok_emb.embedding.weight", "tgt_tok_emb.embedding.weight", "emb_size", "lr", "transformer_dropout_rate", "input_dropout_rate")

def create_params_table(checkpoints_dir: Path, columns: Sequence[str] = None) -> Table:
    columns = columns or DEFAULT_COLUMNS
    table_rows = []
    short_names = {
        "transformer_dropout_rate": "tdr",
        "input_dropout_rate": "idr",
        "dataset_name": "dataset",
        "src_tok_emb.embedding.weight": "s_emb_sz",
        "tgt_tok_emb.embedding.weight": "t_emb_sz",
    }
    for checkpoint_file in sorted(collect_checkpoint_files(checkpoints_dir)):
        rel_file = checkpoint_file.relative_to(checkpoints_dir).as_posix()
        try:
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
            table_rows.append([rel_file] + [merged.get(k, None) for k in columns])
        except Exception as e:
            _log.warning(f"failed to extract hyperparametry from {rel_file} due to {type(e)}: {e}")
    all_columns = ["file"] + list(columns)
    all_columns = [short_names.get(c, c) for c in all_columns]
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
        table = create_params_table(checkpoints_dir)
    else:
        raise NotImplementedError(f"bug: unhandled mode {repr(args.mode)}")
    table.write_file(args.output, fmt=output_format)
    return 0


if __name__ == '__main__':
    sys.exit(main())
