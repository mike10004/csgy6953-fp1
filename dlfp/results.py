#!/usr/bin/env python3

import sys
import csv
from collections import defaultdict
from pathlib import Path
from argparse import ArgumentParser
from typing import Collection
from typing import Any
from typing import NamedTuple
from dlfp.common import Table

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
        return Table(table, *self.table_headers())


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


def create_accuracy_table(attempts_file: Path) -> Table:
    result = measure_accuracy(attempts_file)
    return result.to_table()


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("file", type=Path)
    mode_choices = ("checkpoint", "accuracy")
    parser.add_argument("-m", "--mode", metavar="MODE", choices=mode_choices, required=True)
    format_choices = ("csv", "table", "json")
    parser.add_argument("-t", "--format", metavar="FMT", choices=format_choices)
    parser.add_argument("-o", "--output", metavar="FILE", type=Path)
    args = parser.parse_args()
    if args.mode == "accuracy":
        table = create_accuracy_table(args.file)
        table.write_file(args.output, )

    return 0


if __name__ == '__main__':
    sys.exit(main())
