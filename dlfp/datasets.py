#!/usr/bin/env python3

import logging
from pathlib import Path

from dlfp.utils import PhrasePairDataset
from dlfp.utils import get_repo_root
from dlfp.utils import Split

_log = logging.getLogger(__name__)

def _readlines(pathname: Path) -> list[str]:
    with open(pathname, "r") as ifile:
        return ifile.read().splitlines()


class DatasetResolver:

    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.encoding = 'utf-8'

    @staticmethod
    def default() -> 'DatasetResolver':
        data_root = get_repo_root() / "data" / "datasets"
        return DatasetResolver(data_root)

    def benchmark(self, split: Split) -> PhrasePairDataset:
        stem = {
            "valid": "val"
        }.get(split, split)
        dataset_dir = self.data_root / "benchmark"
        source_filename, target_filename = f"{stem}.source", f"{stem}.target"
        source_lines = (dataset_dir / source_filename).read_text(self.encoding).splitlines()
        target_lines = (dataset_dir / target_filename).read_text(self.encoding).splitlines()
        if len(source_lines) != len(target_lines):
            _log.warning(f"source/target length mismatch: {len(source_lines)} != {len(target_lines)}")
        phrase_pairs = list(zip(source_lines, target_lines))
        return PhrasePairDataset("benchmark", phrase_pairs, language_pair=("clue", "answer"))
