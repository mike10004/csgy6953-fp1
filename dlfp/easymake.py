#!/usr/bin/env python3
import re
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Collection
from typing import NamedTuple
from typing import Optional

from tqdm import tqdm

import dlfp.datasets
from dlfp.datasets import DatasetResolver
from dlfp.datasets import Tokenized
from dlfp.utils import PhrasePairDataset


class PredicateSet:

    def __init__(self,
                 max_tokens: Optional[int] = None,
                 prohibited_substrings: Collection[str] = None,
                 require_regex_match: Optional[str] = None):
        self.max_tokens = max_tokens
        self.prohibited_substrings = frozenset(prohibited_substrings or ())
        self.require_regex_match = require_regex_match

    def required_length(self, tokenized: Tokenized) -> bool:
        return self.max_tokens is None or (len(tokenized.tokens) <= self.max_tokens)

    def regex_match(self, tokenized: Tokenized) -> bool:
        if not self.require_regex_match:
            return True
        return re.fullmatch(self.require_regex_match, tokenized.phrase) is not None

    def no_prohibited_substrings(self, tokenized: Tokenized) -> bool:
        for substring in self.prohibited_substrings:
            if substring in tokenized.phrase:
                return False
        return True

    def all(self, tokenized: Tokenized) -> bool:
        predicates = [
            self.required_length,
            self.no_prohibited_substrings,
            self.regex_match,
        ]
        return all(map(lambda predicate: predicate(tokenized), predicates))


class Result(NamedTuple):

    superset: int
    subset: int
    files: tuple[Path, Path]

    def removed(self) -> int:
        return self.superset - self.subset


def clean(dataset: PhrasePairDataset, source_predicates: PredicateSet, target_predicates: PredicateSet, output_prefix: str) -> Result:
    source_file = Path(output_prefix + "source")
    target_file = Path(output_prefix + "target")
    src_language, tgt_language = dlfp.datasets.get_languages(dataset)
    for file in [source_file, target_file]:
        file.parent.mkdir(parents=True, exist_ok=True)
    superset_count, subset_count = 0, 0
    with open(source_file, "w") as sfile:
        with open(target_file, "w") as tfile:
            for source, target in tqdm(dataset, total=len(dataset), file=sys.stdout):
                superset_count += 1
                source_tokenized = Tokenized(source, src_language.tokenizer(source))
                if not source_predicates.all(source_tokenized):
                    continue
                target_tokenized = Tokenized(target, tgt_language.tokenizer(target))
                if not target_predicates.all(target_tokenized):
                    continue
                subset_count += 1
                print(source, file=sfile)
                print(target, file=tfile)
    return Result(superset_count, subset_count, (source_file, target_file))


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument("--split", action='append')
    parser.add_argument("-o", "--output", type=Path, metavar="DIR")
    args = parser.parse_args()
    resolver = DatasetResolver()
    source_predicates = PredicateSet(prohibited_substrings={"-Across", "-Down", "-across", "-down"})
    target_predicates = PredicateSet(max_tokens=4, require_regex_match=r'^[a-z ]+$')
    dataset_name = "easymark"
    output_dir = args.output or (resolver.data_root / "datasets" / dataset_name)
    for split in (args.split or ["train", "valid", "test"]):
        dataset = resolver.benchmark(split)
        output_prefix = str(output_dir / f"{split}.")
        result = clean(dataset, source_predicates, target_predicates, output_prefix)
        print(f"{split:5}: {result.superset:6d} -> {result.subset:6d}; {[f.name for f in result.files]} written in {output_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
