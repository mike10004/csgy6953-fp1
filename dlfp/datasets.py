#!/usr/bin/env python3

import sys
import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterator
from typing import NamedTuple
from typing import Sequence

import tabulate

from dlfp.utils import Language
from dlfp.utils import LanguageCache
from dlfp.utils import PhrasePairDataset
from dlfp.utils import get_repo_root
from dlfp.utils import Split

_log = logging.getLogger(__name__)

def _readlines(pathname: Path) -> list[str]:
    with open(pathname, "r") as ifile:
        return ifile.read().splitlines()


class DatasetResolver:

    def __init__(self, data_root: Path = None):
        self.data_root = data_root or (get_repo_root() / "data" / "datasets")
        self.encoding = 'utf-8'

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
        def clean_target(dirty: str) -> str:
            if dirty.startswith('"""'):
                return dirty.strip('"')
            return dirty
        phrase_pairs = list(zip(source_lines, map(clean_target, target_lines)))
        return PhrasePairDataset("benchmark", phrase_pairs, language_pair=("clue", "answer"))


def get_languages(dataset: PhrasePairDataset) -> tuple[Language, Language]:
    cache = LanguageCache()
    if dataset.name == "benchmark":
        source = cache.get(dataset, "clue", "spacy", "en_core_web_sm")
        target = cache.get(dataset, "answer", "spacy", "en_core_web_sm")
        return source, target
    if dataset.name == "multi30k_de_en":
        source = cache.get(dataset, "de", "spacy", "de_core_news_sm")
        target = cache.get(dataset, "en", "spacy", "en_core_web_sm")
        return source, target
    else:
        raise NotImplementedError(f"unrecognized dataset: {repr(dataset.name)}")

class Summary(NamedTuple):

    name: str
    size: int
    src_vocab_size: int
    src_max_tokens: int
    tgt_vocab_size: int
    tgt_max_tokens: int

    @staticmethod
    def interrogate(dataset: PhrasePairDataset, src_language: Language, tgt_language: Language) -> 'Summary':
        src_token_histo = _token_count_histo(dataset.source_phrases(), src_language)
        tgt_token_histo = _token_count_histo(dataset.target_phrases(), tgt_language)
        return Summary(
            name=dataset.name,
            size=len(dataset),
            src_vocab_size=len(src_language.vocab),
            src_max_tokens=max(src_token_histo.keys()),
            tgt_vocab_size=len(tgt_language.vocab),
            tgt_max_tokens=max(tgt_token_histo.keys()),
        )

def _token_count_histo(phrases: Iterator[str], lang: Language) -> dict[int, int]:
    counts = defaultdict(int)
    for phrase in phrases:
        tokens = lang.tokenizer(phrase)
        counts[len(tokens)] += 1
    return dict(counts)


class TokenAnalysis(NamedTuple):

    top_k: tuple[tuple[str, Sequence[str]], ...]
    token_count_histo: dict[int, int]

    @staticmethod
    def interrogate(phrases: Iterator[str], k: int, language: Language, min_tokens: int = 0) -> 'TokenAnalysis':
        phrases_with_token_counts: list[tuple[str, Sequence[str]]] = []
        histo = defaultdict(int)
        for phrase in phrases:
            tokens = language.tokenizer(phrase)
            if len(tokens) < min_tokens:
                continue
            histo[len(tokens)] += 1
            phrases_with_token_counts.append((phrase, tokens))
        phrases_with_token_counts.sort(key=lambda x: len(x[1]), reverse=True)
        return TokenAnalysis(
            top_k=tuple(phrases_with_token_counts[:k]),
            token_count_histo=histo,
        )


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("-d", "--dataset", metavar="NAME", action='append', required=True)
    parser.add_argument("-s", "--split", default="train", metavar="SPLIT", choices=('train', 'valid', 'test'), help="train, valid, or test")
    mode_choices = ("summary", "tokens")
    parser.add_argument("-m", "--mode", choices=mode_choices, default="summary", metavar="MODE", help=f"one of {mode_choices}")
    half_choices = ("source", "target")
    parser.add_argument("--half", metavar="HALF", choices=half_choices, default="target", help=f"one of {half_choices}")
    parser.add_argument("-k", type=int, default=10, help="tokens mode: top K")
    args = parser.parse_args()
    data_root = args.data_root
    resolver = DatasetResolver(data_root)
    def get_dataset(dataset_name_) -> PhrasePairDataset:
        fn = getattr(resolver, dataset_name_)
        return fn(split=args.split)

    if args.mode == "summary":
        summaries = []
        for dataset_name in args.dataset:
            dataset = get_dataset(dataset_name)
            src_language, tgt_language = get_languages(dataset)
            summary = Summary.interrogate(dataset, src_language, tgt_language)
            summaries.append(summary)
        headers = Summary._fields
        table = [[getattr(d, k) for k in headers] for d in summaries]
        print(tabulate.tabulate(table, headers=headers))
    elif args.mode == "tokens":
        headers = ["n", "phrase"]
        table = []
        histos = []
        for dataset_name in args.dataset:
            dataset = get_dataset(dataset_name)
            languages = get_languages(dataset)
            index = {
                "source": 0,
                "target": 1,
            }[args.half]
            token_analysis = TokenAnalysis.interrogate(dataset.phrases(index), k=args.k, language=languages[index])
            for phrase, tokens in token_analysis.top_k:
                table.append((len(tokens), phrase, str(tokens)))
            histos.append(token_analysis.token_count_histo)
        print(tabulate.tabulate(table, headers=headers))
        print()
        for histo in histos:
            total = sum(histo.values())
            for token_count, frequency in histo.items():
                print(f"{token_count:2d}: {frequency:6d} ({100.0 * frequency / total:.1f}%)")
    else:
        raise NotImplementedError("unsupported mode")
    return 0


if __name__ == "__main__":
    sys.exit(main())
