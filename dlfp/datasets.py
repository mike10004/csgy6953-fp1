#!/usr/bin/env python3

import re
import sys
import argparse
import logging
from collections import defaultdict
from pathlib import Path
from random import Random
from typing import Iterator
from typing import NamedTuple
from typing import Sequence
from typing import Optional
from typing import Collection

import tabulate
from tqdm import tqdm

from dlfp.common import Table
from dlfp.utils import Language
from dlfp.utils import LanguageCache
from dlfp.utils import PhrasePairDataset
from dlfp.common import get_repo_root
from dlfp.utils import Split

_log = logging.getLogger(__name__)

def _readlines(pathname: Path) -> list[str]:
    with open(pathname, "r") as ifile:
        return ifile.read().splitlines()


class DatasetResolver:

    def __init__(self, data_root: Path = None):
        self.data_root = data_root or (get_repo_root() / "data")
        self.encoding = 'utf-8'

    def by_name(self, dataset_name: str, split: Split) -> PhrasePairDataset:
        fn = getattr(self, dataset_name)
        return fn(split)

    def multi30k_de_en(self, split: Split) -> PhrasePairDataset:
        from torchtext.datasets import Multi30k
        language_pair = ('de', 'en')
        # noinspection PyTypeChecker
        items: list[tuple[str, str]] = list(Multi30k(root=str(self.data_root), split=split, language_pair=language_pair))
        return PhrasePairDataset("multi30k_de_en", items, language_pair)

    def benchmark(self, split: Split) -> PhrasePairDataset:
        stem = {
            "valid": "val"
        }.get(split, split)
        source_filename, target_filename = f"{stem}.source", f"{stem}.target"
        return self._load_marklike("benchmark", source_filename, target_filename)

    def easymark(self, split: Split) -> PhrasePairDataset:
        source_filename, target_filename = f"{split}.source", f"{split}.target"
        return self._load_marklike("easymark", source_filename, target_filename)

    def _load_marklike(self, dataset_name: str, source_filename: str, target_filename: str):
        dataset_dir = self.data_root / "datasets" / dataset_name
        source_lines = (dataset_dir / source_filename).read_text(self.encoding).splitlines()
        target_lines = (dataset_dir / target_filename).read_text(self.encoding).splitlines()
        if len(source_lines) != len(target_lines):
            _log.warning(f"source/target length mismatch: {len(source_lines)} != {len(target_lines)}")
        phrase_pairs = list(zip(source_lines, target_lines))
        return PhrasePairDataset(dataset_name, phrase_pairs, language_pair=("clue", "answer"))


def get_languages(dataset: PhrasePairDataset) -> tuple[Language, Language]:
    cache = LanguageCache()
    if dataset.name in {"benchmark", "easymark"}:
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


class Tokenized(NamedTuple):

    phrase: str
    tokens: Sequence[str]


class TokenAnalysis(NamedTuple):

    top_k: tuple[Tokenized, ...]
    token_count_histo: dict[int, int]

    @staticmethod
    def interrogate(phrases: Iterator[str], k: int, sample: str, language: Language, min_tokens: int = 0) -> 'TokenAnalysis':
        if sample == "random":
            phrases = list(phrases)
            histo = _token_count_histo(iter(phrases), language)
            Random().shuffle(phrases)
            tokenizeds = []
            for phrase in phrases:
                if len(tokenizeds) >= k:
                    break
                tokens = language.tokenizer(phrase)
                tokenizeds.append(Tokenized(phrase, tokens))
        else:
            tokenizeds: list[Tokenized] = []
            histo = defaultdict(int)
            for phrase in phrases:
                tokens = language.tokenizer(phrase)
                if len(tokens) < min_tokens:
                    continue
                histo[len(tokens)] += 1
                tokenizeds.append(Tokenized(phrase, tokens))
            reverse = sample == "longest"
            tokenizeds.sort(key=lambda x: len(x.tokens), reverse=reverse)
        return TokenAnalysis(
            top_k=tuple(tokenizeds[:k]),
            token_count_histo=histo,
        )


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


def filter_dataset(dataset: PhrasePairDataset, source_predicates: PredicateSet, target_predicates: PredicateSet, output_prefix: str) -> Result:
    source_file = Path(output_prefix + "source")
    target_file = Path(output_prefix + "target")
    src_language, tgt_language = get_languages(dataset)
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
    result = Result(superset_count, subset_count, (source_file, target_file))
    return result

def create_subset(dataset_name: str, split: Split, output_dir: Optional[Path], shuffle_seed: int, size: int) -> Result:
    assert dataset_name is not None, "source dataset must be specified"
    assert split is not None, "source dataset split must be specified"
    assert shuffle_seed is not None, "must specify shuffle seed as --shuffle argument"
    assert size is not None, "must specify --size argument"
    resolver = DatasetResolver()
    output_dir = output_dir or (resolver.data_root / "datasets" / dataset_name)
    full_dataset = resolver.by_name(dataset_name, split)
    rng = Random(shuffle_seed)
    subset = full_dataset.shuffle(rng).slice(0, size)
    filename_stem = f"{split}_r{shuffle_seed}_s{size}"
    source_file = output_dir / f"{filename_stem}.source"
    target_file = output_dir / f"{filename_stem}.target"
    with open(source_file, "w") as sfile:
        with open(target_file, "w") as tfile:
            for source, target in tqdm(subset, total=len(subset), file=sys.stdout):
                print(source, file=sfile)
                print(target, file=tfile)
    result = Result(len(full_dataset), len(subset), (source_file, target_file))
    print(f"{dataset_name} {split}: {result.superset:6d} -> {result.subset:6d}; {[f.name for f in result.files]} written in {output_dir}")
    return result


def create_benchmark_variation(resolver: DatasetResolver, source_predicates: PredicateSet, target_predicates: PredicateSet, output_dir: Path):
    splits: list[Split] = ["train", "valid", "test"]
    for split in splits:
        dataset = resolver.benchmark(split)
        output_prefix = str(output_dir / f"{split}.")
        result = filter_dataset(dataset, source_predicates, target_predicates, output_prefix)
        print(f"{split:5}: {result.superset:6d} -> {result.subset:6d}; {[f.name for f in result.files]} written in {output_dir}")


def create_dataset(dataset_name: str, output_dir: Optional[Path] = None, overwrite: bool = False) -> int:
    resolver = DatasetResolver()
    output_dir = output_dir or (resolver.data_root / "datasets" / dataset_name)
    if not overwrite and output_dir.exists():
        _log.error("overwrite=False and output dir already exists: %s", output_dir)
        return 2
    if dataset_name == "easymark":
        source_predicates = PredicateSet(prohibited_substrings={"-Across", "-Down", "-across", "-down"})
        target_predicates = PredicateSet(max_tokens=4, require_regex_match=r'^[a-z ]+$')
        create_benchmark_variation(resolver, source_predicates, target_predicates, output_dir)
    if dataset_name == "onemark":
        source_predicates = PredicateSet(prohibited_substrings={"-Across", "-Down", "-across", "-down"})
        target_predicates = PredicateSet(max_tokens=1, require_regex_match=r'^[a-z ]+$')
        create_benchmark_variation(resolver, source_predicates, target_predicates, output_dir)
    else:
        _log.error("unsupported dataset: %s", repr(dataset_name))
        return 1
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("-d", "--dataset", metavar="NAME", action='append', required=True)
    parser.add_argument("-s", "--split", default="train", metavar="SPLIT", choices=('train', 'valid', 'test'), help="train, valid, or test")
    mode_choices = ("summary", "tokens", "juxtapose", "create", "subset")
    parser.add_argument("-m", "--mode", choices=mode_choices, default="summary", metavar="MODE", help=f"one of {mode_choices}")
    half_choices = ("source", "target")
    parser.add_argument("--half", metavar="HALF", choices=half_choices, default="target", help=f"one of {half_choices}")
    parser.add_argument("-k", type=int, default=10, help="sample size for 'tokens' and 'juxtapose' modes")
    tokens_choices = ("random", "longest", "shortest")
    parser.add_argument("--sample", choices=tokens_choices, metavar="SAMPLE", default="random", help=f"how to sample from dataset; one of {tokens_choices}")
    parser.add_argument("--min-tokens", type=int, default=0)
    parser.add_argument("-o", "--output", metavar="DIR", type=Path, help="output directory for create mode")
    parser.add_argument("--overwrite", action='store_true', help="overwrite existing dataset in create mode")
    parser.add_argument("--size", type=int, metavar="N", help="size of subset")
    parser.add_argument("--shuffle", type=int, metavar="N", help="random seed for subset shuffle")
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
        headers = ["n", "phrase", "tokens"]
        table = []
        histos = []
        for dataset_name in args.dataset:
            dataset = get_dataset(dataset_name)
            languages = get_languages(dataset)
            index = {
                "source": 0,
                "target": 1,
            }[args.half]
            token_analysis = TokenAnalysis.interrogate(dataset.phrases(index), k=args.k, sample=args.sample, language=languages[index], min_tokens=args.min_tokens)
            for phrase, tokens in token_analysis.top_k:
                table.append((len(tokens), phrase, "|".join(tokens)))
            histos.append(token_analysis.token_count_histo)
        print(tabulate.tabulate(table, headers=headers))
        print()
        for histo in histos:
            total = sum(histo.values())
            for token_count in sorted(histo.keys()):
                frequency = histo[token_count]
                print(f"{token_count:2d}: {frequency:6d} ({100.0 * frequency / total:.1f}%)")
    elif args.mode == "juxtapose":
        rng = Random()
        for dataset_name in args.dataset:
            dataset = get_dataset(dataset_name)
            dataset = dataset.shuffle(rng).slice(0, args.k)
            headers = dataset.language_pair
            table = Table(dataset.phrase_pairs, headers=headers)
            table.write(fmt="simple_grid")
    elif args.mode == "create":
        assert args.dataset and len(args.dataset) == 1, "exactly one dataset must be specified in create mode"
        return create_dataset(args.dataset[0], output_dir=args.output, overwrite=args.overwrite)
    elif args.mode == "subset":
        assert args.dataset and len(args.dataset) == 1, "exactly one dataset must be specified in subset mode"
        create_subset(args.dataset[0], args.split, output_dir=args.output, shuffle_seed=args.shuffle, size=args.size)
    else:
        raise NotImplementedError("unsupported mode")
    return 0


if __name__ == "__main__":
    sys.exit(main())
