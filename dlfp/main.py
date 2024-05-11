#!/usr/bin/env python3

import sys
from typing import Optional
from typing import Sequence

import dlfp.running
from dlfp.datasets import DatasetResolver
from dlfp.models import ModelHyperparametry
from dlfp.running import DataSuperset
from dlfp.running import Runnable
from dlfp.running import Runner
from dlfp.running import NodeStrategy
from dlfp.translate import CruciformerCharmarkNodeNavigator
from dlfp.translate import CruciformerNodeNavigator
from dlfp.translate import CruciformerOnemarkNodeNavigator
from dlfp.translate import NodeVisitor
from dlfp.translate import NodeVisitorFactory
from dlfp.utils import Bilinguist
from dlfp.utils import LanguageCache


SUPPORTED_DATASETS = ("easymark", "onemark")

class CruciformerRunner(Runner):

    def describe(self) -> str:
        return "Crossword Clue-Answer Translation"

    def resolve_dataset(self, dataset_name: str = None) -> DataSuperset:
        resolver = DatasetResolver()
        if not dataset_name:
            raise ValueError(f"dataset name must be specified for Cruciformer training; supported: {SUPPORTED_DATASETS}")
        train = resolver.by_name(dataset_name, split='train')
        valid = resolver.by_name(dataset_name, split='valid')
        src_tokenizer_name, src_tokenizer_language = "spacy", "en_core_web_sm"
        tgt_tokenizer_name, tgt_tokenizer_language = "spacy", "en_core_web_sm"
        if dataset_name == "charmark":
            tgt_tokenizer_name, tgt_tokenizer_language = "dlfp", "character"
        return DataSuperset(train, valid, src_tokenizer_name, src_tokenizer_language, tgt_tokenizer_name, tgt_tokenizer_language)

    def create_bilinguist(self, superset: DataSuperset) -> Bilinguist:
        cache = LanguageCache()
        assert ("clue", "answer") == superset.train.language_pair
        source = cache.get(superset.train, "clue", superset.src_tokenizer_name, superset.src_tokenizer_language)
        target = cache.get(superset.train, "answer", superset.tgt_tokenizer_name, superset.tgt_tokenizer_language)
        return Bilinguist(source, target)

    def create_runnable(self, dataset_name: str, h: ModelHyperparametry, device: str) -> Runnable:
        r = super().create_runnable(dataset_name, h, device)
        r.manager.tgt_transform = dlfp.utils.normalize_answer_upper
        return r

    def create_node_strategy(self, strategy_spec: str, dataset_name: str) -> NodeStrategy:
        navigator_type = {
            "easymark": CruciformerNodeNavigator,
            "onemark": CruciformerOnemarkNodeNavigator,
            "charmark": CruciformerCharmarkNodeNavigator,
        }[dataset_name]
        visitor_factory: NodeVisitorFactory = NodeVisitor
        return NodeStrategy(navigator=navigator_type(), visitor_factory=visitor_factory)


def main(argv1: Optional[Sequence[str]]) -> int:
    return dlfp.running.main(CruciformerRunner(), argv1=argv1)


if __name__ == '__main__':
    sys.exit(main())
