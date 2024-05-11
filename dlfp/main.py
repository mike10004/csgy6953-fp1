#!/usr/bin/env python3

import sys

import dlfp.running
from dlfp.datasets import DatasetResolver
from dlfp.models import ModelHyperparametry
from dlfp.running import DataSuperset
from dlfp.running import Runnable
from dlfp.running import Runner
from dlfp.translate import CruciformerNodeNavigator
from dlfp.translate import CruciformerOnemarkNodeNavigator
from dlfp.translate import NodeNavigator
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
        target = cache.get(superset.train, "answer", superset.src_tokenizer_name, superset.src_tokenizer_language)
        return Bilinguist(source, target)

    def create_runnable(self, dataset_name: str, h: ModelHyperparametry, device: str) -> Runnable:
        r = super().create_runnable(dataset_name, h, device)
        r.manager.tgt_transform = dlfp.utils.normalize_answer_upper
        return r

    def create_navigator(self, navigator_spec: str, dataset_name: str) -> NodeNavigator:
        navigator_type = CruciformerNodeNavigator
        if dataset_name == "onemark":
            navigator_type = CruciformerOnemarkNodeNavigator
        return navigator_type()


def main() -> int:
    return dlfp.running.main(CruciformerRunner())


if __name__ == '__main__':
    sys.exit(main())
