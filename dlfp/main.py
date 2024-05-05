#!/usr/bin/env python3

import sys

import dlfp.running
from dlfp.datasets import DatasetResolver
from dlfp.running import DataSuperset
from dlfp.running import Runner
from dlfp.tokens import Linguist
from dlfp.tokens import Biglot
from dlfp.utils import LanguageCache


class CruciformerRunner(Runner):

    def describe(self) -> str:
        return "Crossword Clue-Answer Translation"

    def resolve_dataset(self) -> DataSuperset:
        train = DatasetResolver.default().benchmark(split='train')
        valid = DatasetResolver.default().benchmark(split='valid')
        return DataSuperset(train, valid)

    def create_biglot(self, superset: DataSuperset):
        cache = LanguageCache()
        assert ("clue", "answer") == superset.train.language_pair
        source = Linguist.from_language(cache.get(superset.train, "clue", "spacy", "en_core_web_sm"))
        target = Linguist.from_language(cache.get(superset.train, "answer", "spacy", "en_core_web_sm"))
        return Biglot(source, target)


if __name__ == '__main__':
    sys.exit(dlfp.running.main(CruciformerRunner()))
