#!/usr/bin/env python3

import sys

import dlfp.running
from dlfp.datasets import DatasetResolver
from dlfp.running import DataSuperset
from dlfp.running import Runnable
from dlfp.running import Runner
from dlfp.translate import CruciformerNodeNavigator
from dlfp.utils import Bilinguist
from dlfp.utils import LanguageCache


class CruciformerRunner(Runner):

    def describe(self) -> str:
        return "Crossword Clue-Answer Translation"

    def resolve_dataset(self, dataset_name: str = None) -> DataSuperset:
        resolver = DatasetResolver()
        dataset_name = dataset_name or "benchmark"
        train = resolver.by_name(dataset_name, split='train')
        valid = resolver.by_name(dataset_name, split='valid')
        return DataSuperset(train, valid)

    def create_bilinguist(self, superset: DataSuperset) -> Bilinguist:
        cache = LanguageCache()
        assert ("clue", "answer") == superset.train.language_pair
        source = cache.get(superset.train, "clue", "spacy", "en_core_web_sm")
        target = cache.get(superset.train, "answer", "spacy", "en_core_web_sm")
        return Bilinguist(source, target)

    def create_runnable(self, device: str) -> Runnable:
        r = super().create_runnable(device)
        def to_crossword_answer(answer: str) -> str:
            answer = answer.replace(" ", "")
            answer = answer.upper()
            return answer
        r.manager.tgt_transform = to_crossword_answer
        r.manager.node_navigator = CruciformerNodeNavigator()
        return r


if __name__ == '__main__':
    sys.exit(dlfp.running.main(CruciformerRunner()))
