#!/usr/bin/env python3

import sys

import torch

import dlfp.utils
import dlfp.running
from dlfp.running import DataSuperset
from dlfp.running import Runner
from dlfp.tokens import Biglot
from dlfp.tokens import Linguist
from dlfp.utils import LanguageCache


class DemoDeenRunner(Runner):

    def describe(self) -> str:
        return "Run German-to-English translation demo"

    def resolve_dataset(self) -> DataSuperset:
        train_dataset = dlfp.utils.multi30k_de_en(split='train')
        valid_dataset = dlfp.utils.multi30k_de_en(split='valid')
        return DataSuperset(train_dataset, valid_dataset)

    def create_biglot(self, superset: DataSuperset):
        cache = LanguageCache()
        train_dataset = superset.train
        src_ling = Linguist.from_language(cache.get(train_dataset, "de", "spacy", "de_core_news_sm"))
        tgt_ling = Linguist.from_language(cache.get(train_dataset, "en", "spacy", "en_core_web_sm"))
        biglot = Biglot(src_ling, tgt_ling)
        return biglot



if __name__ == '__main__':
    sys.exit(dlfp.running.main(DemoDeenRunner()))
