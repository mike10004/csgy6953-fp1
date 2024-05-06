#!/usr/bin/env python3

import sys

import dlfp.utils
import dlfp.running
from dlfp.running import DataSuperset
from dlfp.running import Runner
from dlfp.utils import Bilinguist
from dlfp.utils import LanguageCache


class DemoDeenRunner(Runner):

    def describe(self) -> str:
        return "Run German-to-English translation demo"

    def resolve_dataset(self) -> DataSuperset:
        train_dataset = dlfp.utils.multi30k_de_en(split='train')
        valid_dataset = dlfp.utils.multi30k_de_en(split='valid')
        return DataSuperset(train_dataset, valid_dataset)

    def create_bilinguist(self, superset: DataSuperset):
        cache = LanguageCache()
        train_dataset = superset.train
        src_lang = cache.get(train_dataset, "de", "spacy", "de_core_news_sm")
        tgt_lang = cache.get(train_dataset, "en", "spacy", "en_core_web_sm")
        bilinguist = Bilinguist(src_lang, tgt_lang)
        return bilinguist



if __name__ == '__main__':
    sys.exit(dlfp.running.main(DemoDeenRunner()))
