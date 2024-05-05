#!/usr/bin/env python3

from random import Random
from unittest import TestCase
import torch
import torchtext.data.utils
from torchtext.vocab import Vocab

import dlfp_tests.tools
from dlfp.tokens import Tokenage
from dlfp.datasets import DatasetResolver
from dlfp.utils import VocabCache

dlfp_tests.tools.suppress_cuda_warning()

class DatasetResolverTest(TestCase):

    def test_benchmark(self):
        resolver = DatasetResolver.default()
        dataset = resolver.benchmark(split="train")
        vocab_cache = VocabCache()
        for language in dataset.language_pair:
            vocab = vocab_cache.get(dataset, language, tokenizer_name="spacy", tokenizer_language="en_core_web_sm")
            print(f"{language:<8} {len(vocab)}")
            vocab_words = sorted(vocab.get_stoi().keys())
            rng = Random(3812531)
            rng.shuffle(vocab_words)
            sample = vocab_words[:10]
            print(sample)


