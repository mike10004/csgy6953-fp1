#!/usr/bin/env python3

import numpy as np
from torch import Tensor
from unittest import TestCase
from torch.utils.data import DataLoader
import dlfp_tests.tools
from dlfp_tests.tools import load_multi30k_dataset
from dlfp_tests.tools import get_multi30k_de_en_bilinguist


dlfp_tests.tools.suppress_cuda_warning()


def show_examples1(batch_size: int = 4, max_batch: int = 10):
    train_iter = load_multi30k_dataset(split='train')
    raw_dataloader = DataLoader(train_iter, batch_size=batch_size)
    for b_idx, raw_batch in enumerate(raw_dataloader):
        if b_idx >= max_batch:
            break
        print(type(raw_batch), len(raw_batch))
        for i_idx, raw_item in enumerate(raw_batch):
            print(f"{b_idx}:{i_idx}: {raw_item}")



class BiglotTest(TestCase):

    verbose = False

    def test_init(self):
        bilinguist = get_multi30k_de_en_bilinguist()
        train_iter = load_multi30k_dataset(split='train')
        for i, (src_phrase, dst_phrase) in enumerate(train_iter):
            if i >= 10:
                break
            tokenized = bilinguist.source.tokenizer(src_phrase)
            self.assertSetEqual({str}, set([type(x) for x in tokenized]))
            some_word = tokenized[len(tokenized) // 2]
            vocab = bilinguist.source.vocab
            some_token = vocab[some_word]
            word = vocab.lookup_token(some_token)
            token = vocab[word]
            self.assertEqual(some_token, token)
            self.assertEqual(some_word, word)

    def test_show_examples(self):
        train_iter = load_multi30k_dataset(split='train')
        p0_de, p0_en = train_iter.phrase_pairs[0]
        batch_size = 2
        bilinguist = get_multi30k_de_en_bilinguist()
        de_tokens = bilinguist.source.tokenizer(p0_de)
        de_vocab = bilinguist.source.vocab
        de_indices = np.array(de_vocab(de_tokens))
        if self.verbose: print(de_indices)
        cooked_dataloader = DataLoader(train_iter, batch_size=batch_size, collate_fn=bilinguist.collate)
        de_batch, en_batch = next(iter(cooked_dataloader))
        self.assertIsInstance(de_batch, Tensor)
        self.assertIsInstance(en_batch, Tensor)
        if self.verbose: print(de_batch.shape)
        if self.verbose: print(en_batch.shape)
        t0_de = de_batch[:,0].detach().cpu().numpy()
        actual_p0_de = [de_vocab.lookup_token(t_id) for t_id in t0_de]
        if self.verbose: print(actual_p0_de)
        np.testing.assert_array_equal(de_tokens, actual_p0_de[1:len(de_indices)+1])

    def test_specials(self):
        bilinguist = get_multi30k_de_en_bilinguist()
        languages = bilinguist.languages()
        src, tgt = languages
        self.assertIs(src.specials, tgt.specials)
        for language in languages:
            for token, idx in zip(language.specials.tokens, language.specials.indexes):
                with self.subTest((language, token, idx)):
                    actual_token = language.vocab.lookup_token(idx)
                    self.assertEqual(token, actual_token)
                    # print(f"trying vocab({token}) with token of type {type(token)}")
                    actual_indexes = language.vocab([token])
                    self.assertListEqual([idx], actual_indexes)
