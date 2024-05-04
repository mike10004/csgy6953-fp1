#!/usr/bin/env python3

import numpy as np
from torch import Tensor
from torchtext.data.utils import get_tokenizer
from unittest import TestCase
from torch.utils.data import DataLoader
from dlfp.tokens import Tokenage
from dlfp.tokens import Specials
from dlfp.tokens import SpecialSymbols
import dlfp_tests.tools
from dlfp_tests.tools import PhrasePairDataset


MULTI30K_DE_EN_DATASETS: dict[str, PhrasePairDataset] = {}  # split -> dataset


def load_multi30k_dataset(split: str = 'train') -> PhrasePairDataset:
    dataset = MULTI30K_DE_EN_DATASETS.get(split, None)
    if dataset is None:
        dataset = dlfp_tests.tools.multi30k_de_en(split=split)
        MULTI30K_DE_EN_DATASETS[split] = dataset
    return dataset


def init_multi30k(dataset: PhrasePairDataset = None) -> Tokenage:
    train_iter = dataset or load_multi30k_dataset(split='train')
    SRC_LANGUAGE, TGT_LANGUAGE = "de", "en"
    assert (SRC_LANGUAGE, TGT_LANGUAGE) == train_iter.language_pair
    t = Tokenage({
        SRC_LANGUAGE: get_tokenizer('spacy', language='de_core_news_sm'),
        TGT_LANGUAGE: get_tokenizer('spacy', language='en_core_web_sm'),
    }, language_pair=(SRC_LANGUAGE, TGT_LANGUAGE), specials=Specials.create())
    t.init_vocab_transform(train_iter)
    return t



def show_examples1(batch_size: int = 4, max_batch: int = 10):
    train_iter = load_multi30k_dataset(split='train')
    raw_dataloader = DataLoader(train_iter, batch_size=batch_size)
    for b_idx, raw_batch in enumerate(raw_dataloader):
        if b_idx >= max_batch:
            break
        print(type(raw_batch), len(raw_batch))
        for i_idx, raw_item in enumerate(raw_batch):
            print(f"{b_idx}:{i_idx}: {raw_item}")


def show_examples2(tokenage: Tokenage, batch_size: int = 4, max_batch: int = 10):
    train_iter = load_multi30k_dataset(split='train')
    cooked_dataloader = DataLoader(train_iter, batch_size=batch_size, collate_fn=tokenage.collate_fn)
    for b_idx, cooked_batch in enumerate(cooked_dataloader):
        if b_idx >= max_batch:
            break
        for i_idx, cooked_item in enumerate(cooked_batch):
            print(f"{b_idx}:{i_idx}: {cooked_item.shape}")





class TokenageTest(TestCase):

    def test_init(self):
        print("loading")
        tokenage = init_multi30k()
        print("loaded")
        train_iter = load_multi30k_dataset(split='train')
        SRC_LANGUAGE = train_iter.language_pair[0]
        for i, (src_phrase, dst_phrase) in enumerate(train_iter):
            if i >= 10:
                break
            tokenized = tokenage.token_transform[SRC_LANGUAGE](src_phrase)
            self.assertSetEqual({str}, set([type(x) for x in tokenized]))
            some_word = tokenized[len(tokenized) // 2]
            vocab = tokenage.vocab_transform[SRC_LANGUAGE]
            some_token = vocab[some_word]
            word = vocab.lookup_token(some_token)
            token = vocab[word]
            self.assertEqual(some_token, token)
            self.assertEqual(some_word, word)

    def test_show_examples(self):
        train_iter = load_multi30k_dataset(split='train')
        p0_de, p0_en = train_iter.phrase_pairs[0]
        batch_size = 2
        tokenage = init_multi30k(train_iter)
        de_tokens = tokenage.token_transform["de"](p0_de)
        de_vocab = tokenage.vocab_transform["de"]
        de_indices = np.array(de_vocab(de_tokens))
        print(de_indices)
        cooked_dataloader = DataLoader(train_iter, batch_size=batch_size, collate_fn=tokenage.collate_fn)
        de_batch, en_batch = next(iter(cooked_dataloader))
        self.assertIsInstance(de_batch, Tensor)
        self.assertIsInstance(en_batch, Tensor)
        print(de_batch.shape)
        print(en_batch.shape)
        t0_de = de_batch[:,0].detach().cpu().numpy()
        actual_p0_de = [de_vocab.lookup_token(t_id) for t_id in t0_de]
        print(actual_p0_de)
        np.testing.assert_array_equal(de_tokens, actual_p0_de[1:len(de_indices)+1])

    def test_specials(self):
        train_iter = load_multi30k_dataset(split='train')
        tokenage = init_multi30k(train_iter)
        for language in tokenage.language_pair:
            vocab = tokenage.vocab_transform[language]
            for token, idx in zip(tokenage.specials.tokens, tokenage.specials.indexes):
                with self.subTest((language, token, idx)):
                    actual_token = vocab.lookup_token(idx)
                    self.assertEqual(token, actual_token)
                    print(f"trying vocab({token}) with token of type {type(token)}")
                    actual_indexes = vocab([token])
                    self.assertListEqual([idx], actual_indexes)




class SpecialSymbolsTest(TestCase):

    def test_as_tuple(self):
        special_symbols = ['<unk>', '<pad>', '<box>', '<eos>']
        self.assertListEqual(special_symbols, SpecialSymbols().as_list())

