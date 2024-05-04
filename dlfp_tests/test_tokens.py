
from typing import Iterable
from typing import Sequence
from typing import Tuple

import numpy as np
from torch import Tensor
from torchtext.data.utils import get_tokenizer
from unittest import TestCase
from torchtext.datasets import multi30k, Multi30k
import torch.utils.data.dataset
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from dlfp.tokens import Tokenage
import dlfp.utils

multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'


class PhrasePairDataset(Dataset[Tuple[str, str]], Iterable[Tuple[str, str]]):

    def __init__(self, phrase_pairs: list[Tuple[str, str]]):
        super().__init__()
        self.phrase_pairs = phrase_pairs

    def __getitem__(self, index) -> Tuple[str, str]:
        return self.phrase_pairs[index]

    def __len__(self) -> int:
        return len(self.phrase_pairs)

    def __iter__(self):
        return iter(self.phrase_pairs)


def multi30k_pipe(split: str) -> PhrasePairDataset:
    data_dir = str(dlfp.utils.get_repo_root() / "data")
    # noinspection PyTypeChecker
    items: list[Tuple[str, str]] = list(Multi30k(root=data_dir, split=split, language_pair=(SRC_LANGUAGE, TGT_LANGUAGE)))
    return PhrasePairDataset(items)

def init_multi30k(dataset: PhrasePairDataset = None) -> Tokenage:
    train_iter = dataset or multi30k_pipe(split='train')
    t = Tokenage({
        SRC_LANGUAGE: get_tokenizer('spacy', language='de_core_news_sm'),
        TGT_LANGUAGE: get_tokenizer('spacy', language='en_core_web_sm'),
    }, language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    t.init_vocab_transform(train_iter)
    return t



def show_examples1(batch_size: int = 4, max_batch: int = 10):
    train_iter = multi30k_pipe(split='train')
    raw_dataloader = DataLoader(train_iter, batch_size=batch_size)
    for b_idx, raw_batch in enumerate(raw_dataloader):
        if b_idx >= max_batch:
            break
        print(type(raw_batch), len(raw_batch))
        for i_idx, raw_item in enumerate(raw_batch):
            print(f"{b_idx}:{i_idx}: {raw_item}")


def show_examples2(tokenage: Tokenage, batch_size: int = 4, max_batch: int = 10):
    train_iter = multi30k_pipe(split='train')
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
        train_iter = multi30k_pipe(split='train')
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
        train_iter = multi30k_pipe(split='train')
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
