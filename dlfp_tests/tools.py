#!/usr/bin/env python3

from unittest import TestCase
from typing import Tuple

import torch
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from dlfp.tokens import Specials
from dlfp.tokens import Tokenage
import dlfp.utils
from dlfp.utils import PhrasePairDataset


MULTI30K_DE_EN_DATASETS: dict[str, PhrasePairDataset] = {}  # split -> dataset
DEVICE = None


def multi30k_de_en(split: str) -> PhrasePairDataset:
    SRC_LANGUAGE = 'de'
    TGT_LANGUAGE = 'en'
    data_dir = str(dlfp.utils.get_repo_root() / "data")
    language_pair = (SRC_LANGUAGE, TGT_LANGUAGE)
    # noinspection PyTypeChecker
    items: list[Tuple[str, str]] = list(Multi30k(root=data_dir, split=split, language_pair=language_pair))
    return PhrasePairDataset(items, language_pair)



def load_multi30k_dataset(split: str = 'train') -> PhrasePairDataset:
    dataset = MULTI30K_DE_EN_DATASETS.get(split, None)
    if dataset is None:
        dataset = multi30k_de_en(split=split)
        MULTI30K_DE_EN_DATASETS[split] = dataset
    return dataset


def init_multi30k_de_en_tokenage(dataset: PhrasePairDataset = None) -> Tokenage:
    train_iter = dataset or load_multi30k_dataset(split='train')
    SRC_LANGUAGE, TGT_LANGUAGE = "de", "en"
    assert (SRC_LANGUAGE, TGT_LANGUAGE) == train_iter.language_pair
    t = Tokenage({
        SRC_LANGUAGE: get_tokenizer('spacy', language='de_core_news_sm'),
        TGT_LANGUAGE: get_tokenizer('spacy', language='en_core_web_sm'),
    }, language_pair=(SRC_LANGUAGE, TGT_LANGUAGE), specials=Specials.create())
    t.init_vocab_transform(train_iter)
    return t


def get_device() -> str:
    global DEVICE
    if DEVICE is None:
        DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    return DEVICE

