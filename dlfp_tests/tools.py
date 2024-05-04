#!/usr/bin/env python3

import warnings
from configparser import ConfigParser
from pathlib import Path
from typing import Optional
from typing import Callable
from typing import TypeVar
from random import Random
from typing import Tuple

import torch
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from dlfp.tokens import Specials
from dlfp.tokens import Tokenage
import dlfp.utils
from dlfp.tokens import Tokenizer
from dlfp.utils import PhrasePairDataset


MULTI30K_DE_EN_DATASETS: dict[str, PhrasePairDataset] = {}  # split -> dataset
DEVICE = None
_CUDA_WARNINGS_SUPPRESSED = None
_CFG = None
T = TypeVar("T")
_TEST_TOKENAGE = None


def get_cfg() -> ConfigParser:
    global _CFG
    if _CFG is None:
        _CFG = load_cfg()
    return _CFG


def load_cfg() -> ConfigParser:
    cfg = ConfigParser()
    cfg.read([str(dlfp.utils.get_repo_root() / "tests.ini")])
    return cfg


def cfg_value(key: str, section: str = "common", typer: Callable[[str], T] = None) -> Optional[T]:
    typer = typer or str
    cfg = get_cfg()
    value = cfg.get(section, key, fallback=None)
    if value is not None:
        value = typer(value)
    return value


def suppress_cuda_warning():
    global _CUDA_WARNINGS_SUPPRESSED
    if _CUDA_WARNINGS_SUPPRESSED is not None:
        return
    if not cfg_value("suppress_cuda_warning", typer=bool):
        _CUDA_WARNINGS_SUPPRESSED = False
        return
    with warnings.catch_warnings(record=True) as warnings_list:
        import torch.cuda
        torch.cuda.is_available()
        import spacy
        dlfp.utils.noop(spacy)
    for w in warnings_list:
        if w.category is UserWarning and ("CUDA initialization" in str(w) or "Can't initialize NVML" in str(w)):
            pass
        else:
            warnings.warn("unexpected warning: {w}", category=w.category)
    _CUDA_WARNINGS_SUPPRESSED = True


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


def init_multi30k_de_en_tokenage() -> Tokenage:
    global _TEST_TOKENAGE
    if _TEST_TOKENAGE is None:
        train_iter = load_multi30k_dataset(split='train')
        vocab_cache_file = _cache_dir() / "vocab_cache.pkl"
        vocab_transform = None
        try:
            vocab_transform = torch.load(str(vocab_cache_file))
        except FileNotFoundError:
            pass
        if vocab_transform is None:
            _TEST_TOKENAGE = _first_init_multi30k_de_en_tokenage(train_iter)
            vocab_cache_file.parent.mkdir(exist_ok=True, parents=True)
            torch.save(_TEST_TOKENAGE.vocab_transform, str(vocab_cache_file))
        else:
            SRC_LANGUAGE, TGT_LANGUAGE = "de", "en"
            language_pair = SRC_LANGUAGE, TGT_LANGUAGE
            tokenizers = _get_tokenizers()
            _TEST_TOKENAGE = Tokenage(
                language_pair=language_pair,
                token_transform=_get_tokenizers(),
                vocab_transform=vocab_transform,
                text_transform={
                    SRC_LANGUAGE: Tokenage.sequential_transforms(
                        tokenizers[SRC_LANGUAGE], vocab_transform[SRC_LANGUAGE], Tokenage.tensor_transform,
                    ),
                    TGT_LANGUAGE: Tokenage.sequential_transforms(
                        tokenizers[TGT_LANGUAGE], vocab_transform[TGT_LANGUAGE], Tokenage.tensor_transform,
                    ),
            }, specials=Specials.create())
    return _TEST_TOKENAGE


def _get_tokenizers() -> dict[str, Tokenizer]:
    SRC_LANGUAGE, TGT_LANGUAGE = "de", "en"
    return {
        SRC_LANGUAGE: get_tokenizer('spacy', language='de_core_news_sm'),
        TGT_LANGUAGE: get_tokenizer('spacy', language='en_core_web_sm'),
    }

def _first_init_multi30k_de_en_tokenage(train_iter: PhrasePairDataset) -> Tokenage:
    SRC_LANGUAGE, TGT_LANGUAGE = "de", "en"
    language_pair = SRC_LANGUAGE, TGT_LANGUAGE
    assert (SRC_LANGUAGE, TGT_LANGUAGE) == train_iter.language_pair
    t = Tokenage.from_token_transform(language_pair, _get_tokenizers(), train_iter)
    return t


def get_device() -> str:
    global DEVICE
    if DEVICE is None:
        DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    return DEVICE


def truncate_dataset(dataset: PhrasePairDataset, size: int, shuffle_seed: Optional[int] = ...) -> PhrasePairDataset:
    phrase_pairs = dataset.phrase_pairs
    if not shuffle_seed is ...:
        rng = Random(shuffle_seed)
        phrase_pairs = list(phrase_pairs)
        rng.shuffle(phrase_pairs)
    return PhrasePairDataset(phrase_pairs[:size], language_pair=dataset.language_pair)


def _cache_dir() -> Path:
    return dlfp.utils.get_repo_root() / "data" / "cache"


