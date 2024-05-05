#!/usr/bin/env python3

import warnings
from configparser import ConfigParser
from pathlib import Path
from typing import Optional
from typing import Callable
from typing import TypeVar
from random import Random

import torch
from dlfp.utils import Specials
from dlfp.tokens import Biglot
from dlfp.tokens import Linguist
import dlfp.utils
from dlfp.utils import PhrasePairDataset
from dlfp.utils import VocabCache


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


def load_multi30k_dataset(split: str = 'train') -> PhrasePairDataset:
    dataset = MULTI30K_DE_EN_DATASETS.get(split, None)
    if dataset is None:
        dataset = dlfp.utils.multi30k_de_en(split=split)
        MULTI30K_DE_EN_DATASETS[split] = dataset
    return dataset


def init_multi30k_de_en_tokenage() -> Biglot:
    global _TEST_TOKENAGE
    if _TEST_TOKENAGE is None:
        dataset = load_multi30k_dataset(split='train')
        cache = VocabCache()
        src_lang = cache.get(dataset, "de", "spacy", "de_core_news_sm")
        tgt_lang = cache.get(dataset, "en", "spacy", "en_core_web_sm")
        src_ling = Linguist.from_language(src_lang)
        tgt_ling = Linguist.from_language(tgt_lang)
        _TEST_TOKENAGE = Biglot(src_ling, tgt_ling)
    return _TEST_TOKENAGE


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
    return PhrasePairDataset(dataset.name, phrase_pairs[:size], language_pair=dataset.language_pair)


def _cache_dir() -> Path:
    return dlfp.utils.get_repo_root() / "data" / "cache"


