#!/usr/bin/env python3

import warnings
from configparser import ConfigParser
from pathlib import Path
from typing import NamedTuple
from typing import Optional
from typing import Callable
from typing import TypeVar
from random import Random

import torch

import dlfp.utils
import dlfp.common
import dlfp.models
from dlfp.datasets import DatasetResolver
from dlfp.utils import Restored
from dlfp.utils import Bilinguist
from dlfp.utils import PhrasePairDataset
from dlfp.utils import LanguageCache
from dlfp.utils import Split


MULTI30K_DE_EN_DATASETS: dict[str, PhrasePairDataset] = {}  # split -> dataset
DEVICE = None
_CUDA_WARNINGS_SUPPRESSED = None
_CFG = None
T = TypeVar("T")
_TEST_MULTI30K_BILINGUIST = None


def get_cfg() -> ConfigParser:
    global _CFG
    if _CFG is None:
        _CFG = load_cfg()
    return _CFG


def load_cfg() -> ConfigParser:
    cfg = ConfigParser()
    cfg.read([str(dlfp.common.get_repo_root() / "tests.ini")])
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
        dlfp.common.noop(spacy)
    for w in warnings_list:
        if w.category is UserWarning and ("CUDA initialization" in str(w) or "Can't initialize NVML" in str(w)):
            pass
        else:
            warnings.warn("unexpected warning: {w}", category=w.category)
    _CUDA_WARNINGS_SUPPRESSED = True


def load_multi30k_dataset(split: Split = 'train') -> PhrasePairDataset:
    dataset = MULTI30K_DE_EN_DATASETS.get(split, None)
    if dataset is None:
        dataset = DatasetResolver().multi30k_de_en(split=split)
        MULTI30K_DE_EN_DATASETS[split] = dataset
    return dataset


def get_multi30k_de_en_bilinguist() -> Bilinguist:
    global _TEST_MULTI30K_BILINGUIST
    if _TEST_MULTI30K_BILINGUIST is None:
        dataset = load_multi30k_dataset(split='train')
        cache = LanguageCache()
        src_lang = cache.get(dataset, "de", "spacy", "de_core_news_sm")
        tgt_lang = cache.get(dataset, "en", "spacy", "en_core_web_sm")
        _TEST_MULTI30K_BILINGUIST = Bilinguist(src_lang, tgt_lang)
    return _TEST_MULTI30K_BILINGUIST


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
    return dlfp.common.get_repo_root() / "data" / "cache"


def get_testdata_dir() -> Path:
    return dlfp.common.get_repo_root() / "dlfp_tests" / "testdata"


class RestoredContainer(NamedTuple):

    model: dlfp.models.Cruciformer
    bilinguist: Bilinguist
    model_hp: dlfp.models.ModelHyperparametry



def load_restored_cruciform(checkpoint_file: Path, device: str, dataset_name: str = "easymark") -> RestoredContainer:
    restored = Restored.from_file(checkpoint_file, device=device)
    train_dataset = DatasetResolver().by_name(dataset_name, "train")
    cache = LanguageCache()
    source = cache.get(train_dataset, "clue", "spacy", "en_core_web_sm")
    target = cache.get(train_dataset, "answer", "spacy", "en_core_web_sm")
    bilinguist = Bilinguist(source, target)
    ok, train_hp, model_hp = dlfp.models.get_hyperparameters(restored)
    if not ok:
        raise ValueError(f"could not extract hyperparameters from {checkpoint_file}")
    model = dlfp.models.create_model(
        src_vocab_size=len(bilinguist.source.vocab),
        tgt_vocab_size=len(bilinguist.target.vocab),
        h=model_hp,
    ).to(device)
    model.load_state_dict(restored.model_state_dict)
    model.eval()
    return RestoredContainer(model, bilinguist, model_hp)
