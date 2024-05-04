#!/usr/bin/env python3

from typing import Tuple

from torchtext.datasets import Multi30k
import dlfp.utils
from dlfp.utils import PhrasePairDataset





def multi30k_de_en(split: str) -> PhrasePairDataset:
    SRC_LANGUAGE = 'de'
    TGT_LANGUAGE = 'en'
    data_dir = str(dlfp.utils.get_repo_root() / "data")
    language_pair = (SRC_LANGUAGE, TGT_LANGUAGE)
    # noinspection PyTypeChecker
    items: list[Tuple[str, str]] = list(Multi30k(root=data_dir, split=split, language_pair=language_pair))
    return PhrasePairDataset(items, language_pair)
