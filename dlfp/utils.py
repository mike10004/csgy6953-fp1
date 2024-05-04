#!/usr/bin/env python3

from typing import Iterable
from typing import Tuple
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset


# noinspection PyUnusedLocal
def noop(*args, **kwargs):
    pass

def get_repo_root() -> Path:
    return Path(__file__).absolute().parent.parent


def equal_scalar(t: Tensor, s: float) -> Tensor:
    """Tests whether each value of a tensor is equal to a scalar.

    This is exactly what (tensor == scalar) does, but the linter is unwilling to
    believe that == returns a tensor instead of a bool, so this function may be used
    to avoid linter warnings.
    """
    result = t == s
    # noinspection PyTypeChecker
    return result


def generate_square_subsequent_mask(sz, device):
    mask = equal_scalar(torch.triu(torch.ones((sz, sz), device=device)), 1).transpose(0, 1)
    mask = mask.float().masked_fill(equal_scalar(mask, 0), float('-inf')).masked_fill(equal_scalar(mask, 1), float(0.0))
    return mask

class PhrasePairDataset(Dataset[Tuple[str, str]], Iterable[Tuple[str, str]]):

    def __init__(self, phrase_pairs: list[Tuple[str, str]], language_pair: Tuple[str, str]):
        super().__init__()
        self.phrase_pairs = tuple(phrase_pairs)
        self.language_pair = language_pair

    def __getitem__(self, index) -> Tuple[str, str]:
        return self.phrase_pairs[index]

    def __len__(self) -> int:
        return len(self.phrase_pairs)

    def __iter__(self):
        return iter(self.phrase_pairs)

