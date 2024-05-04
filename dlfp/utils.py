#!/usr/bin/env python3

import os
from datetime import datetime
from typing import Iterable
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from pathlib import Path

import torch
from torch import nn
from torch import Tensor
from torch.optim import Optimizer
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


def timestamp() -> str:
    return datetime.now().strftime("%y%m%d-%H%M")


class EpochResult(NamedTuple):

    epoch_index: int
    train_loss: float
    valid_loss: float


class Checkpointer:

    def __init__(self,
                 checkpoints_dir: Path,
                 model: nn.Module,
                 optimizer: Optional[Optimizer] = None):
        self.checkpoints_dir = checkpoints_dir
        self.model = model
        self.optimizer = optimizer
        self.only_min_valid_loss = False
        self.min_valid_loss = None
        self._previous_checkpoint_file = None
        self.retain_all = False
        self._epoch_results = []

    def is_checkpointable(self, epoch_result: EpochResult) -> bool:
        if not self.only_min_valid_loss:
            return True
        if self.min_valid_loss is None:
            return True
        if epoch_result.valid_loss < self.min_valid_loss:
            self.min_valid_loss = epoch_result.valid_loss
            return True
        return False

    def checkpoint(self, epoch_result: EpochResult):
        self._epoch_results.append(epoch_result._asdict())
        if self.is_checkpointable(epoch_result):
            checkpoint = {
                'epoch_results': self._epoch_results,
                'model_state_dict': self.model.state_dict(),
            }
            if self.optimizer is not None:
                checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            checkpoint_file = self.checkpoints_dir / f"checkpoint-epoch{epoch_result.epoch_index:03d}.pt"
            checkpoint_file.parent.mkdir(exist_ok=True, parents=True)
            torch.save(checkpoint, str(checkpoint_file))
            if self._previous_checkpoint_file is not None:
                try:
                    os.remove(self._previous_checkpoint_file)
                except IOError:
                    pass
            self._previous_checkpoint_file = checkpoint_file
