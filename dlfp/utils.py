#!/usr/bin/env python3

import os
from datetime import datetime
from typing import Iterable
from typing import Any
from typing import Literal
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Callable
from typing import Tuple
from typing import TypeVar
from pathlib import Path

import torch
import torchtext.vocab
import torchtext.data.utils
from torchtext.vocab import Vocab
from torch import nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data.dataset import Dataset

T = TypeVar("T")
Split = Literal["train", "valid", "test"]
Tokenizer = Callable[[str], Sequence[str]]


# noinspection PyUnusedLocal
def noop(*args, **kwargs):
    pass

def identity(x: T) -> T:
    return x


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


class SpecialIndexes(NamedTuple):

    unk: int = 0
    pad: int = 1
    bos: int = 2
    eos: int = 3


class SpecialSymbols(NamedTuple):

    unk: str = '<unk>'
    pad: str = '<pad>'
    bos: str = '<bos>'
    eos: str = '<eos>'

    def as_list(self) -> list[str]:
        # noinspection PyTypeChecker
        return list(self)


class Specials(NamedTuple):

    indexes: SpecialIndexes
    tokens: SpecialSymbols

    @staticmethod
    def create() -> 'Specials':
        return Specials(SpecialIndexes(), SpecialSymbols())


class PhrasePairDataset(Dataset[Tuple[str, str]], Iterable[Tuple[str, str]]):

    def __init__(self, name: str, phrase_pairs: list[Tuple[str, str]], language_pair: Tuple[str, str]):
        super().__init__()
        self.name = name
        self.phrase_pairs = tuple(phrase_pairs)
        self.language_pair = language_pair

    def __getitem__(self, index) -> Tuple[str, str]:
        return self.phrase_pairs[index]

    def __len__(self) -> int:
        return len(self.phrase_pairs)

    def __iter__(self):
        return iter(self.phrase_pairs)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


class EpochResult(NamedTuple):

    epoch_index: int
    train_loss: float
    valid_loss: float


class Restored(NamedTuple):

    epoch_results: list[EpochResult]
    model_state_dict: dict[str, Any]
    optimizer_state_dict: Optional[dict[str, Any]] = None

    @staticmethod
    def from_checkpoint(checkpoint: dict[str, Any]) -> 'Restored':
        epoch_results = [EpochResult(**d) for d in checkpoint['epoch_results']]
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint.get('optimizer_state_dict', None)
        return Restored(epoch_results, model_state_dict, optimizer_state_dict)

    @staticmethod
    def from_file(checkpoint_file: Path, device=None) -> 'Restored':
        checkpoint = torch.load(str(checkpoint_file), map_location=device)
        return Restored.from_checkpoint(checkpoint)


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
        self.quiet = False

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
        def _print(*args, **kwargs):
            if not self.quiet:
                print(f"epoch {epoch_result.epoch_index:2d}:", *args, **kwargs)
        _print(f"train loss {epoch_result.train_loss:.4f}; valid loss {epoch_result.valid_loss:.4f}")
        self._epoch_results.append(epoch_result._asdict())
        if not self.is_checkpointable(epoch_result):
            return
        checkpoint = {
            'epoch_results': self._epoch_results,
            'model_state_dict': self.model.state_dict(),
        }
        if self.optimizer is not None:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        checkpoint_file = self.checkpoints_dir / f"checkpoint-epoch{epoch_result.epoch_index:03d}.pt"
        checkpoint_file.parent.mkdir(exist_ok=True, parents=True)
        torch.save(checkpoint, str(checkpoint_file))
        message = f"saved checkpoint {checkpoint_file.relative_to(self.checkpoints_dir)}"
        if self._previous_checkpoint_file is not None:
            try:
                os.remove(self._previous_checkpoint_file)
                message = f"{message}; deleted previous"
            except IOError:
                pass
        self._previous_checkpoint_file = checkpoint_file
        _print(message)


def multi30k_de_en(split: str) -> PhrasePairDataset:
    from torchtext.datasets import Multi30k
    SRC_LANGUAGE = 'de'
    TGT_LANGUAGE = 'en'
    data_dir = str(get_repo_root() / "data")
    language_pair = (SRC_LANGUAGE, TGT_LANGUAGE)
    # noinspection PyTypeChecker
    items: list[Tuple[str, str]] = list(Multi30k(root=data_dir, split=split, language_pair=language_pair))
    return PhrasePairDataset("multi30k_de_en", items, language_pair)


class Language(NamedTuple):

    name: str
    tokenizer: Tokenizer
    vocab: Vocab
    specials: Specials

    def to_tensor(self, token_ids: list[int]):
        """Add BOS/EOS and create tensor for input sequence indices."""
        return torch.cat((torch.tensor([self.specials.indexes.bos]),
                          torch.tensor(token_ids),
                          torch.tensor([self.specials.indexes.eos])))
class LanguageCache:

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or (get_repo_root() / "data" / "cache" / "vocab")
        self.specials = Specials.create()

    def get(self, dataset: PhrasePairDataset, dataset_language: str, tokenizer_name: str, tokenizer_language: str) -> Language:
        tokenizer = torchtext.data.utils.get_tokenizer(tokenizer=tokenizer_name, language=tokenizer_language)
        directory = self.cache_dir / tokenizer_name / tokenizer_language
        language_index = dataset.language_pair.index(dataset_language)
        vocab_file = directory / f"{dataset.name}-{dataset_language}.vocab.pt"
        try:
            vocab = torch.load(str(vocab_file))
        except FileNotFoundError:
            vocab = None
        if vocab is not None:
            return self.to_language(dataset_language, vocab, tokenizer)
        phrases = [phrase_pair[language_index] for phrase_pair in dataset]
        vocab = self.build_vocab(phrases, tokenizer)
        vocab.set_default_index(self.specials.indexes.unk)
        vocab_file.parent.mkdir(exist_ok=True, parents=True)
        torch.save(vocab, str(vocab_file))
        return self.to_language(dataset_language, vocab, tokenizer)

    def to_language(self, name: str, vocab: Vocab, tokenizer: Tokenizer) -> Language:
        return Language(
            name=name,
            tokenizer=tokenizer,
            vocab=vocab,
            specials=self.specials,
        )

    def build_vocab(self, phrases: Iterable[str], tokenizer: Tokenizer) -> Vocab:
        def _yield_tokens():
            for phrase in phrases:
                yield tokenizer(phrase)
        vocab = torchtext.vocab.build_vocab_from_iterator(_yield_tokens(), specials=self.specials.tokens.as_list())
        return vocab
