#!/usr/bin/env python3

import os
import math
import logging
from random import Random
from typing import Iterable
from typing import Any
from typing import Iterator
from typing import Union
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
import torch.nn.utils.rnn
from torchtext.vocab import Vocab
from torch import nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data.dataset import Dataset

import dlfp.common

_log = logging.getLogger(__name__)
T = TypeVar("T")
Pathish = Union[Path, str]
Split = Literal["train", "valid", "test"]
Tokenizer = Callable[[str], Sequence[str]]
TextTransform = Callable[[str], Tensor]
PhrasePair = tuple[str, str]



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

    def to_tensor(self, token_ids: list[int]):
        """Add BOS/EOS and create tensor for input sequence indices."""
        return torch.cat((torch.tensor([self.indexes.bos]),
                          torch.tensor(token_ids),
                          torch.tensor([self.indexes.eos])))

class PhrasePairDataset(Dataset[PhrasePair], Iterable[PhrasePair]):

    def __init__(self,
                 name: str,
                 phrase_pairs: Sequence[PhrasePair],
                 language_pair: tuple[str, str],
                 data_hash_pair: tuple[str, str] = None):
        super().__init__()
        self.name = name
        self.phrase_pairs = tuple(phrase_pairs)
        self.language_pair = language_pair
        self.data_hash_pair = data_hash_pair

    def __getitem__(self, index) -> Tuple[str, str]:
        return self.phrase_pairs[index]

    def __len__(self) -> int:
        return len(self.phrase_pairs)

    def __iter__(self):
        return iter(self.phrase_pairs)

    def source_phrases(self) -> Iterator[str]:
        yield from self.phrases(0)

    def target_phrases(self) -> Iterator[str]:
        yield from self.phrases(1)

    def phrases(self, index: int) -> Iterator[str]:
        for phrase_pair in self:
            yield phrase_pair[index]

    def shuffle(self, rng: Random) -> 'PhrasePairDataset':
        phrase_pairs = list(self.phrase_pairs)
        rng.shuffle(phrase_pairs)
        return PhrasePairDataset(self.name, phrase_pairs, self.language_pair, self.data_hash_pair)

    def slice(self, start_inclusive: int, stop_exclusive: int) -> 'PhrasePairDataset':
        return PhrasePairDataset(self.name, self.phrase_pairs[start_inclusive:stop_exclusive], self.language_pair, self.data_hash_pair)

    def partition(self, count: int) -> list['PhrasePairDataset']:
        partition_size = int(math.ceil(len(self) / count))
        assert partition_size >= 1
        parts = []
        start = 0
        while len(parts) < count:
            part = self.slice(start, start + partition_size)
            parts.append(part)
        return parts

    def normalize_answers(self, new_name: str = None) -> 'PhrasePairDataset':
        new_name = new_name or f"{self.name}_norm"
        phrase_pairs = []
        for clue, answer in self.phrase_pairs:
            norm_answer = normalize_answer(answer)
            phrase_pairs.append((clue, norm_answer))
        return PhrasePairDataset(new_name, phrase_pairs, self.language_pair, data_hash_pair=None)

    def filter(self, predicate: Callable[[PhrasePair], bool]) -> 'PhrasePairDataset':
        phrase_pairs = filter(predicate, self.phrase_pairs)
        return PhrasePairDataset(self.name, phrase_pairs, self.language_pair, self.data_hash_pair)


class EpochResult(NamedTuple):

    epoch_index: int
    train_loss: float
    valid_loss: float
    last_epoch: bool = False

    def to_row(self) -> tuple[int, float, float]:
        return self.epoch_index, self.train_loss, self.valid_loss

    @staticmethod
    def headers() -> tuple[str, str, str]:
        return "epoch_index", "train_loss", "valid_loss"


class Restored(NamedTuple):

    epoch_results: list[EpochResult]
    model_state_dict: dict[str, Any]
    optimizer_state_dict: Optional[dict[str, Any]] = None
    extra: Optional[dict[str, Any]] = None

    @staticmethod
    def from_checkpoint(checkpoint: dict[str, Any]) -> 'Restored':
        epoch_results = [EpochResult(**d) for d in checkpoint['epoch_results']]
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint.get('optimizer_state_dict', None)
        extra = checkpoint.get('extra', {})
        return Restored(epoch_results, model_state_dict, optimizer_state_dict, extra)

    @staticmethod
    def from_file(checkpoint_file: Path, device=None) -> 'Restored':
        checkpoint = torch.load(str(checkpoint_file), map_location=device)
        return Restored.from_checkpoint(checkpoint)

    def model_param_shapes(self) -> dict[str, tuple[int, ...]]:
        # torch.Size([93559, 512]) src_tok_emb.embedding.weight
        # torch.Size([55145, 512]) tgt_tok_emb.embedding.weight
        d = {}
        for key, param in self.model_state_dict.items():
            if isinstance(param, Tensor):
                d[key] = tuple(param.shape)
        return d


class Checkpointable(NamedTuple):

    epoch_result: EpochResult
    model: nn.Module
    optimizer: Optimizer


class Checkpointer:

    def __init__(self,
                 checkpoints_dir: Path):
        self.checkpoints_dir = checkpoints_dir
        self.only_min_valid_loss = False
        self.min_valid_loss = None
        self._min_valid_loss_checkpoint_file = None
        self._previous_checkpoint_files = set()
        self.retain_all = False
        self._epoch_results = []
        self.quiet = False
        self.extra = None
        self.save_optimizer = False

    def is_saveworthy(self, is_min_valid_loss: bool, last_epoch: bool) -> bool:
        if last_epoch:
            return True
        if not self.only_min_valid_loss:
            return True
        if is_min_valid_loss:
            return True
        return False

    def checkpoint(self, ckpt: Checkpointable):
        epoch_result = ckpt.epoch_result
        def _print(*args, **kwargs):
            if not self.quiet:
                print(f"epoch {epoch_result.epoch_index:2d}:", *args, **kwargs)
        _print(f"train loss {epoch_result.train_loss:.4f}; valid loss {epoch_result.valid_loss:.4f}")
        self._epoch_results.append(epoch_result._asdict())
        is_min_valid_loss = False
        if self.min_valid_loss is None or epoch_result.valid_loss < self.min_valid_loss:
            self.min_valid_loss = epoch_result.valid_loss
            is_min_valid_loss = True
        if not self.is_saveworthy(is_min_valid_loss, epoch_result.last_epoch):
            return
        checkpoint = {
            'epoch_results': self._epoch_results,
            'model_state_dict': ckpt.model.state_dict(),
        }
        if self.extra is not None:
            checkpoint['extra'] = self.extra
        if self.save_optimizer:
            checkpoint['optimizer_state_dict'] = ckpt.optimizer.state_dict()
        checkpoint_file = self.checkpoints_dir / f"checkpoint-epoch{epoch_result.epoch_index:03d}.pt"
        checkpoint_file.parent.mkdir(exist_ok=True, parents=True)
        torch.save(checkpoint, str(checkpoint_file))
        message = f"saved checkpoint {checkpoint_file.relative_to(self.checkpoints_dir)}"
        if is_min_valid_loss:
            message = f"{message} (min valid loss)"
        deleted = []
        for previous_checkpoint_file in self._previous_checkpoint_files:
            if previous_checkpoint_file != self._min_valid_loss_checkpoint_file:
                try:
                    os.remove(previous_checkpoint_file)
                    deleted.append(previous_checkpoint_file)
                except IOError:
                    pass
        if deleted:
            message = f"{message} (deleted previous: {[f.relative_to(self.checkpoints_dir).as_posix() for f in deleted]})"
        self._previous_checkpoint_files.add(checkpoint_file)
        if is_min_valid_loss:
            self._min_valid_loss_checkpoint_file = checkpoint_file
        _print(message)


class Language(NamedTuple):

    name: str
    tokenizer: Tokenizer
    vocab: Vocab
    text_transform: TextTransform
    specials: Specials

    @staticmethod
    def from_parts(name: str, tokenizer: Tokenizer, vocab: Vocab, specials: Specials):
        text_transform = Language.compose([
            tokenizer,
            vocab,
            specials.to_tensor,
        ])
        return Language(name, tokenizer, vocab, text_transform, specials)

    @staticmethod
    def compose(transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func


def tokenize_characters(text: str) -> list[str]:
    return list(text)


class LanguageCache:

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or (dlfp.common.get_repo_root() / "data" / "cache" / "vocab")
        self.specials = Specials.create()

    @staticmethod
    def get_tokenizer(tokenizer_name: str, tokenizer_language: str) -> Tokenizer:
        if tokenizer_name == "dlfp":
            if tokenizer_language == "character":
                return tokenize_characters  # tokenize 'ABC' as ('A', 'B', 'C')
            raise ValueError("unrecognized tokenizer language")
        tokenizer = torchtext.data.utils.get_tokenizer(tokenizer=tokenizer_name, language=tokenizer_language)
        return tokenizer

    def get(self, dataset: PhrasePairDataset, dataset_language: str, tokenizer_name: str, tokenizer_language: str) -> Language:
        tokenizer = self.get_tokenizer(tokenizer_name, tokenizer_language)
        directory = self.cache_dir / tokenizer_name / tokenizer_language
        language_index = dataset.language_pair.index(dataset_language)
        phrases = [phrase_pair[language_index] for phrase_pair in dataset]
        if not dataset.data_hash_pair:
            _log.info("dataset lacks hashes; not using cache")
            vocab = self.build_vocab(phrases, tokenizer)
            return self.to_language(dataset_language, vocab, tokenizer)
        data_hash = dataset.data_hash_pair[language_index]
        vocab_file = directory / f"{dataset.name}-{dataset_language}.vocab.{data_hash}.pt"
        try:
            vocab = torch.load(str(vocab_file))
        except FileNotFoundError:
            vocab = None
        if vocab is not None:
            return self.to_language(dataset_language, vocab, tokenizer)
        vocab = self.build_vocab(phrases, tokenizer)
        vocab.set_default_index(self.specials.indexes.unk)
        vocab_file.parent.mkdir(exist_ok=True, parents=True)
        torch.save(vocab, str(vocab_file))
        return self.to_language(dataset_language, vocab, tokenizer)

    def to_language(self, name: str, vocab: Vocab, tokenizer: Tokenizer) -> Language:
        return Language.from_parts(
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


class Bilinguist(NamedTuple):

    source: Language
    target: Language

    def collate(self, batch: Iterable[Tuple[str, str]]):
        """Collate data samples into batch tensors."""
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(self.source.text_transform(src_sample.rstrip("\n")))
            tgt_batch.append(self.target.text_transform(tgt_sample.rstrip("\n")))
        src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=self.source.specials.indexes.pad)
        tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=self.target.specials.indexes.pad)
        return src_batch, tgt_batch

    def languages(self) -> Tuple[Language, Language]:
        return self.source, self.target


def normalize_answer(answer: str, alphabet: str = "abcdefghijklmnopqrstuvwxyz") -> str:
    answer = answer.lower()
    answer = answer.replace(" ", "")
    reconstruct = False
    for x in answer:
        if not x.isalpha():
            reconstruct = True
            break
    if reconstruct:
        answer = "".join(x for x in answer if x in alphabet)
    return answer


def normalize_answer_upper(answer: str) -> str:
    return normalize_answer(answer).upper()
