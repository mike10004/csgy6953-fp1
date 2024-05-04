from typing import Iterable
from typing import NamedTuple
from typing import Iterator
from typing import Sequence
from typing import Callable
from typing import Tuple

import torch
import torch.nn.utils.rnn
from torch import Tensor
from torchtext.vocab import Vocab
from torchtext.vocab import build_vocab_from_iterator

Tokenizer = Callable[[str], Sequence[str]]
TextTransform = Callable[[str], Tensor]
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0,1,2,3


IterablePhrasePair = Iterable[Sequence[str]]


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


class Tokenage:

    def __init__(self,
                 language_pair: Tuple[str, str],
                 token_transform: dict[str, Tokenizer],
                 vocab_transform: dict[str, Vocab],
                 text_transform: dict[str, TextTransform],
                 specials: Specials):
        self.language_pair = language_pair
        self.token_transform = token_transform
        self.vocab_transform = vocab_transform
        self.text_transform = text_transform
        self.specials = specials

    @staticmethod
    def from_token_transform(language_pair: Tuple[str, str],
                             token_transform: dict[str, Tokenizer],
                             data_iter: IterablePhrasePair) -> 'Tokenage':
        assert len(token_transform) == 2, "expect exactly 2 tokenizers"
        assert len(language_pair) == 2, "expect exactly 2 languages"
        assert len(set(language_pair)) == 2, "expect exactly 2 languages"
        assert sorted(token_transform.keys()) == sorted(language_pair), "expect language pair to match tokenizer keys"
        vocab_transform: dict[str, Vocab] = {}
        text_transform: dict[str, TextTransform] = {}
        specials = Specials.create()
        for language_index, language in enumerate(language_pair):
            vocab = build_vocab_from_iterator(Tokenage._yield_tokens(token_transform, data_iter, language_index, language), specials=specials.tokens.as_list())
            vocab_transform[language] = vocab
            vocab_transform[language].set_default_index(UNK_IDX)
            # src and tgt language text transforms to convert raw strings into tensors indices
            text_transform[language] = Tokenage.sequential_transforms(
                token_transform[language], #Tokenization
               vocab_transform[language], #Numericalization
               Tokenage.tensor_transform, # Add BOS/EOS and create tensor
            )
        return Tokenage(language_pair, token_transform, vocab_transform, text_transform, specials)

    @staticmethod
    def _yield_tokens(token_transform: dict[str, Tokenizer],
                      data_iter: IterablePhrasePair,
                      language_index: int,
                      language: str) -> Iterator[str]:
        for data_sample in data_iter:
            yield token_transform[language](data_sample[language_index])


    @staticmethod
    def sequential_transforms(*transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func

    # function to add BOS/EOS and create tensor for input sequence indices
    @staticmethod
    def tensor_transform(token_ids: list[int]):
        return torch.cat((torch.tensor([BOS_IDX]),
                          torch.tensor(token_ids),
                          torch.tensor([EOS_IDX])))


    # function to collate data samples into batch tesors
    def collate_fn(self, batch: Iterable[Tuple[str, str]]):
        src_batch, tgt_batch = [], []
        for sample_pair in batch:
            for language, sample, out_batch in zip(self.language_pair, sample_pair, [src_batch, tgt_batch]):
                out_batch.append(self.text_transform[language](sample.rstrip("\n")))
        src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch
