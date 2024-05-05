from typing import Iterable
from typing import Iterator
from typing import NamedTuple
from typing import Sequence
from typing import Callable
from typing import Tuple

import torch
import torch.nn.utils.rnn
from torch import Tensor
from dlfp.utils import Language

TextTransform = Callable[[str], Tensor]


IterablePhrasePair = Iterable[Tuple[str, str]]



class Linguist(NamedTuple):

    language: Language
    text_transform: TextTransform

    @staticmethod
    def from_language(language: Language):
        text_transform = Linguist.compose([
            language.tokenizer,
            language.vocab,
            language.to_tensor,
        ])
        return Linguist(language, text_transform)

    @staticmethod
    def compose(transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func


class Biglot(NamedTuple):

    source: Linguist
    target: Linguist

    def collate(self, batch: Iterable[Tuple[str, str]]):
        """Collate data samples into batch tensors."""
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(self.source.text_transform(src_sample.rstrip("\n")))
            tgt_batch.append(self.target.text_transform(tgt_sample.rstrip("\n")))
        src_batch = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=self.source.language.specials.indexes.pad)
        tgt_batch = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=self.target.language.specials.indexes.pad)
        return src_batch, tgt_batch

    def languages(self) -> Tuple[Language, Language]:
        return self.source.language, self.target.language