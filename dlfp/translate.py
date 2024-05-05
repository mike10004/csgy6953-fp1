#!/usr/bin/env python3

from typing import Callable
from typing import Iterator
from typing import NamedTuple
from typing import Union

import torch
from torch import Tensor
from dlfp.utils import generate_square_subsequent_mask
from dlfp.tokens import Biglot
from dlfp.models import Seq2SeqTransformer


class PhraseEncoding(NamedTuple):

    indexes: Tensor
    mask: Tensor

    def num_tokens(self) -> int:
        return self.mask.shape[0]


class Translator:

    def __init__(self, model: Seq2SeqTransformer, tokenage: Biglot, device):
        self.device = device
        self.model = model
        self.tokenage = tokenage
        self.target_length_margin: int = 5

    def greedy_decode(self, src_phrase: PhraseEncoding) -> Tensor:
        suggestions = list(self.greedy_suggest(src_phrase, 1))
        return suggestions[0]

    def greedy_suggest(self, src_phrase: PhraseEncoding, get_max_rank: Union[int, Callable[[int], int]]) -> Iterator[Tensor]:
        if isinstance(get_max_rank, int):
            constant = get_max_rank
            get_max_rank = lambda _: constant
        max_len = src_phrase.num_tokens() + self.target_length_margin
        src, src_mask = src_phrase
        model = self.model
        start_symbol: int = self.tokenage.source.language.specials.indexes.bos
        src: Tensor = src.to(self.device)
        src_mask = src_mask.to(self.device)
        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)
        yield from self._next(0, memory, ys, max_len, get_max_rank, src)

    def _next(self, i: int, memory: Tensor, ys: Tensor, max_len, get_max_rank, src):
        if i >= (max_len - 1):
            yield ys.detach()
            return
        memory = memory.to(self.device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device=self.device).type(torch.bool)).to(self.device)
        out = self.model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = self.model.generator(out[:, -1])
        max_rank = get_max_rank(i)
        next_words_probs, next_words = torch.topk(prob, k=max_rank, dim=1)
        next_words = next_words.cpu().numpy().flatten()
        for next_word in next_words:
            if next_word == self.tokenage.target.language.specials.indexes.eos:
                yield ys.detach()
            else:
                ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
                yield from self._next(i + 1, memory, ys, max_len, get_max_rank, src)

    def indexes_to_phrase(self, indexes: Tensor) -> str:
        indexes = indexes.flatten()
        tokens = self.tokenage.target.language.vocab.lookup_tokens(list(indexes.cpu().numpy()))
        return (" ".join(tokens)
                .replace(self.tokenage.target.language.specials.tokens.bos,      "")
                .replace(self.tokenage.target.language.specials.tokens.eos, ""))

    # def translate(self, src_sentence: str) -> str:
    #     return self.suggest(src_sentence, 1)[0]
    #
    # def suggest(self, src_sentence: str, count: int) -> list[str]:
    def translate(self, src_sentence: str) -> str:
        self.model.eval()
        src_encoding = self.encode_source(src_sentence)

        tgt_indexes = self.greedy_decode(src_encoding).flatten()
        return self.indexes_to_phrase(tgt_indexes)

    def encode_source(self, phrase: str) -> PhraseEncoding:
        src = self.tokenage.source.text_transform(phrase).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        return PhraseEncoding(src, src_mask)

