#!/usr/bin/env python3

from collections import deque
from typing import Callable
from typing import Collection
from typing import Iterable
from typing import Iterator
from typing import NamedTuple
from typing import Union

import torch
from torch import Tensor
from torchtext.vocab import Vocab

from dlfp.utils import generate_square_subsequent_mask
from dlfp.utils import Bilinguist
from dlfp.models import Seq2SeqTransformer


class PhraseEncoding(NamedTuple):

    indexes: Tensor
    mask: Tensor

    def num_tokens(self) -> int:
        return self.mask.shape[0]


class Node:

    def __init__(self, y: Tensor, prob: float, complete: bool = False):
        self.y = y.detach()
        flat_y = self.y.flatten()
        self.current_word = flat_y[-1].item()
        self._sequence_length = flat_y.shape[0]
        self.parent = None
        self.children = []
        self.complete = complete
        self.prob = prob

    def sequence_length(self) -> int:
        return self._sequence_length

    def add_child(self, child: 'Node'):
        child.parent = self
        self.children.append(child)

    def lineage(self) -> list['Node']:
        path_to_root = []
        current = self
        while current is not None:
            path_to_root.append(current)
            current = current.parent
        return list(reversed(path_to_root))

    def __repr__(self):
        return f"Node({self.y.flatten().cpu().numpy().tolist()},c={len(self.children)})"

    @staticmethod
    def cumulative_probability(nodes: Iterable['Node']) -> float:
        count = 0
        p = 1.0
        for node in nodes:
            p *= node.prob
            count += 1
        if count == 0:
            return 0
        return p

    def bfs(self) -> Iterator['Node']:
        queue = deque([self])
        while queue:
            current = queue.popleft()
            yield current
            for child in current.children:
                queue.append(child)



class NodeFilter:

    def get_max_len(self, input_len: int) -> int:
        return input_len + 5

    def get_max_rank(self, tgt_sequence_len: int) -> int:
        return 1

    def include(self, node: Node) -> bool:
        return True


class MultiRankNodeFilter(NodeFilter):

    def __init__(self, max_rank: int):
        self.max_rank = max_rank

    def get_max_rank(self, tgt_sequence_len: int) -> int:
        return self.max_rank


class GermanToEnglishNodeFilter(MultiRankNodeFilter):

    def __init__(self, max_rank: int = 1, unrepeatables: Collection[int] = None):
        super().__init__(max_rank=max_rank)
        self.no_skip = False
        self.unrepeatables = frozenset(unrepeatables or ())

    @staticmethod
    def default_unrepeatables(target_vocab: Vocab) -> set[int]:
        index_period = target_vocab(['.'])[0]
        return {index_period}

    def include(self, node: Node) -> bool:
        # node.current_word == self.index_period and child.current_word == self.index_period
        if self.no_skip:
            return True
        if node.parent is None:
            return True
        if node.current_word in self.unrepeatables and node.current_word == node.parent.current_word:
            return False
        return True


class Translator:

    def __init__(self, model: Seq2SeqTransformer, bilinguist: Bilinguist, device: str, node_filter: NodeFilter = None):
        self.device = device
        self.model = model
        self.bilinguist = bilinguist
        self.node_filter = node_filter or NodeFilter()
        self.index_period = bilinguist.target.vocab(['.'])[0]

    def greedy_decode(self, src_phrase: PhraseEncoding) -> Tensor:
        for node in self.greedy_suggest(src_phrase):
            if node.complete:
                return node.y
        raise NotImplementedError("BUG: shouldn't reach here")

    def greedy_suggest(self, src_phrase: PhraseEncoding) -> Iterator[Node]:
        with torch.no_grad():
            max_len = self.node_filter.get_max_len(src_phrase.num_tokens())
            src, src_mask = src_phrase
            model = self.model
            start_symbol: int = self.bilinguist.source.specials.indexes.bos
            src: Tensor = src.to(self.device)
            src_mask = src_mask.to(self.device)
            memory = model.encode(src, src_mask).to(self.device)
            # self.saved_memory = memory.detach()
            ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)
            root = Node(ys, prob=1.0)
            yield root
            yield from self._next(root, memory, ys, max_len)

    def _next(self, node: Node, memory: Tensor, ys: Tensor, max_len) -> Iterator[Node]:
        tgt_sequence_len = node.sequence_length()
        if tgt_sequence_len >= max_len:
            child = Node(ys, prob=1.0, complete=True)
            node.add_child(child)
            yield child
            return
        # memory = memory.to(self.device)
        # assert torch.equal(memory, self.saved_memory)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device=self.device).type(torch.bool)).to(self.device)
        out = self.model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = self.model.generator(out[:, -1])
        max_rank = self.node_filter.get_max_rank(tgt_sequence_len)
        # s = Softmax(dim=1)
        # prob_softmax = s(prob)
        next_words_probs, next_words = torch.topk(prob, k=max_rank, dim=1)
        # next_words_probs_s, next_words_s = torch.topk(prob_softmax, k=max_rank, dim=1)
        # assert torch.equal(next_words, next_words_s)
        next_words_probs = next_words_probs.detach().flatten().cpu().numpy()
        # next_words_probs = next_words_probs / np.sum(next_words_probs)
        next_words = next_words.detach().flatten().cpu().numpy()
        # for next_word, next_prob in zip(next_words, next_words_probs_s.flatten().cpu().numpy()):
        for next_word, next_prob in zip(next_words, next_words_probs):
            if next_word == self.bilinguist.target.specials.indexes.eos:
                child = Node(ys, next_prob, complete=True)
                node.add_child(child)
                yield child
            else:
                child_ys = torch.cat([ys, torch.ones(1, 1).type_as(ys.data).fill_(next_word)], dim=0)
                child = Node(child_ys, next_prob)
                node.add_child(child)
                yield node
                if self.node_filter.include(child):
                    yield from self._next(child, memory, child_ys, max_len)

    def indexes_to_phrase(self, indexes: Tensor) -> str:
        indexes = indexes.flatten()
        tokens = self.bilinguist.target.vocab.lookup_tokens(list(indexes.cpu().numpy()))
        return (" ".join(tokens)
                .replace(self.bilinguist.target.specials.tokens.bos, "")
                .replace(self.bilinguist.target.specials.tokens.eos, ""))

    # def translate(self, src_sentence: str) -> str:
    #     return self.suggest(src_sentence, 1)[0]
    #
    # def suggest(self, src_sentence: str, count: int) -> list[str]:
    def translate(self, src_sentence: str) -> str:
        self.model.eval()
        with torch.no_grad():
            src_encoding = self.encode_source(src_sentence)
            tgt_indexes = self.greedy_decode(src_encoding).flatten()
            return self.indexes_to_phrase(tgt_indexes)

    def encode_source(self, phrase: str) -> PhraseEncoding:
        src = self.bilinguist.source.text_transform(phrase).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        return PhraseEncoding(src, src_mask)

