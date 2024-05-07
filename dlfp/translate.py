#!/usr/bin/env python3

from collections import deque
from typing import Collection
from typing import Iterable
from typing import Iterator
from typing import NamedTuple

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

    def cumulative_probability(self) -> float:
        nodes = self.lineage()
        assert len(nodes) > 0
        p = 1.0
        for node in nodes:
            p *= node.prob
        return p

    def bfs(self) -> Iterator['Node']:
        queue = deque([self])
        while queue:
            current = queue.popleft()
            yield current
            for child in current.children:
                queue.append(child)



class NodeNavigator:

    def get_max_len(self, input_len: int) -> int:
        return input_len + 5

    def get_max_rank(self, tgt_sequence_len: int) -> int:
        return 1

    def include(self, node: Node) -> bool:
        return True


class MultiRankNodeNavigator(NodeNavigator):

    def __init__(self, max_rank: int):
        self.max_rank = max_rank

    def get_max_rank(self, tgt_sequence_len: int) -> int:
        return self.max_rank


class GermanToEnglishNodeNavigator(MultiRankNodeNavigator):

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


class Suggestion(NamedTuple):

    phrase: str
    probability: float

    @staticmethod
    def sort_key_by_probability(suggestion: 'Suggestion') -> float:
        return suggestion.probability


class Translator:

    def __init__(self, model: Seq2SeqTransformer, bilinguist: Bilinguist, device: str):
        self.device = device
        self.model = model
        self.bilinguist = bilinguist

    def greedy_decode(self, src_phrase: PhraseEncoding) -> Tensor:
        for node in self.greedy_suggest(src_phrase):
            if node.complete:
                return node.y
        raise NotImplementedError("BUG: shouldn't reach here")

    def greedy_suggest(self, src_phrase: PhraseEncoding, navigator: NodeNavigator = None) -> Iterator[Node]:
        with torch.no_grad():
            navigator = navigator or NodeNavigator()
            max_len = navigator.get_max_len(src_phrase.num_tokens())
            src, src_mask = src_phrase
            model = self.model
            start_symbol: int = self.bilinguist.source.specials.indexes.bos
            src: Tensor = src.to(self.device)
            src_mask = src_mask.to(self.device)
            memory = model.encode(src, src_mask).to(self.device)
            ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)
            root = Node(ys, prob=1.0)
            visitor = NodeVisitor(self, max_len, memory, navigator)
            yield from visitor.visit(root)


    def indexes_to_phrase(self, indexes: Tensor) -> str:
        indexes = indexes.flatten()
        tokens = self.bilinguist.target.vocab.lookup_tokens(list(indexes.cpu().numpy()))
        return (" ".join(tokens)
                .replace(self.bilinguist.target.specials.tokens.bos, "")
                .replace(self.bilinguist.target.specials.tokens.eos, ""))

    def translate(self, src_sentence: str) -> str:
        return self.suggest(src_sentence, count=1)[0].phrase

    def suggest(self, src_sentence: str, count: int, navigator: NodeNavigator = None) -> list[Suggestion]:
        suggestions = []
        self.model.eval()
        with torch.no_grad():
            src_encoding = self.encode_source(src_sentence)
            for node in self.greedy_suggest(src_encoding, navigator):
                if node.complete:
                    tgt_indexes = node.y.flatten()
                    tgt_phrase = self.indexes_to_phrase(tgt_indexes)
                    suggestions.append(Suggestion(tgt_phrase, node.cumulative_probability()))
        suggestions.sort(key=Suggestion.sort_key_by_probability, reverse=True)
        return suggestions[:count]

    def encode_source(self, phrase: str) -> PhraseEncoding:
        src = self.bilinguist.source.text_transform(phrase).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        return PhraseEncoding(src, src_mask)


class NodeVisitor:

    def __init__(self, parent: Translator, max_len: int, memory: Tensor, navigator: NodeNavigator):
        self.parent = parent
        self.max_len = max_len
        self.memory = memory
        self.navigator = navigator

    def visit(self, node: Node) -> Iterator[Node]:
        ys = node.y
        tgt_sequence_len = node.sequence_length()
        if tgt_sequence_len >= self.max_len:
            node.complete = True
        yield node
        if node.complete:
            return
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device=self.parent.device).type(torch.bool)).to(self.parent.device)
        out = self.parent.model.decode(ys, self.memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = self.parent.model.generator(out[:, -1])
        max_rank = self.navigator.get_max_rank(tgt_sequence_len)
        next_words_probs, next_words = torch.topk(prob, k=max_rank, dim=1)
        next_words_probs = next_words_probs.detach().flatten().cpu().numpy()
        next_words = next_words.detach().flatten().cpu().numpy()
        for next_word, next_prob in zip(next_words, next_words_probs):
            if next_word == self.parent.bilinguist.target.specials.indexes.eos:
                child = Node(ys, next_prob, complete=True)
                node.add_child(child)
                yield from self.visit(child)
            else:
                child_ys = torch.cat([ys, torch.ones(1, 1).type_as(ys.data).fill_(next_word)], dim=0)
                child = Node(child_ys, next_prob)
                node.add_child(child)
                if self.navigator.include(child):
                    yield from self.visit(child)
