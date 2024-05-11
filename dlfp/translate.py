#!/usr/bin/env python3

import csv
from collections import deque
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Callable
from typing import Collection
from typing import Iterator
from typing import NamedTuple
from typing import Protocol
from typing import Sequence

import torch
from torch import Tensor
from torchtext.vocab import Vocab

import dlfp.common
from dlfp.utils import SpecialIndexes
from dlfp.utils import Specials
from dlfp.utils import generate_square_subsequent_mask
from dlfp.utils import Bilinguist
from dlfp.models import Cruciformer


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
        if isinstance(prob, Tensor):
            prob = prob.item()
        self.prob = prob

    def current_word_token(self, vocab: Vocab) -> str:
        return vocab.lookup_token(self.current_word)

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

    def normalize_probs(self, next_word_probs: Tensor) -> Tensor:
        return next_word_probs


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


class CruciformerNodeNavigator(NodeNavigator):

    def __init__(self, max_len: int = 6, max_ranks: Sequence[int] = (100, 3, 2)):
        self.max_len = max_len   # includes bos and eos tokens
        self.max_ranks = tuple([-1] + list(max_ranks))
        assert len(self.max_ranks) > 1
        self.softmax = torch.nn.Softmax(dim=0)

    def get_max_rank(self, tgt_sequence_len: int) -> int:
        if tgt_sequence_len >= len(self.max_ranks):
            return self.max_ranks[-1]
        return self.max_ranks[tgt_sequence_len]


    def get_max_len(self, input_len: int) -> int:
        return self.max_len

    def normalize_probs(self, next_word_probs: Tensor) -> Tensor:
        return self.softmax(next_word_probs)


class CruciformerOnemarkNodeNavigator(CruciformerNodeNavigator):

    def __init__(self, max_len: int = 2, max_ranks: Sequence[int] = (100, 1)):
        super().__init__(max_len, max_ranks)

    def include(self, node: Node) -> bool:
        return True


DEFAULT_CHARMARK_MAX_RANKS = (
    12, 10, 10, 10,
     5,  3,  2,  2,
     2,  2,  1)


class CruciformerCharmarkNodeNavigator(CruciformerNodeNavigator):

    def __init__(self, required_len, max_ranks: Sequence[int] = None):
        max_ranks = max_ranks or DEFAULT_CHARMARK_MAX_RANKS
        super().__init__(required_len, max_ranks)
        self.required_len = required_len
        self.eos_index = SpecialIndexes().eos

    def get_max_len(self, input_len: int) -> int:
        return self.required_len

    def include(self, node: Node) -> bool:
        # don't offer nodes that represent incomplete words but are already at the required length
        if node.sequence_length() >= self.required_len and node.current_word != self.eos_index:
            return False
        # don't offer nodes that represent complete words but are not yet at the required length
        if node.sequence_length() < self.required_len and node.current_word == self.eos_index:
            return False
        return True


class Suggestion(NamedTuple):

    phrase: str
    probability: float

    @staticmethod
    def sort_key_by_probability(suggestion: 'Suggestion') -> float:
        return suggestion.probability


def indexes_to_phrase(indexes: Tensor, vocab: Vocab, strip_indexes: Collection[int]) -> str:
    indexes = indexes.detach().flatten().cpu().numpy()
    indexes = [idx for idx in indexes if not idx in strip_indexes]
    tokens = vocab.lookup_tokens(indexes)
    return " ".join(tokens)


class Translator:

    def __init__(self, model: Cruciformer, bilinguist: Bilinguist, device: str):
        self.device = device
        self.model = model
        self.bilinguist = bilinguist
        specials = Specials.create()
        self.strip_indexes = {specials.indexes.bos, specials.indexes.eos}

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
        return indexes_to_phrase(indexes, self.bilinguist.target.vocab, self.strip_indexes)

    def translate(self, src_sentence: str) -> str:
        return self.suggest(src_sentence, count=1)[0].phrase

    def suggest(self,
                src_sentence: str,
                count: int,
                navigator: NodeNavigator = None,
                nodes_callback: Callable[[list[Node]], None] = None) -> list[Suggestion]:
        nodes = []
        for node in self.suggest_nodes(src_sentence, navigator=navigator):
            nodes.append(node)
        if nodes_callback is not None:
            nodes_callback(nodes)
        suggestions = [self.to_suggestion(node) for node in nodes]
        suggestions.sort(key=Suggestion.sort_key_by_probability, reverse=True)
        return suggestions[:count]

    def to_suggestion(self, node: Node) -> Suggestion:
        tgt_indexes = node.y.flatten()
        tgt_phrase = self.indexes_to_phrase(tgt_indexes)
        s = Suggestion(tgt_phrase, node.cumulative_probability())
        return s

    def suggest_nodes(self, src_sentence: str, navigator: NodeNavigator = None) -> Iterator[Node]:
        self.model.eval()
        with torch.no_grad():
            src_encoding = self.encode_source(src_sentence)
            for node in self.greedy_suggest(src_encoding, navigator):
                if node.complete:
                    yield node

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
        next_words_probs, next_words = torch.topk(prob, k=prob.shape[1], dim=1)
        next_words: Tensor = next_words.flatten()
        next_words_probs: Tensor = next_words_probs.flatten()
        next_words_probs = self.navigator.normalize_probs(next_words_probs)
        max_rank = self.navigator.get_max_rank(tgt_sequence_len)
        next_words = next_words[:max_rank]
        next_words_probs = next_words_probs[:max_rank]
        for next_word, next_prob in zip(next_words, next_words_probs):
            child_ys = torch.cat([ys, torch.ones(1, 1).type_as(ys.data).fill_(next_word)], dim=0)
            child = Node(child_ys, next_prob)
            if next_word == self.parent.bilinguist.target.specials.indexes.eos:
                child.complete = True
            node.add_child(child)
            if self.navigator.include(child):
                yield from self.visit(child)


class Attempt(NamedTuple):

    attempt_index: int
    source: str
    target: str
    rank: int
    suggestion_count: int
    top: tuple[str, ...]
    nodes: Optional[list[Node]] = None

    @staticmethod
    def headers(top_k: int) -> list[str]:
        return list(Attempt._fields[:-1]) + [f"top_{i+1}" for i in range(top_k)]

    def to_row(self) -> list[Any]:
        return [self.index, self.source, self.target, self.rank, self.suggestion_count] + list(self.top)


def write_nodes(nodes_folder: Path,
                attempt: Attempt,
                target_vocab: Vocab,
                specials: Specials):
    answer = dlfp.utils.normalize_answer_upper(attempt.target)
    filename = f"{attempt.attempt_index:06d}-{answer}-rank{attempt.rank}-nodes{len(attempt.nodes)}.csv"
    strip_indexes = {specials.indexes.bos, specials.indexes.eos}
    with dlfp.common.open_write(nodes_folder / filename, newline="") as ofile:
        csv_writer = csv.writer(ofile)
        csv_writer.writerow(["seq_len", "guess", "cumu_prob", "word", "prob", "..."])
        for node in attempt.nodes:
            lineage = node.lineage()
            guess = dlfp.translate.indexes_to_phrase(node.y.flatten(), target_vocab, strip_indexes)
            guess = dlfp.utils.normalize_answer_upper(guess)
            row = [len(lineage), guess, node.cumulative_probability()]
            for n in lineage:
                row.append(n.current_word_token(target_vocab))
                row.append(n.prob)
            csv_writer.writerow(row)
