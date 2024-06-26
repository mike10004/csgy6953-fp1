#!/usr/bin/env python3

"""Module for classes and methods relating to sequence-to-sequence translation."""

import csv
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Callable
from typing import Collection
from typing import Iterator
from typing import NamedTuple
from typing import Sequence

import numpy as np
import torch
from torch import Tensor
from torchtext.vocab import Vocab

import dlfp.common
from dlfp.utils import SpecialIndexes
from dlfp.utils import Specials
from dlfp.utils import generate_square_subsequent_mask
from dlfp.utils import Bilinguist
from dlfp.models import Cruciformer


ProbEstimator = Callable[[Iterator[float]], float]


class PhraseEncoding(NamedTuple):

    """Value class that represents a source phrase encoding."""

    indexes: Tensor
    mask: Tensor

    def num_tokens(self) -> int:
        return self.mask.shape[0]


class Node:

    """Value class that represents a node in the search tree for sequence generation."""

    def __init__(self, y: Tensor, prob: float, complete: bool = False):
        self.y = y.detach()
        flat_y = self.y.flatten()
        self.current_word = flat_y[-1].item()
        self._sequence_length = flat_y.shape[0]
        self.parent = None
        # self.children = []
        self.complete = complete
        if isinstance(prob, Tensor):
            prob = prob.item()
        self.prob = prob

    def current_word_token(self, vocab: Vocab) -> str:
        return vocab.lookup_token(self.current_word)

    def sequence_length(self) -> int:
        return self._sequence_length

    def lineage(self) -> list['Node']:
        path_to_root = []
        current = self
        while current is not None:
            path_to_root.append(current)
            current = current.parent
        return list(reversed(path_to_root))

    def __repr__(self):
        complete_infix = "c" if self.complete else "i"
        return f"Node({self.y.flatten().cpu().numpy().tolist()},p={self.prob:.4e},{complete_infix})"

    def cumulative_probability(self, prob_estimator: ProbEstimator = None) -> float:
        nodes = self.lineage()
        assert len(nodes) > 0  # always includes self
        prob_estimator = prob_estimator or _product
        p = prob_estimator(map(lambda node: node.prob, iter(nodes)))
        return p


def _product(probabilities: Iterator[float]) -> float:
    p = 1.0
    for component in probabilities:
        p *= component
    return p


def _mean_estimator(p: Iterator[float]) -> float:
    return np.mean(list(p))


class NodeNavigator:

    """Service class that provides navigation hints for the search tree traversal.

    This class, in combination with `NodeVisitor`, implements a variant of beam search.
    """

    def get_max_len(self, input_len: int) -> int:
        return input_len + 5

    def get_max_rank(self, tgt_sequence_len: int) -> int:
        return 1

    def notify(self, node: Node):
        pass

    def consider(self, node: Node, next_word: int, next_prob: float) -> bool:
        return True

    def include(self, node: Node) -> bool:
        return True

    def normalize_probs(self, next_word_probs: Tensor) -> Tensor:
        return next_word_probs

    def calculate_cumulative_probability(self, component_probs: Iterator[float]) -> float:
        return _product(component_probs)

    def is_probability_ascending(self) -> bool:
        return True

    def pretruncate_disabled(self) -> bool:
        return False


class MultiRankNodeNavigator(NodeNavigator):

    def __init__(self, max_rank: int):
        self.max_rank = max_rank

    def get_max_rank(self, tgt_sequence_len: int) -> int:
        return self.max_rank


class CruciformerNodeNavigator(NodeNavigator):

    def __init__(self,
                 max_len: int = 6,
                 max_ranks: Sequence[int] = (100, 3, 2),
                 probnorm: Optional[str] = None,
                 estimator: Optional[str] = None):
        self.max_len = max_len   # includes bos and eos tokens
        self.max_ranks = tuple([-1] + list(max_ranks))
        assert len(self.max_ranks) > 1
        self.probnorm = parse_probnorm(probnorm)
        self.estimator = parse_estimator(estimator)
        self._probability_ascending = probnorm != "logsoftmax"

    def get_max_rank(self, tgt_sequence_len: int) -> int:
        if tgt_sequence_len >= len(self.max_ranks):
            return self.max_ranks[-1]
        return self.max_ranks[tgt_sequence_len]


    def get_max_len(self, input_len: int) -> int:
        return self.max_len

    def normalize_probs(self, next_word_probs: Tensor) -> Tensor:
        return self.probnorm(next_word_probs)

    def calculate_cumulative_probability(self, component_probs: Iterator[float]) -> float:
        return super().calculate_cumulative_probability(component_probs)

    def is_probability_ascending(self) -> bool:
        return self._probability_ascending

    @classmethod
    def factory(cls, tgt_phrase: str, kwargs: dict[str, Any]) -> 'CruciformerNodeNavigator':
        return cls(**kwargs)


class CruciformerOnemarkNodeNavigator(CruciformerNodeNavigator):

    def __init__(self, max_len: int = 2, max_ranks: Sequence[int] = (100, 1)):
        super().__init__(max_len, max_ranks)

    def include(self, node: Node) -> bool:
        return True

DEFAULT_CHARMARK_MAX_RANKS = (2,)


def probnorm_translate(probs: Tensor) -> Tensor:
    probs = probs - torch.min(probs)
    return probs

def probnorm_translate_and_scale(probs: Tensor) -> Tensor:
    probs = probnorm_translate(probs)
    probs = probs / torch.sum(probs)
    return probs


LOG_SOFTMAX = torch.nn.LogSoftmax(dim=0)
TANH = torch.nn.Tanh()

def probnorm_tanh(probs: Tensor) -> Tensor:
    probs = TANH(probs)  # maps values to [-1, 1]
    return (probs + 1.0) / 2.0


def probnorm_logsoftmax(p: Tensor) -> Tensor:
    p = LOG_SOFTMAX(p)
    return p


def probnorm_unitnorm(p: Tensor, epsilon: float = 1e-8) -> Tensor:
    p = probnorm_translate(p)
    # https://stackoverflow.com/a/50415896/2657036
    return p / (epsilon + torch.sqrt(torch.sum(torch.square(p))))


def parse_probnorm(probnorm: Optional[str]) -> Callable[[Tensor], Tensor]:
    if not probnorm or probnorm == "softmax":
        return torch.nn.Softmax(dim=0)
    return {
        "translate": probnorm_translate,
        "scale": probnorm_translate_and_scale,
        "tanh": probnorm_tanh,
        "logsoftmax": probnorm_logsoftmax,
        "unit": probnorm_unitnorm,
    }[probnorm]


def _sum_estimator(p: Iterator[float]) -> float:
    return np.sum(tuple(p))


def parse_estimator(estimator: Optional[str]) -> ProbEstimator:
    if estimator == "mean":
        return _mean_estimator
    if estimator == "sum":
        return _sum_estimator
    return _product


class CruciformerCharmarkNodeNavigator(CruciformerNodeNavigator):

    """Service class that provides navigation hints for the letter model search tree."""

    def __init__(self, required_len: int, *, max_ranks: Sequence[int] = None, probnorm: Optional[str] = None):
        max_ranks = max_ranks or DEFAULT_CHARMARK_MAX_RANKS
        super().__init__(required_len, max_ranks=max_ranks, probnorm=probnorm)
        self.required_len = required_len
        indexes = SpecialIndexes()
        self.eos_index = indexes.eos
        self.unconsidered = {indexes.bos, indexes.pad, indexes.unk}

    @classmethod
    def factory(cls, tgt_phrase: str, kwargs: dict[str, Any]) -> 'CruciformerCharmarkNodeNavigator':
        # +2 for bos and eos tokens
        return CruciformerCharmarkNodeNavigator(len(tgt_phrase) + 2, **kwargs)

    def get_max_len(self, input_len: int) -> int:
        return self.required_len

    def consider(self, node: Node, next_word: int, next_prob: float) -> bool:
        if next_word == self.eos_index:
            if (node.sequence_length() + 1) < self.required_len:
                return False
        if next_word in self.unconsidered:
            return False
        return super().consider(node, next_word, next_prob)

    def include(self, node: Node) -> bool:
        # don't offer nodes that represent incomplete words but are already at the required length
        if node.sequence_length() >= self.required_len and node.current_word != self.eos_index:
            return False
        # don't offer nodes that represent complete words but are not yet at the required length
        # (the consider method should prevent this from happening)
        if node.sequence_length() < self.required_len and node.current_word == self.eos_index:
            return False
        return True

    def pretruncate_disabled(self) -> bool:
        return True


class Suggestion(NamedTuple):

    """Value class that represents an answer suggestion."""

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

    """Service class that performs sequence-to-sequence translation by generating output sequences."""

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
        prob_estimator = None if navigator is None else navigator.calculate_cumulative_probability
        suggestions = [self.to_suggestion(node, prob_estimator) for node in nodes]
        reverse = True if navigator is None else navigator.is_probability_ascending()
        suggestions.sort(key=Suggestion.sort_key_by_probability, reverse=reverse)
        return suggestions[:count]

    def to_suggestion(self, node: Node, prob_estimator: ProbEstimator = None) -> Suggestion:
        tgt_indexes = node.y.flatten()
        tgt_phrase = self.indexes_to_phrase(tgt_indexes)
        s = Suggestion(tgt_phrase, node.cumulative_probability(prob_estimator))
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

    """Service class that performs sequence generation search tree traversal.

    This class, in combination with `NodeNavigator`, implements a variant of beam search.
    """

    def __init__(self, parent: Translator, max_len: int, memory: Tensor, navigator: NodeNavigator):
        self.parent = parent
        self.max_len = max_len
        self.memory = memory
        self.navigator = navigator

    def _is_eos_index(self, index: int) -> bool:
        return index == self.parent.bilinguist.target.specials.indexes.eos

    def _generate_next(self, node: Node) -> tuple[Tensor, Tensor]:
        ys = node.y
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device=self.parent.device).type(torch.bool)).to(self.parent.device)
        out = self.parent.model.decode(ys, self.memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = self.parent.model.generator(out[:, -1])
        next_words_probs, next_words = torch.topk(prob, k=prob.shape[1], dim=1)
        next_words: Tensor = next_words.flatten()
        next_words_probs: Tensor = next_words_probs.flatten()
        return next_words, next_words_probs

    def visit(self, node: Node) -> Iterator[Node]:
        ys = node.y
        tgt_sequence_len = node.sequence_length()
        if tgt_sequence_len >= self.max_len:
            node.complete = True
        yield node
        self.navigator.notify(node)
        if node.complete:
            return
        next_words, next_words_probs = self._generate_next(node)
        next_words_probs = self.navigator.normalize_probs(next_words_probs)
        max_rank = self.navigator.get_max_rank(tgt_sequence_len)
        if not self.navigator.pretruncate_disabled():
            next_words = next_words[:max_rank]
            next_words_probs = next_words_probs[:max_rank]
        num_considered = 0
        next_iterator = zip(next_words, next_words_probs)
        if not self.navigator.is_probability_ascending():
            next_iterator = reversed(list(next_iterator))
        for next_word, next_prob in next_iterator:
            if num_considered >= max_rank:
                break
            if not self.navigator.consider(node, next_word, next_prob):
                continue
            child_ys = torch.cat([ys, torch.ones(1, 1).type_as(ys.data).fill_(next_word)], dim=0)
            child = Node(child_ys, next_prob)
            child.parent = node
            if self._is_eos_index(next_word):
                child.complete = True
            if self.navigator.include(child):
                yield from self.visit(child)
            num_considered += 1


class Attempt(NamedTuple):

    """Value class that represents an attempt at a sequence translation."""

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
        return [self.attempt_index, self.source, self.target, self.rank, self.suggestion_count] + list(self.top)


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
