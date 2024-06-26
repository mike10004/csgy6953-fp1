#!/usr/bin/env python3

import csv
import time
import tempfile
import unittest
from pathlib import Path
from random import Random
from typing import Collection
from typing import Optional
from typing import Sequence
from unittest import TestCase

import torch
from torch import Tensor
from torchtext.vocab import Vocab

import dlfp.translate
import dlfp.models
from dlfp.models import create_model
from dlfp.translate import CruciformerNodeNavigator
from dlfp.translate import MultiRankNodeNavigator
from dlfp.translate import NodeNavigator
from dlfp.translate import Suggestion
from dlfp.translate import Translator
from dlfp.translate import Attempt
from dlfp.translate import Node
from dlfp.translate import NodeVisitor
from dlfp.translate import CruciformerCharmarkNodeNavigator
import dlfp_tests.tools
from dlfp.utils import PhrasePair
from dlfp.utils import Restored
from dlfp.utils import Bilinguist
from dlfp.common import get_repo_root
from dlfp.utils import SpecialIndexes

dlfp_tests.tools.suppress_cuda_warning()


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


class TranslatorTest(TestCase):

    def setUp(self):
        self.deen_bilinguist = dlfp_tests.tools.get_multi30k_de_en_bilinguist()
        self.device = dlfp_tests.tools.get_device()

    def test_translate(self):
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            device = dlfp_tests.tools.get_device()
            model = create_model(
                src_vocab_size=len(self.deen_bilinguist.source.vocab),
                tgt_vocab_size=len(self.deen_bilinguist.target.vocab),
            ).to(device)
            translator = Translator(model, self.deen_bilinguist, device)
            translated = translator.translate("Ein Mann in grün hält eine Gitarre, während der andere Mann sein Hemd ansieht.").strip()
            self.assertEqual("Russia cloth spoof spoof Madrid sewing Madrid Russia cloth Russia cloth Madrid Madrid sewing cloth cloth sewing Russia sewing sewing cloth cloth", translated, "translation")

    def _load_restored_deen(self, bilinguist: Bilinguist, device: str):
        try:
            restored = Restored.from_file(get_repo_root() / "checkpoints" / "deen-checkpoint-epoch009.pt", device=device)
        except FileNotFoundError:
            self.skipTest("checkpoint file not found")
        model = create_model(
            src_vocab_size=len(bilinguist.source.vocab),
            tgt_vocab_size=len(bilinguist.target.vocab),
        ).to(device)
        model.load_state_dict(restored.model_state_dict)
        model.eval()
        return model

    def test_translate_trained(self):
        device = dlfp_tests.tools.get_device()
        model = self._load_restored_deen(self.deen_bilinguist, device)
        translator = Translator(model, self.deen_bilinguist, device)
        translated = translator.translate("Ein Mann in grün hält eine Gitarre, während der andere Mann sein Hemd ansieht.").strip()
        expected = "A man in green holds a guitar while the other man looks at his shirt ."
        self.assertEqual(expected, translated, "translation")

    def test_greedy_decode(self):
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            with torch.no_grad():
                device = dlfp_tests.tools.get_device()
                model = self._load_restored_deen(self.deen_bilinguist, device)
                translator = Translator(model, self.deen_bilinguist, device)
                src_phrase = translator.encode_source("Ein Mann in grün hält eine Gitarre")
                indexes = translator.greedy_decode(src_phrase)
                actual = translator.indexes_to_phrase(indexes)
                self.assertEqual("A man in green is holding a guitar .", actual.strip())

    def test_suggest_10(self):
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            with torch.no_grad():
                device = dlfp_tests.tools.get_device()
                model = self._load_restored_deen(self.deen_bilinguist, device)
                navigator = GermanToEnglishNodeNavigator(
                    max_rank=2,
                    unrepeatables=GermanToEnglishNodeNavigator.default_unrepeatables(self.deen_bilinguist.target.vocab),
                )
                translator = Translator(model, self.deen_bilinguist, device)
                src_phrase = "Ein Mann in grün hält eine Gitarre"
                limit = 10
                suggestions = translator.suggest(src_phrase, count=limit, navigator=navigator)
                self._check_output(suggestions, limit=limit)

    def _load_restored_cruciform(self):
        return dlfp_tests.tools.load_restored_cruciform(get_repo_root() / "checkpoints" / "05092329-checkpoint-epoch009.pt", device=self.device)

    def _load_restored_cruciform_charmark(self):
        return dlfp_tests.tools.load_restored_cruciform(get_repo_root() / "checkpoints" / "charmark-test-checkpoint.pt", device=self.device, dataset_name="charmark")

    def test_suggest_cruciform(self):
        src_phrases = [
            "Pound of verse",
            "Puts on",
            "Just know",   # training set
            "Tell frankly, in slang",  # training set
            "Like a roofer's drinks?",  # training set
            "\"I dare you\"",  # training set
        ]
        self._test_suggest_cruciform(src_phrases, (3, 2, 1))

    def _test_suggest_cruciform(self, src_phrases: list[str], max_ranks: Sequence[int]):
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            with torch.no_grad():
                rc = self._load_restored_cruciform()
                navigator = CruciformerNodeNavigator(max_ranks=max_ranks)
                translator = Translator(rc.model, rc.bilinguist, self.device)
                for src_phrase in src_phrases:
                    limit = 100
                    with self.subTest(src_phrase):
                        start = time.time()
                        suggestions = translator.suggest(src_phrase, count=limit, navigator=navigator)
                        finish = time.time()
                        # self.assertGreater(len(suggestions), 10)
                        print(src_phrase, len(suggestions), f"{finish-start:.1f}s", suggestions[:3])

    def test_suggest_nodes_cruciform(self):
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)
            nodes_folder = tempdir / "nodes"
            with torch.random.fork_rng():
                torch.random.manual_seed(0)
                with torch.no_grad():
                    rc = self._load_restored_cruciform()
                    navigator = CruciformerNodeNavigator(max_ranks=(100, 3, 2, 1))
                    translator = Translator(rc.model, rc.bilinguist, self.device)
                    for src_phrase in [
                        "Pound of verse",
                    ]:
                        with self.subTest(src_phrase):
                            nodes: list[Node] = []
                            for node in translator.suggest_nodes(src_phrase, navigator=navigator):
                                nodes.append(node)
                            attempt = Attempt(0, src_phrase, "SOMETHING", 123, len(nodes), (), nodes)
                            dlfp.translate.write_nodes(nodes_folder, attempt, rc.bilinguist.target.vocab, rc.bilinguist.target.specials)
                            node_count = len(nodes)
                            print(node_count, "nodes")
            nodes_files = list(nodes_folder.iterdir())
            for file in nodes_files:
                print(file)
            self.assertEqual(1, len(nodes_files))
            with open(nodes_files[0], "r") as ifile:
                csv_reader = csv.reader(ifile)
                _ = next(csv_reader)
                node_rows = list(csv_reader)
            self.assertEqual(node_count, len(node_rows))

    def test_suggest_nodes_cruciform_one(self):
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            with torch.no_grad():
                rc = self._load_restored_cruciform()
                navigator = CruciformerNodeNavigator(max_ranks=(1, 2, 1, 1))
                translator = Translator(rc.model, rc.bilinguist, self.device)
                src_phrase  = "Pound of verse"
                for complete_node in translator.suggest_nodes(src_phrase, navigator=navigator):
                    print()
                    lineage = complete_node.lineage()
                    for node_index, node in enumerate(lineage):
                        print(node_index, node.current_word, node.current_word_token(rc.bilinguist.target.vocab))

    def test_greedy_suggest(self, verbose: bool = False):
        # from dlfp.translate2 import Translator as Translator2
        # from dlfp.translate2 import NodeVisitor as NodeVisitor2
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            with torch.no_grad():
                device = dlfp_tests.tools.get_device()
                model = self._load_restored_deen(self.deen_bilinguist, device)
                navigator = GermanToEnglishNodeNavigator(
                    max_rank=2,
                    unrepeatables=GermanToEnglishNodeNavigator.default_unrepeatables(self.deen_bilinguist.target.vocab),
                )
                translator = Translator(model, self.deen_bilinguist, device)
                src_phrase = translator.encode_source("Ein Mann in grün hält eine Gitarre")
                completes = []
                visited = 0
                for index, node in enumerate(translator.greedy_suggest(src_phrase, navigator=navigator)):
                    visited += 1
                    if node.complete:
                        completes.append(node)
                        actual = translator.indexes_to_phrase(node.y)
                        if verbose:
                            print(actual)
                        if index == 0:
                            self.assertEqual("A man in green is holding a guitar .", actual.strip())
                print(f"{len(completes)} nodes completed; {visited} visited")
                root = completes[0].lineage()[0]
                print(root)
                # for child in root.children:
                #     print(child)
                print()
                self.assertIsNone(root.parent)
                assigned: list[Suggestion] = []
                probability_sum = 0.0
                for complete in completes:
                    probability = complete.cumulative_probability()
                    probability_sum += probability
                    actual = translator.indexes_to_phrase(complete.y)
                    assigned.append(Suggestion(actual, probability))
                assigned.sort(key=Suggestion.sort_key_by_probability, reverse=True)
                for a_index, s in enumerate(assigned):
                    if a_index >= 10:
                        break
                    print(f"{s.probability/probability_sum:.6f} {s.phrase}")
                self._check_output(assigned)

    def _check_output(self, assigned: list[Suggestion], limit: int = None):
        index = 0
        with self.subTest("content"):
            with open(dlfp_tests.tools.get_testdata_dir() / "translate-expected-1.csv") as ifile:
                expecteds = [Suggestion(t, float(p)) for p, t in csv.reader(ifile)]
            if limit is not None:
                expecteds = expecteds[:limit]
            for (e_phrase, e_probability), (a_phrase, a_probabilty) in zip(expecteds, assigned):
                self.assertEqual(e_phrase, a_phrase, f"phrase {index}")
                self.assertAlmostEqual(e_probability, a_probabilty, delta=0.05)
                index += 1
        if limit is None:
            with self.subTest("length"):
                self.assertEqual(len(expecteds), len(assigned), "lengths of lists")
                self.assertGreater(index, 100)

    def test_charmark_suggest(self):
        src_phrases  = [
            ("Exactitude", "RIGOR"),
            ("Swag", "LOOT"),
        ]
        self._test_charmark_suggest(src_phrases, (3, 2, 1))

    @unittest.skip("for demo only")
    def test_charmark_suggest_big(self):
        src_phrases  = [
            # ("Exactitude", "RIGOR"),
            ("Swag", "LOOT"),
        ]
        self._test_charmark_suggest(src_phrases, (12, 6, 3, 2, 1))

    def _test_charmark_suggest(self, src_phrases: list[PhrasePair], max_ranks: Sequence[int]):
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            with torch.no_grad():
                rc = self._load_restored_cruciform_charmark()
                translator = Translator(rc.model, rc.bilinguist, self.device)
                for src_phrase, tgt_phrase in src_phrases:
                    with self.subTest(src_phrase):
                        navigator = VerboseCharmarkNavigator(required_len=len(tgt_phrase)+2, max_ranks=max_ranks, quiet=True)
                        nodes = list(translator.suggest_nodes(src_phrase, navigator=navigator))
                        print(len(nodes), "suggestions")
                        self.assertGreater(len(nodes), 0)

    def test_zip_tensors(self):
        w = torch.tensor([[1, 2, 3]])
        p = torch.tensor([[7.5, 1.3, 9.2]])
        zipped = list(zip(w.flatten(), p.flatten()))
        self.assertListEqual([
            (1, 7.5),
            (2, 1.3),
            (3, 9.2),
        ], zipped)

    def test_softmax(self):
        softmax = torch.nn.Softmax(dim=-1)
        s = softmax(torch.tensor([[1, 4.5, 4.6]]))
        print(s)
        self.assertAlmostEqual(s[0][1], s[0][2], delta=0.5)


class VerboseCharmarkNavigator(CruciformerCharmarkNodeNavigator):

    def __init__(self, required_len: int, *, max_ranks: Sequence[int] = None, probnorm: Optional[str] = None, quiet: bool = False):
        super().__init__(required_len, max_ranks=max_ranks, probnorm=probnorm)
        self.quiet = quiet

    def notify(self, node: Node):
        self._print("notify", node.sequence_length(), node)

    # noinspection PyMethodMayBeStatic
    def _print(self, action: str, current_len: int, *args):
        if not self.quiet:
            print(f"{action:10} {current_len}", *args)

    def consider(self, node: Node, next_word: int, next_prob: float) -> bool:
        result = super().consider(node, next_word, next_prob)
        self._print("consider", node.sequence_length(), next_word, result)
        return result

    def include(self, node: Node) -> bool:
        result = super().include(node)
        self._print("include", node.sequence_length() - 1, node.current_word, result)
        return result


class PseudoGeneratorNodeVisitor(NodeVisitor):

    def __init__(self, max_len: int, navigator: NodeNavigator):
        # noinspection PyTypeChecker
        parent: Translator = None
        # noinspection PyTypeChecker
        memory: Tensor = None
        super().__init__(parent, max_len, memory, navigator)
        self.rng = Random(18462)
        self.special_indexes = SpecialIndexes()
        self.special_chars = {
            self.special_indexes.bos: "<",
            self.special_indexes.eos: ">",
            self.special_indexes.pad: "_",
            self.special_indexes.unk: "?",
        }
        self.tgt_vocab = [self.special_indexes.eos] + list(range(4, 4 + 26))
        assert len(self.tgt_vocab) == 27, f"expect vocab length 27, got {len(self.tgt_vocab)} in {self.tgt_vocab}"
        assert len(set(self.tgt_vocab)) == len(self.tgt_vocab), f"expect uniques in {self.tgt_vocab}"

    def _is_eos_index(self, index: int) -> bool:
        return index == self.special_indexes.eos

    def to_char(self, index: int) -> str:
        ch = self.special_chars.get(index, None)
        if ch is not None:
            return ch
        index -= 4  #
        return chr(index + 65)

    def to_word(self, node: Node):
        indexes = node.y.flatten().numpy().tolist()
        return "".join(map(self.to_char, indexes))

    def _generate_next(self, node: Node) -> tuple[Tensor, Tensor]:
        candidates = list(self.tgt_vocab)
        self.rng.shuffle(candidates)
        probs = torch.linspace(1.0, 1e-6, steps=len(candidates))
        return torch.tensor(candidates, dtype=torch.int64), probs


class NodeVisitorTest(TestCase):

    def test_visit(self):
        required_len = 5
        navigator = CruciformerCharmarkNodeNavigator(required_len)
        visitor = PseudoGeneratorNodeVisitor(required_len, navigator)
        indexes = SpecialIndexes()
        root = Node(torch.tensor([[indexes.bos]]), 1.0, False)
        complete = []
        visited_count = 0
        for node in visitor.visit(root):
            if visited_count > 1_000_000:
                break
            if node.complete:
                self.assertEqual(required_len, node.sequence_length())
                complete.append(node)
                if len(complete) >= 100:
                    break
            visited_count += 1
        print(visited_count, 'visited')
        print(len(complete), 'complete')

        for node in complete:
            answer = visitor.to_word(node)
            print(answer)
            self.assertEqual(required_len, len(answer))

    def test_suggestion_count(self):
        for required_len in range(5, 19):
            with self.subTest(required_len=required_len):
                count = 1
                for i in range(required_len):
                    try:
                        factor = dlfp.translate.DEFAULT_CHARMARK_MAX_RANKS[i]
                    except IndexError:
                        factor = 1
                    count *= factor
                print(f"{required_len:2d} -> {count:7d}")
                self.assertLess(count, 5_000_000)

    def test_norm_methods(self, verbose: bool = True):
        next_words_probs = torch.tensor([ 2.5581,  2.3117,  1.9621,  1.5940,  1.5705,  1.4213,  1.3919,  1.3370,
         1.1915,  0.7935,  0.6412,  0.6107,  0.3704,  0.3376, -0.0883, -0.2303,
        -0.4242, -0.6058, -0.6532, -0.6712, -1.2395, -1.5089, -1.9172, -2.0092,
        -2.2799, -2.3521, -2.6116, -2.8697, -2.9313, -4.9003])
        norm_functions = [
            "softmax",
            "translate",
            "scale",
            "tanh",
            "logsoftmax",
            "unit",
        ]
        for fn_name, fn in list(map(lambda x: (x, dlfp.translate.parse_probnorm(x)), norm_functions)):
            with self.subTest(fn_name):
                outcome = fn(next_words_probs)
                if verbose:
                    print(outcome)
                self.assertIsInstance(outcome, Tensor)
