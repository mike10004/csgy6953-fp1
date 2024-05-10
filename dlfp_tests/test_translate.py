#!/usr/bin/env python3

import csv
import tempfile
from pathlib import Path
from unittest import TestCase
import torch

import dlfp.translate
from dlfp.datasets import DatasetResolver
from dlfp.models import create_model
from dlfp.translate import CruciformerNodeNavigator
from dlfp.translate import GermanToEnglishNodeNavigator
from dlfp.translate import Suggestion
from dlfp.translate import Translator
from dlfp.translate import Attempt
from dlfp.translate import Node
import dlfp_tests.tools
from dlfp.utils import LanguageCache
from dlfp.utils import Restored
from dlfp.utils import Bilinguist
from dlfp.common import get_repo_root

dlfp_tests.tools.suppress_cuda_warning()


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
        try:
            restored = Restored.from_file(get_repo_root() / "checkpoints" / "cruciform-checkpoint-epoch009.pt", device=self.device)
        except FileNotFoundError:
            self.skipTest("checkpoint file not found")
        train_dataset = DatasetResolver().benchmark("train")
        cache = LanguageCache()
        source = cache.get(train_dataset, "clue", "spacy", "en_core_web_sm")
        target = cache.get(train_dataset, "answer", "spacy", "en_core_web_sm")
        bilinguist = Bilinguist(source, target)
        model = create_model(
            src_vocab_size=len(bilinguist.source.vocab),
            tgt_vocab_size=len(bilinguist.target.vocab),
        ).to(self.device)
        model.load_state_dict(restored.model_state_dict)
        model.eval()
        return model, bilinguist

    def test_suggest_cruciform(self):
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            with torch.no_grad():
                model, bilinguist = self._load_restored_cruciform()
                navigator = CruciformerNodeNavigator(max_ranks=(3, 2, 1))
                translator = Translator(model, bilinguist, self.device)
                for src_phrase in [
                    "Pound of verse",
                    "Puts on",
                    "Just know",   # training set
                    "Tell frankly, in slang",  # training set
                    "Like a roofer's drinks?",  # training set
                    "\"I dare you\"",  # training set
                ]:
                    limit = 100
                    with self.subTest(src_phrase):
                        suggestions = translator.suggest(src_phrase, count=limit, navigator=navigator)
                        # self.assertGreater(len(suggestions), 10)
                        print(src_phrase, len(suggestions), suggestions)

    def test_suggest_nodes_cruciform(self):
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)
            nodes_folder = tempdir / "nodes"
            with torch.random.fork_rng():
                torch.random.manual_seed(0)
                with torch.no_grad():
                    model, bilinguist = self._load_restored_cruciform()
                    navigator = CruciformerNodeNavigator(max_ranks=(100, 3, 2, 1))
                    translator = Translator(model, bilinguist, self.device)
                    for src_phrase in [
                        "Pound of verse",
                    ]:
                        with self.subTest(src_phrase):
                            nodes: list[Node] = []
                            for node in translator.suggest_nodes(src_phrase, navigator=navigator):
                                nodes.append(node)
                            attempt = Attempt(0, src_phrase, "SOMETHING", 123, len(nodes), (), nodes)
                            dlfp.translate.write_nodes(nodes_folder, attempt, bilinguist.target.vocab, bilinguist.target.specials)
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
                    model, bilinguist = self._load_restored_cruciform()
                    navigator = CruciformerNodeNavigator(max_ranks=(1, 2, 1, 1))
                    translator = Translator(model, bilinguist, self.device)
                    src_phrase  = "Pound of verse"
                    for complete_node in translator.suggest_nodes(src_phrase, navigator=navigator):
                        print()
                        lineage = complete_node.lineage()
                        for node_index, node in enumerate(lineage):
                            print(node_index, node.current_word, node.current_word_token(bilinguist.target.vocab))


    def test_greedy_suggest(self, verbose: bool = False):
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
                for child in root.children:
                    print(child)
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