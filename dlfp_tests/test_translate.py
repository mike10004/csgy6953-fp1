#!/usr/bin/env python3

from random import Random
from unittest import TestCase
import torch

from dlfp.train import create_model
from dlfp.translate import Node
from dlfp.translate import Translator
import dlfp_tests.tools
from dlfp.utils import Restored
from dlfp.utils import Bilinguist
from dlfp.utils import get_repo_root

dlfp_tests.tools.suppress_cuda_warning()


class TranslatorTest(TestCase):

    def setUp(self):
        self.biglot = dlfp_tests.tools.init_multi30k_de_en_tokenage()

    def test_translate(self):
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            device = dlfp_tests.tools.get_device()
            model = create_model(
                src_vocab_size=len(self.biglot.source.vocab),
                tgt_vocab_size=len(self.biglot.target.vocab),
                DEVICE=device,
            )
            translator = Translator(model, self.biglot, device)
            translated = translator.translate("Ein Mann in grün hält eine Gitarre, während der andere Mann sein Hemd ansieht.").strip()
            self.assertEqual("Russia cloth spoof spoof Madrid sewing Madrid Russia cloth Russia cloth Madrid Madrid sewing cloth cloth sewing Russia sewing sewing cloth cloth", translated, "translation")

    def _load_restored_deen(self, biglot: Bilinguist, device: str):
        try:
            restored = Restored.from_file(get_repo_root() / "checkpoints" / "deen-checkpoint-epoch009.pt", device=device)
        except FileNotFoundError:
            self.skipTest("checkpoint file not found")
        model = create_model(
            src_vocab_size=len(biglot.source.vocab),
            tgt_vocab_size=len(biglot.target.vocab),
            DEVICE=device,
        )
        model.load_state_dict(restored.model_state_dict)
        model.eval()
        return model

    def test_translate_trained(self):
        device = dlfp_tests.tools.get_device()
        model = self._load_restored_deen(self.biglot, device)
        translator = Translator(model, self.biglot, device)
        translated = translator.translate("Ein Mann in grün hält eine Gitarre, während der andere Mann sein Hemd ansieht.").strip()
        expected = "A man in green holds a guitar while the other man looks at his shirt ."
        self.assertEqual(expected, translated, "translation")

    def test_greedy_decode(self):
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            with torch.no_grad():
                device = dlfp_tests.tools.get_device()
                model = self._load_restored_deen(self.biglot, device)
                translator = Translator(model, self.biglot, device)
                src_phrase = translator.encode_source("Ein Mann in grün hält eine Gitarre")
                indexes = translator.greedy_decode(src_phrase)
                actual = translator.indexes_to_phrase(indexes)
                self.assertEqual("A man in green is holding a guitar .", actual.strip())

    def test_greedy_suggest(self, verbose: bool = False):
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            with torch.no_grad():
                device = dlfp_tests.tools.get_device()
                model = self._load_restored_deen(self.biglot, device)
                translator = Translator(model, self.biglot, device)
                src_phrase = translator.encode_source("Ein Mann in grün hält eine Gitarre")
                completes = []
                visited = 0
                for index, node in enumerate(translator.greedy_suggest(src_phrase, 2)):
                    visited += 1
                    # if len(completes) > 100:
                    #     break
                    if node.complete:
                        completes.append(node)
                        actual = translator.indexes_to_phrase(node.y)
                        if verbose:
                            print(actual)
                        if index == 0:
                            self.assertEqual("A man in green is holding a guitar .", actual.strip())
                print(f"{len(completes)} nodes completed; {visited} visited")
                # for node in completes[:100]:
                #     lineage = node.lineage()
                #     print(lineage)
                root = completes[0].lineage()[0]
                print(root)
                for child in root.children:
                    print(child)
                print()
                self.assertIsNone(root.parent)
                sample = completes[1:]
                rng = Random(0x3951551)
                rng.shuffle(sample)
                sample = sample[:100]
                assigned = []
                probability_sum = 0.0
                for complete in completes:
                    lineage = complete.lineage()
                    probability = Node.cumulative_probability(lineage)
                    probability_sum += probability
                    actual = translator.indexes_to_phrase(complete.y)
                    assigned.append((probability, actual))
                assigned.sort(key=lambda a: a[0], reverse=True)
                for a_index, (probability, phrase) in enumerate(assigned):
                    if a_index >= 10:
                        break
                    print(f"{probability/probability_sum:.6f} {phrase}")




