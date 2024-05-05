#!/usr/bin/env python3

from unittest import TestCase
import torch

from dlfp.train import create_model
from dlfp.translate import Translator
import dlfp_tests.tools
from dlfp.utils import Restored
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
                src_vocab_size=len(self.biglot.source.language.vocab),
                tgt_vocab_size=len(self.biglot.target.language.vocab),
                DEVICE=device,
            )
            translator = Translator(model, self.biglot, device)
            translated = translator.translate("Ein Mann in grün hält eine Gitarre, während der andere Mann sein Hemd ansieht.").strip()
            self.assertEqual("Russia cloth spoof spoof Madrid sewing Madrid Russia cloth Russia cloth Madrid Madrid sewing cloth cloth sewing Russia sewing sewing cloth cloth", translated, "translation")

    def _load_restored_deen(self, biglot, device):
        try:
            restored = Restored.from_file(get_repo_root() / "checkpoints" / "deen-checkpoint-epoch009.pt", device=device)
        except FileNotFoundError:
            self.skipTest("checkpoint file not found")
        model = create_model(
            src_vocab_size=len(biglot.source.language.vocab),
            tgt_vocab_size=len(biglot.target.language.vocab),
            DEVICE=device,
        )
        model.load_state_dict(restored.model_state_dict)
        return model

    def test_translate_trained(self):
        device = dlfp_tests.tools.get_device()
        model = self._load_restored_deen(self.biglot, device)
        translator = Translator(model, self.biglot, device)
        translated = translator.translate("Ein Mann in grün hält eine Gitarre, während der andere Mann sein Hemd ansieht.").strip()
        expected = "A man in green holds a guitar while the other man looks at his shirt ."
        self.assertEqual(expected, translated, "translation")

    def test_greedy_decode(self):
        device = dlfp_tests.tools.get_device()
        model = self._load_restored_deen(self.biglot, device)
        translator = Translator(model, self.biglot, device)
        src_phrase = translator.encode_source("Ein Mann in grün hält eine Gitarre")
        indexes = translator.greedy_decode(src_phrase)
        actual = translator.indexes_to_phrase(indexes)
        self.assertEqual("A man in green is holding a guitar .", actual.strip())
