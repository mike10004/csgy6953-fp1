#!/usr/bin/env python3

from unittest import TestCase
import torch

from dlfp.train import create_model
from dlfp.translate import Translator
import dlfp_tests.tools

dlfp_tests.tools.suppress_cuda_warning()


class TranslatorTest(TestCase):

    def test_translate(self):
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            tokenage = dlfp_tests.tools.init_multi30k_de_en_tokenage()
            device = dlfp_tests.tools.get_device()
            src_lang, tgt_lang = tokenage.language_pair
            model = create_model(
                src_vocab_size=len(tokenage.vocab_transform[src_lang]),
                tgt_vocab_size=len(tokenage.vocab_transform[tgt_lang]),
                DEVICE=device,
            )
            translator = Translator(model, tokenage, device)
            translated = translator.translate("Ein Mann in grün hält eine Gitarre, während der andere Mann sein Hemd ansieht.").strip()
            self.assertEqual("Russia cloth spoof spoof Madrid sewing Madrid Russia cloth Russia cloth Madrid Madrid sewing cloth cloth sewing Russia sewing sewing cloth cloth", translated, "translation")
