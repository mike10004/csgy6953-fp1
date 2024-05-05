#!/usr/bin/env python3

from unittest import TestCase
import torch

import dlfp.utils
from dlfp.utils import SpecialSymbols

class ModuleMethodsTest(TestCase):

    def test_generate_square_subsequent_mask(self):
        sz = 5
        mask = dlfp.utils.generate_square_subsequent_mask(sz, device="cpu")
        self.assertEqual((sz, sz), mask.shape)
        # self.assertIs(torch.bool, mask.dtype)
        ninf = float("-inf")
        self.assertTrue(torch.all(torch.isclose(torch.tensor([
            [0., ninf, ninf, ninf, ninf],
            [0., 0., ninf, ninf, ninf],
            [0., 0., 0., ninf, ninf],
            [0., 0., 0., 0., ninf],
            [0., 0., 0., 0., 0.]
        ]), mask)))
        self.assertTrue(torch.equal(torch.tensor([
            [False, True, True, True, True],
            [False, False, True, True, True],
            [False, False, False, True, True],
            [False, False, False, False, True],
            [0., 0., 0., 0., 0.]
        ]), torch.isinf(mask)))



class SpecialSymbolsTest(TestCase):

    def test_as_tuple(self):
        special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        self.assertListEqual(special_symbols, SpecialSymbols().as_list())

