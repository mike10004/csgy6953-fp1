#!/usr/bin/env python3

from pathlib import Path
from unittest import TestCase
from dlfp.running import TrainHyperparametry


class TrainHyperparametryTest(TestCase):

    def test_from_args(self):
        with self.subTest("default"):
            c = TrainHyperparametry.from_args(None)
            self.assertEqual(TrainHyperparametry(), c)
        with self.subTest("lr"):
            c = TrainHyperparametry.from_args(["lr=0.001"])
            self.assertEqual(TrainHyperparametry(lr=0.001), c)