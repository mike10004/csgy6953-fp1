#!/usr/bin/env python3

from pathlib import Path
from unittest import TestCase

from dlfp.datasets import DatasetResolver
from dlfp.running import TrainHyperparametry
import dlfp_tests.tools
from dlfp.common import get_repo_root
from dlfp.running import ModelManager

dlfp_tests.tools.suppress_cuda_warning()


class TrainHyperparametryTest(TestCase):

    def test_from_args(self):
        with self.subTest("default"):
            c = TrainHyperparametry.from_args(None)
            self.assertEqual(TrainHyperparametry(), c)
        with self.subTest("lr"):
            c = TrainHyperparametry.from_args(["lr=0.001"])
            self.assertEqual(TrainHyperparametry(lr=0.001), c)


class ModelManagerTest(TestCase):

    def test_evaluate(self):
        device = dlfp_tests.tools.get_device()
        model, bilinguist = dlfp_tests.tools.load_restored_cruciform(get_repo_root() / "checkpoints" / "05092033-checkpoint-epoch009.pt", device=device)
        manager = ModelManager(model, bilinguist, device)
        dataset = DatasetResolver().easymark(split="valid")
        attempts = []
        manager.evaluate(dataset, suggestion_count=5, callback=attempts.append, limit=100)
