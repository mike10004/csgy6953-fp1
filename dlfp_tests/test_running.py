#!/usr/bin/env python3

from pathlib import Path
from unittest import TestCase
from dlfp.running import TrainConfig


class TrainConfigTest(TestCase):

    def test_from_args(self):
        checkpoints_dir = Path(__file__).parent
        with self.subTest("default"):
            c = TrainConfig.from_args("foo", checkpoints_dir, None)
            self.assertEqual(TrainConfig("foo", checkpoints_dir), c)
        with self.subTest("lr"):
            c = TrainConfig.from_args("foo", checkpoints_dir, ["lr=0.001"])
            self.assertEqual(TrainConfig("foo", checkpoints_dir, lr=0.001), c)