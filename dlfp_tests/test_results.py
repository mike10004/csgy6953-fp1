#!/usr/bin/env python3

from pathlib import Path
from unittest import TestCase

from dlfp.results import Evaluation
from dlfp.results import Collated
import dlfp_tests.tools



class EvaluationTest(TestCase):

    def test_from_attempts_file(self):
        attempts_file = dlfp_tests.tools.get_testdata_dir() / "0512-om4" / "23-5-20240512-1659-attempts-checkpoint-epoch009_valid_r98765_s1000_20240512-224503.csv"
        evalu = Evaluation.from_attempts_file(attempts_file)
        self.assertTupleEqual((5,), evalu.max_ranks, "max_ranks")


class CollatedTest(TestCase):

    def test_from_info_file(self):
        info_file: Path = dlfp_tests.tools.get_testdata_dir() / "0512-om4" / "23-0-20240512-1659-info.json"
        collated = Collated.from_info_file(info_file)
        self.assertEqual(9, len(collated.evaluations), "evaluation count")
        self.assertDictEqual({
            'transformer_dropout_rate': 0.0,
        }, collated.nondefault_hyperparameters())
