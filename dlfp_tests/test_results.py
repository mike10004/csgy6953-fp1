#!/usr/bin/env python3

from pathlib import Path
from unittest import TestCase

import dlfp.common
import dlfp.models
from dlfp.models import ModelHyperparametry
from dlfp.results import Evaluation
from dlfp.results import Collated
import dlfp_tests.tools
from dlfp.utils import Restored


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


class ParameterCounter:

    def count_parameters(self, checkpoint: Restored, model_hp: ModelHyperparametry) -> str:
        dataset_name = checkpoint.extra["metadata"]["dataset_name"]
        from torchsummary import summary
        import dlfp.datasets
        import dlfp.models
        dataset = dlfp.datasets.DatasetResolver().by_name(dataset_name, "train")
        src_lang, tgt_lang = dlfp.datasets.get_languages(dataset)
        model = dlfp.models.create_model(src_lang.vocab_size(), tgt_lang.vocab_size(), model_hp)
        stats = summary(model, verbose=0)
        for count, suffix in [
            (1_000_000, "m"),
            (1_000, "k"),
        ]:
            if stats.trainable_params > count:
                return f"{stats.trainable_params/count:.1f}{suffix}"
        return str(stats.trainable_params)


class ParameterCounterTest(TestCase):

    def test_charmark(self):
        checkpoint = Restored.from_file(dlfp.common.get_repo_root() / "checkpoints" / "charmark-final.pt", device="cpu")
        _, _, model_hp = dlfp.models.get_hyperparameters(checkpoint)
        print("charmark", ParameterCounter().count_parameters(checkpoint, model_hp))
    def test_onemark(self):
        checkpoint = Restored.from_file(dlfp.common.get_repo_root() / "checkpoints" / "onemark-final.pt", device="cpu")
        _, _, model_hp = dlfp.models.get_hyperparameters(checkpoint)
        print("onemark", ParameterCounter().count_parameters(checkpoint, model_hp))