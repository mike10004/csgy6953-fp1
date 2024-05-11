#!/usr/bin/env python3

import os
import glob
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase

import dlfp.main
import dlfp.common
import dlfp_tests.tools
from dlfp.main import CruciformerRunner
import dlfp.running
from dlfp.running import Runnable
from dlfp.running import TrainConfig
from dlfp.train import TrainLoaders
from dlfp.utils import Restored
from dlfp.translate import CruciformerCharmarkNodeNavigator
from dlfp.translate import CruciformerNodeNavigator
from dlfp.translate import CruciformerOnemarkNodeNavigator


dlfp_tests.tools.suppress_cuda_warning()

class ShortDatasetCruciformerRunner(CruciformerRunner):

    def _to_loaders(self, r: Runnable, train_config: TrainConfig):
        return TrainLoaders.from_datasets(
            r.superset.train.slice(0, 100),
            r.superset.valid.slice(0, 10),
            collate_fn=r.bilinguist.collate,
            batch_size=train_config.train_hp.batch_size,
            train_shuffle=not train_config.train_hp.train_data_shuffle_disabled,
        )


class ModuleMethodsTest(TestCase):

    def test_main_charmark_train(self, save_model: bool = False):
        with tempfile.TemporaryDirectory() as tempdir:
            args = [
                "--mode", "train",
                "--dataset", "charmark",
                "-t", "epoch_count=1",
                "-t", "batch_size=8",
                "--output", str(tempdir),
            ]
            dlfp.running.main(ShortDatasetCruciformerRunner(), args)
            checkpoint_file = glob.glob(os.path.join(tempdir, "checkpoints", "*", "*.pt"))[0]
            checkpoint = Restored.from_file(Path(checkpoint_file), device="cpu")
            tgt_emb_weight_shape = checkpoint.model_param_shapes()["tgt_tok_emb.embedding.weight"]
            self.assertTupleEqual((30, 512), tgt_emb_weight_shape)
            if save_model:
                shutil.copyfile(checkpoint_file, dlfp.common.get_repo_root() / "checkpoints" / "charmark-test-checkpoint.pt")

    def test_main_charmark_eval(self):
        checkpoint_file = dlfp.common.get_repo_root() / "checkpoints" / "charmark-test-checkpoint.pt"
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)
            results_file = tempdir / "results.csv"
            args = [
                "--mode", "eval",
                "--dataset", "charmark",
                "-f", str(checkpoint_file),
                "--output", str(results_file),
                "-e", "node_strategy=max_ranks=3,2,1",
                "-e", "limit=10",
                "-e", "shuffle_seed=1001",
            ]
            dlfp.running.main(CruciformerRunner(), args)
            print(results_file.read_text())


class CruciformerRunnerTest(TestCase):

    def test_create_node_strategy(self):
        runner = CruciformerRunner()
        for dataset, expected_type in [
            ("easymark", CruciformerNodeNavigator),
            ("onemark", CruciformerOnemarkNodeNavigator),
            ("charmark", CruciformerCharmarkNodeNavigator),
        ]:
            with self.subTest(dataset, kwargs="empty"):
                strat = runner.create_node_strategy(None, dataset)
                navigator = strat.navigator_factory("somephrase")
                self.assertIsInstance(navigator, expected_type)
        with self.subTest("custom"):
            custom_nav_strat = runner.create_node_strategy("max_ranks=3,2,1", "charmark")
            navigator = custom_nav_strat.navigator_factory("foo")
            self.assertIsInstance(navigator, CruciformerCharmarkNodeNavigator)
            self.assertTupleEqual(navigator.max_ranks[1:], (3,2,1))