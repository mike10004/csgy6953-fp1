#!/usr/bin/env python3
import logging
import os
import csv
import glob
import tempfile
from pathlib import Path
from unittest import TestCase

import dlfp.models
from dlfp.datasets import DatasetResolver
from dlfp.main import CruciformerRunner
from dlfp.running import EvalConfig
from dlfp.running import TrainHyperparametry
import dlfp_tests.tools
from dlfp.common import get_repo_root
from dlfp.running import Runnable
from dlfp.utils import Restored

dlfp_tests.tools.suppress_cuda_warning()
_log = logging.getLogger(__name__)

class TrainHyperparametryTest(TestCase):

    def test_from_args(self):
        with self.subTest("default"):
            c = TrainHyperparametry.from_args(None)
            self.assertEqual(TrainHyperparametry(), c)
        with self.subTest("lr"):
            c = TrainHyperparametry.from_args(["lr=0.001"])
            self.assertEqual(TrainHyperparametry(lr=0.001), c)


class RunnableTest(TestCase):

    def test_run_eval(self):
        logging.basicConfig(level=logging.INFO)
        device = dlfp_tests.tools.get_device()
        checkpoint_file = get_repo_root() / "checkpoints" / "05092329-checkpoint-epoch009.pt"
        restored = Restored.from_file(checkpoint_file, device=device)
        # model, bilinguist, model_hp = dlfp_tests.tools.load_restored_cruciform(, device=device)
        runner = CruciformerRunner()
        ok, train_hp, model_hp = dlfp.models.get_hyperparameters(restored)
        self.assertTrue(ok, "failed to restore model hyperparameters")
        limit, shuffle_seed = 3, 965835
        eval_config = EvalConfig(limit=limit, shuffle_seed=shuffle_seed, nodes_folder=Path("auto"))
        with tempfile.TemporaryDirectory(prefix="fp1") as tempdir:
            output_file = Path(tempdir) / "output.csv"
            _log.debug("run_eval starting")
            runner.run_eval(restored, "easymark", model_hp, device, output_file, eval_config)
            _log.debug("run_eval complete")
            with open(output_file, "r") as ifile:
                attempts = list(csv.DictReader(ifile))
            self.assertEqual(limit, len(attempts))
            nodes_folder = Path(glob.glob(os.path.join(output_file.parent, "*-nodes"))[0])
            nodes_files = sorted(nodes_folder.iterdir())
            self.assertEqual(limit, len(nodes_files))
            print(nodes_files[0].read_text())
