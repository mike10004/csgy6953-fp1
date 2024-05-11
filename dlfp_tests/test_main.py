#!/usr/bin/env python3

import os
import glob
import tempfile
from pathlib import Path
from unittest import TestCase

import dlfp.main
from dlfp.main import CruciformerRunner
import dlfp.running
from dlfp.running import Runnable
from dlfp.running import TrainConfig
from dlfp.train import TrainLoaders
from dlfp.utils import Restored


class ShortDatasetCruciformerRunner(CruciformerRunner):

    def _to_loaders(self, r: Runnable, train_config: TrainConfig):
        return TrainLoaders.from_datasets(
            r.superset.train.slice(0, 100),
            r.superset.valid.slice(0, 10),
            collate_fn=r.bilinguist.collate,
            batch_size=train_config.train_hp.batch_size,
            train_shuffle=not train_config.train_hp.train_data_shuffle_disabled,
        )


# def resolve_dataset(self, dataset_name: str = None) -> DataSuperset:
    #     superset = super().resolve_dataset(dataset_name)
    #     return DataSuperset(
    #         superset.train.slice(0, 100),
    #         superset.valid.slice(0, 10),
    #         superset.src_tokenizer_name,
    #         superset.src_tokenizer_language,
    #         superset.tgt_tokenizer_name,
    #         superset.tgt_tokenizer_language,
    #     )



class ModuleMethodsTest(TestCase):

    def test_main_charmark(self):
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