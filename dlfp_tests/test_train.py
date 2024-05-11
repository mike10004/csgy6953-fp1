#!/usr/bin/env python3

from unittest import TestCase
import torch
import torch.random
from torch import Tensor
from torch.utils.data import DataLoader

from dlfp.models import ModelHyperparametry
from dlfp.models import Cruciformer
from dlfp.models import create_model
from dlfp.train import Trainer
from dlfp.train import TrainLoaders
from dlfp.common import noop
import dlfp_tests.tools

dlfp_tests.tools.suppress_cuda_warning()


class TrainerTest(TestCase):

    verbose: bool = False

    def test_create_mask(self):
        train_dataset = dlfp_tests.tools.load_multi30k_dataset(split='train')
        bilinguist = dlfp_tests.tools.get_multi30k_de_en_bilinguist()
        dataloader = DataLoader(train_dataset, batch_size=16, collate_fn=bilinguist.collate)
        src, tgt = next(iter(dataloader))
        tgt_input = tgt[:-1, :]
        masks = Trainer.create_mask_static(src, tgt_input, device="cpu", pad_idx=bilinguist.source.specials.indexes.pad)
        self.assertSetEqual({torch.bool}, set(mask.dtype for mask in masks))


    def test_train(self):
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            bilinguist = dlfp_tests.tools.get_multi30k_de_en_bilinguist()
            batch_size = 16
            train_dataset = dlfp_tests.tools.load_multi30k_dataset(split='train')
            valid_dataset = dlfp_tests.tools.load_multi30k_dataset(split='valid')
            train_dataset = dlfp_tests.tools.truncate_dataset(train_dataset, size=batch_size * 20)
            valid_dataset = dlfp_tests.tools.truncate_dataset(valid_dataset, size=batch_size * 2)
            device = dlfp_tests.tools.get_device()
            model_hp = ModelHyperparametry(batch_first=False)
            model = create_model(
                src_vocab_size=len(bilinguist.source.vocab),
                tgt_vocab_size=len(bilinguist.target.vocab),
                h=model_hp,
            ).to(device)
            trainer = Trainer(model, pad_idx=bilinguist.source.specials.indexes.pad, device=device)
            trainer.hide_progress = not self.verbose
            callback = noop if not self.verbose else None
            loaders = TrainLoaders.from_datasets(train_dataset, valid_dataset, batch_size=batch_size, collate_fn=bilinguist.collate)
            epoch_count = 2
            results = trainer.train(loaders, epoch_count, callback=callback)
            self.assertEqual(epoch_count, len(results))
            model.eval()
            # loaders = TrainLoaders.from_datasets(train_dataset, valid_dataset, batch_size=batch_size, collate_fn=bilinguist.collate)
            mean_loss = trainer.run(loaders.valid, 'valid')
            print("mean loss", mean_loss)
            expected_mean_loss = 0.4069085568189621
            self.assertAlmostEqual(expected_mean_loss, mean_loss, delta=0.001)


def get_weight_sample(model: Cruciformer, trainer: Trainer, tensor_pair) -> Tensor:
    model.eval()
    source, target = tensor_pair
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = trainer.create_mask(source, target)
    src_emb = model.positional_encoding(model.src_tok_emb(source)).transpose(0, 1)
    encoderlayer = model.transformer.encoder.layers[0]
    x = encoderlayer.norm1(src_emb)
    output, weights = encoderlayer.self_attn(x, x, x, attn_mask = src_mask, key_padding_mask=src_padding_mask)
    return weights