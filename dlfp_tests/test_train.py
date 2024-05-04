#!/usr/bin/env python3

from unittest import TestCase
import numpy as np
import torch
import torch.random
from torch import Tensor
from torch.utils.data import DataLoader

from dlfp.models import Seq2SeqTransformer
from dlfp.train import create_model
from dlfp.train import Trainer
from dlfp.train import TrainLoaders
from dlfp.utils import noop
import dlfp_tests.tools


class TrainerTest(TestCase):

    verbose: bool = False

    def test_create_mask(self):
        train_dataset = dlfp_tests.tools.load_multi30k_dataset(split='train')
        tokenage = dlfp_tests.tools.init_multi30k_de_en_tokenage()
        dataloader = DataLoader(train_dataset, batch_size=16, collate_fn=tokenage.collate_fn)
        src, tgt = next(iter(dataloader))
        tgt_input = tgt[:-1, :]
        masks = Trainer.create_mask_static(src, tgt_input, device="cpu", pad_idx=tokenage.specials.indexes.pad)
        self.assertSetEqual({torch.bool}, set(mask.dtype for mask in masks))
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = masks


    def test_train(self):
        with torch.random.fork_rng():
            torch.random.manual_seed(0)
            tokenage = dlfp_tests.tools.init_multi30k_de_en_tokenage()
            batch_size = 2
            train_dataset = dlfp_tests.tools.load_multi30k_dataset(split='train')
            valid_dataset = dlfp_tests.tools.load_multi30k_dataset(split='valid')
            train_dataset = dlfp_tests.tools.truncate_dataset(train_dataset, size=batch_size * 100)
            valid_dataset = dlfp_tests.tools.truncate_dataset(valid_dataset, size=batch_size * 2)
            device = dlfp_tests.tools.get_device()
            src_lang, tgt_lang = tokenage.language_pair
            model = create_model(
                src_vocab_size=len(tokenage.vocab_transform[src_lang]),
                tgt_vocab_size=len(tokenage.vocab_transform[tgt_lang]),
                DEVICE=device,
            )
            trainer = Trainer(model, pad_idx=tokenage.specials.indexes.pad, device=device)
            trainer.hide_progress = not self.verbose
            callback = noop if not self.verbose else None
            loaders = TrainLoaders.from_datasets(train_dataset, valid_dataset, collate_fn=tokenage.collate_fn)
            epoch_count = 1
            results = trainer.train(loaders, epoch_count, callback=callback)
            self.assertEqual(epoch_count, len(results))
            weights = get_weight_sample(model, trainer, next(iter(loaders.valid)))
            print(weights.shape)
            expected = [
                0.0951, 0.1064, 0.1283, 0.1175, 0.0844, 0.0741, 0.0697, 0.0753, 0.0837, 0.0854, 0.0801, 0.0000, 0.0000,
                0.0961, 0.1084, 0.1252, 0.1174, 0.0904, 0.0783, 0.0719, 0.0744, 0.0799, 0.0791, 0.0789, 0.0000, 0.0000,
                0.1053, 0.1108, 0.1099, 0.0985, 0.0867, 0.0769, 0.0739, 0.0827, 0.0870, 0.0916, 0.0767, 0.0000, 0.0000,
                0.0975, 0.0971, 0.0850, 0.0901, 0.0857, 0.0787, 0.0788, 0.0976, 0.0995, 0.1015, 0.0883, 0.0000, 0.0000,
                0.0896, 0.1009, 0.0965, 0.0914, 0.0936, 0.0796, 0.0747, 0.0911, 0.0961, 0.0940, 0.0925, 0.0000, 0.0000,
                0.0934, 0.1056, 0.0828, 0.0891, 0.0875, 0.0849, 0.0895, 0.0938, 0.0994, 0.0895, 0.0846, 0.0000, 0.0000,
                0.0985, 0.1121, 0.0989, 0.0905, 0.0783, 0.0846, 0.0807, 0.0911, 0.0930, 0.0910, 0.0812, 0.0000, 0.0000,
                0.1074, 0.1090, 0.1019, 0.0935, 0.0824, 0.0838, 0.0807, 0.0856, 0.0910, 0.0867, 0.0781, 0.0000, 0.0000,
                0.1068, 0.1162, 0.1167, 0.1061, 0.0868, 0.0829, 0.0737, 0.0780, 0.0795, 0.0786, 0.0746, 0.0000, 0.0000,
                0.1022, 0.1245, 0.1138, 0.1109, 0.0877, 0.0858, 0.0786, 0.0783, 0.0765, 0.0730, 0.0687, 0.0000, 0.0000,
                0.1038, 0.1178, 0.1076, 0.1038, 0.0888, 0.0842, 0.0816, 0.0791, 0.0794, 0.0770, 0.0770, 0.0000, 0.0000,
                0.0993, 0.1197, 0.1021, 0.1055, 0.0893, 0.0768, 0.0816, 0.0774, 0.0791, 0.0849, 0.0842, 0.0000, 0.0000,
                0.0968, 0.1176, 0.1023, 0.1008, 0.0866, 0.0754, 0.0813, 0.0755, 0.0803, 0.0919, 0.0915, 0.0000, 0.0000,
            ]
            expected = np.array(expected, dtype=np.float32).reshape(13, 13)
            weight_slice = weights[0].detach().cpu().numpy()
            np.testing.assert_array_almost_equal(expected, weight_slice, decimal=4)


def get_weight_sample(model: Seq2SeqTransformer, trainer: Trainer, tensor_pair) -> Tensor:
    model.eval()
    source, target = tensor_pair
    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = trainer.create_mask(source, target)
    src_emb = model.positional_encoding(model.src_tok_emb(source))
    encoderlayer = model.transformer.encoder.layers[0]
    x = encoderlayer.norm1(src_emb)
    output, weights = encoderlayer.self_attn(x, x, x, attn_mask = src_mask, key_padding_mask=src_padding_mask)
    return weights