import itertools
from unittest import TestCase

import tabulate
import torch.random
from torch.utils.data import DataLoader

import dlfp.models
from dlfp.main import CruciformerRunner
from dlfp.models import ModelHyperparametry
from dlfp.models import create_model
import torchsummary
import dlfp_tests.tools
from dlfp.train import Trainer

dlfp_tests.tools.suppress_cuda_warning()

class ModuleMethodsTest(TestCase):

    def test_hyperparameters(self):
        cr = CruciformerRunner()
        bl = cr.create_bilinguist(cr.resolve_dataset("easymark"))
        src_vocab_size = len(bl.source.vocab)
        tgt_vocab_size = len(bl.target.vocab)
        table = []
        headers = ["n_enc_layers", "n_dec_layers", "nhead", "emb_size", "ffn_dim", "trainable params"]
        hyperparameter_counts = []
        for num_encoder_layers, num_decoder_layers, nhead, ffn_dim, emb_size in itertools.product([3], [3], [8], [512, 256], [512, 256]):
            h = ModelHyperparametry(
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=ffn_dim,
                emb_size=emb_size,
            )
            model = create_model(src_vocab_size, tgt_vocab_size, h)
            summary = torchsummary.summary(model, verbose=0)
            table.append((num_encoder_layers, num_decoder_layers, nhead, emb_size, ffn_dim, summary.trainable_params))
            hyperparameter_counts.append(summary.trainable_params)
        print(tabulate.tabulate(table, headers=headers, tablefmt="simple_grid"))
        self.assertEqual(len(hyperparameter_counts), len(set(hyperparameter_counts)), "expect unique parameter counts")

    def test_shapes(self):
        cr = CruciformerRunner()
        dataset = cr.resolve_dataset("easymark")
        bl = cr.create_bilinguist(dataset)
        with torch.random.fork_rng():
            torch.random.manual_seed(13251)
            batch_size=31
            train_loader = DataLoader(dataset.train, batch_size=batch_size, collate_fn=bl.collate, shuffle=True)
            src_vocab_size = len(bl.source.vocab)
            tgt_vocab_size = len(bl.target.vocab)
            h = ModelHyperparametry()
            model = dlfp.models.create_model(src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size, h=h)
            device = dlfp_tests.tools.get_device()
            model = model.to(device)
            with torch.random.fork_rng():
                torch.random.manual_seed(12345)
                src, tgt = next(iter(train_loader))
                src = src.to(device)
                tgt = tgt.to(device)
                print(src.shape, tgt.shape)
                tgt_input = tgt[:-1, :]
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = Trainer.create_mask_static(src, tgt_input, device=device, pad_idx=1)
                # def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor, tgt_mask: Tensor,
                #                 src_padding_mask: Tensor, tgt_padding_mask: Tensor,
                #                 memory_key_padding_mask: Tensor):
                args = [src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask]
                # model(*args)
                torchsummary.summary(model, input_data=args)
