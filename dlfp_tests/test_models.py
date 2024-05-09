import itertools
from unittest import TestCase

import tabulate

from dlfp.main import CruciformerRunner
from dlfp.models import ModelHyperparametry
from dlfp.models import create_model
import torchsummary


class ModuleMethodsTest(TestCase):

    def test_hyperparameters(self):
        cr = CruciformerRunner()
        bl = cr.create_bilinguist(cr.resolve_dataset())
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
