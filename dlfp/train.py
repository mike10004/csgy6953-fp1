#!/usr/bin/env python3

from typing import Callable

import torch
from torch import nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from dlfp.models import Seq2SeqTransformer
from dlfp.utils import generate_square_subsequent_mask

LossFunction = Callable[[Tensor, Tensor], Tensor]


def create_model(src_vocab_size: int, tgt_vocab_size: int, DEVICE):
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3


    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                     NHEAD, src_vocab_size, tgt_vocab_size, FFN_HID_DIM)
    count = 0
    for p in transformer.parameters():
        count += p.numel()
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Architecture
    transformer = transformer.to(DEVICE)
    return transformer


class Trainer:

    def __init__(self, model: Seq2SeqTransformer, loss_fn: LossFunction, pad_idx: int, device):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.pad_idx = pad_idx

    def create_mask(self, src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device=self.device)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=self.device).type(torch.bool)

        src_padding_mask = (src == self.pad_idx).transpose(0, 1)
        tgt_padding_mask = (tgt == self.pad_idx).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def run(self, dataloader: DataLoader, runtype: str, optimizer: Optimizer = None):
        assert runtype in ['train', 'valid']
        model = self.model
        if runtype == 'train':
            model.train()
        else:
            model.eval()

        losses = 0

        num_points = 0

        for src, tgt in dataloader:
            num_points += src.shape[1]
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask,
                           tgt_padding_mask, src_padding_mask)

            if runtype == 'train': optimizer.zero_grad()

            tgt_out = tgt[1:, :]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

            if runtype == 'train':
                loss.backward()
                optimizer.step()

            losses += loss.item()

        return losses / num_points
