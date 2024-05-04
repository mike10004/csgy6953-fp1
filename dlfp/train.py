#!/usr/bin/env python3

from typing import Callable
from typing import NamedTuple
from typing import Optional

import torch
from torch import nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from dlfp.models import Seq2SeqTransformer
from dlfp.utils import PhrasePairDataset
from dlfp.utils import generate_square_subsequent_mask
from dlfp.utils import equal_scalar
from dlfp.utils import EpochResult

LossFunction = Callable[[Tensor, Tensor], Tensor]


def create_model(src_vocab_size: int, tgt_vocab_size: int, DEVICE):
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3


    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                     NHEAD, src_vocab_size, tgt_vocab_size, FFN_HID_DIM)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # Architecture
    transformer = transformer.to(DEVICE)
    return transformer


def create_optimizer(model: nn.Module, lr=0.0001, betas=(0.9, 0.98), eps=1e-9):
    return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)


def _print_result(epoch_result: EpochResult):
    print(f"Epoch {epoch_result.epoch_index + 1:2d}: Train loss {epoch_result.train_loss:.3f}; Valid loss {epoch_result.valid_loss:.3f}")




class TrainLoaders(NamedTuple):

    train: DataLoader
    valid: DataLoader

    @staticmethod
    def from_datasets(train_dataset: PhrasePairDataset,
                      valid_dataset: PhrasePairDataset,
                      *,
                      collate_fn,
                      batch_size: int = 128,
                      valid_batch_size: Optional[int] = None) -> 'TrainLoaders':
        valid_batch_size = valid_batch_size or batch_size
        train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=valid_batch_size, collate_fn=collate_fn)
        return TrainLoaders(train_loader, valid_loader)




class Trainer:

    def __init__(self, model: Seq2SeqTransformer, pad_idx: int, device, loss_fn: LossFunction = None):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn or torch.nn.CrossEntropyLoss(ignore_index=pad_idx)
        self.pad_idx = pad_idx
        self.optimizer_factory = create_optimizer
        self.hide_progress = False

    def create_mask(self, src, tgt):
        return self.create_mask_static(src, tgt, device=self.device, pad_idx=self.pad_idx)

    @staticmethod
    def create_mask_static(src, tgt, device, pad_idx):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device=device).type(torch.bool)
        src_mask = torch.zeros((src_seq_len, src_seq_len),device=device).type(torch.bool)

        src_padding_mask = equal_scalar(src, pad_idx).transpose(0, 1)
        tgt_padding_mask = equal_scalar(tgt, pad_idx).transpose(0, 1)
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def train(self, loaders: TrainLoaders, epoch_count: int, callback: Callable[[EpochResult], None] = None) -> list[EpochResult]:
        callback = callback or _print_result
        optimizer = self.optimizer_factory(self.model)
        epoch_results = []
        for epoch in range(epoch_count):
            train_loss = self.run(loaders.train, runtype='train', optimizer=optimizer)
            val_loss = self.run(loaders.valid, runtype='valid')
            result = EpochResult(epoch, train_loss, val_loss)
            callback(result)
            epoch_results.append(result)
        return epoch_results

    def run(self, dataloader: DataLoader, runtype: str, optimizer: Optimizer = None) -> float:
        assert runtype in {'train', 'valid', 'test'}
        model = self.model
        if runtype == 'train':
            model.train()
        else:
            model.eval()

        losses = 0

        num_points = 0

        for src, tgt in tqdm(dataloader, total=len(dataloader), disable=self.hide_progress):
            num_points += src.shape[1]
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask,
                           tgt_padding_mask, src_padding_mask)

            if runtype == 'train':
                optimizer.zero_grad()

            tgt_out = tgt[1:, :]
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

            if runtype == 'train':
                loss.backward()
                optimizer.step()

            losses += loss.item()

        return losses / num_points