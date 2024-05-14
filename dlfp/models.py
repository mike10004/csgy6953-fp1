#!/usr/bin/env python3

"""Module for classes and methods relating to models."""

import math
from typing import NamedTuple
from typing import Optional

import torch
from torch import nn
from torch import Tensor
from torch.nn import Module
from torch.nn import Dropout
from torch.nn import Transformer
from torch.optim import Optimizer

import dlfp.common
from dlfp.utils import Restored

class TokenEmbedding(nn.Module):

    """Model that embeds tokenized words into smaller latent space that captures the meaning of the words.

    Reference: [Attention Is All You Need](https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).
    """

    def __init__(self, vocab_size: int, emb_size: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor) -> Tensor:
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class PositionalEncoding(nn.Module):

    """Positional encoding

    Unlike recurrent networks, self-attention does not automatically capture
    the order that words appear. We need to fix this with a special positional
    encoding. Naive positional encodings like a one-hot approach or normalizing
    the index between $[0,1]$ fail due to variable sequence lenghts.

    The clever solution in the original paper is to use a vector of sinusoidal
    functions at different frequencies.

    The video linked [here](https://www.youtube.com/watch?v=dichIcUZfOw) provides
    a high-level introduction while the article
    [here](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)
    explains more of the math behind why they work. In particular, the positional encoding
    is able to capture the *relative* distance between indices.

    """
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor) -> Tensor:
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class Cruciformer(nn.Module):

    """Transformer model.

    Source:
    """

    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 transformer_dropout_rate: float = 0.1,
                 pe_dropout_rate: float = 0.1,
                 tgt_pos_enc_disabled: bool = False,
                 batch_first: bool = False):
        super(Cruciformer, self).__init__()
        # this dropout layer was a mistake, but it is retained with rate=0 in order
        # to keep state dicts of stored checkpoints consistent
        self.input_dropout = Dropout(0.0)
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       batch_first=batch_first,
                                       dropout=transformer_dropout_rate)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=pe_dropout_rate)
        self.tgt_pos_enc_disabled = tgt_pos_enc_disabled
        self.batch_first = batch_first

    def forward(self, src: Tensor, trg: Tensor, src_mask: Tensor, tgt_mask: Tensor,
                src_padding_mask: Tensor, tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src = self.input_dropout(src)
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.tgt_tok_emb(trg)
        if not self.tgt_pos_enc_disabled:
            tgt_emb = self.positional_encoding(tgt_emb)
        if self.batch_first:
            src_emb = src_emb.transpose(0, 1)
            tgt_emb = tgt_emb.transpose(0, 1)
        outs = self.transformer.forward(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        pos_encoded = self.positional_encoding(self.src_tok_emb(src))
        return self.transformer.encoder(pos_encoded, src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        pos_encoded = self.positional_encoding(self.tgt_tok_emb(tgt))
        return self.transformer.decoder(pos_encoded, memory, tgt_mask)


class ModelHyperparametry(NamedTuple):

    nhead: int = 8
    emb_size: int = 512
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    dim_feedforward: int = 512
    transformer_dropout_rate: float = 0.1
    pe_dropout_rate: float = 0.1
    input_dropout_rate: float = 0.0  # not actually supported
    tgt_pos_enc_disabled: bool = False
    batch_first: bool = False

    @staticmethod
    def from_args(arguments: Optional[list[str]]) -> 'ModelHyperparametry':
        h = dlfp.common.nt_from_args(ModelHyperparametry, arguments, types={
            'nhead': int,
            'emb_size': int,
            'num_encoder_layers': int,
            'num_decoder_layers': int,
            'dim_feedforward': int,
            'tgt_pos_enc_disabled': int,
            'batch_first': int,
        })
        return h


def create_model(src_vocab_size: int, tgt_vocab_size: int, h: ModelHyperparametry = None):
    h = h or ModelHyperparametry()

    transformer = Cruciformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        num_encoder_layers=h.num_encoder_layers,
        num_decoder_layers=h.num_decoder_layers,
        emb_size=h.emb_size,
        nhead=h.nhead,
        dim_feedforward=h.dim_feedforward,
        transformer_dropout_rate=h.transformer_dropout_rate,
        pe_dropout_rate=h.pe_dropout_rate,
        tgt_pos_enc_disabled=h.tgt_pos_enc_disabled,
        batch_first=h.batch_first,
    )
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


class TrainHyperparametry(NamedTuple):

    epoch_count: int = 10
    batch_size: int = 128
    lr: float = 0.0001
    betas: tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-9
    train_data_shuffle_disabled: bool = False

    def create_optimizer(self, model: Module) -> Optimizer:
        return torch.optim.Adam(model.parameters(), lr=self.lr, betas=self.betas, eps=self.eps)

    @staticmethod
    def from_args(arguments: Optional[list[str]]) -> 'TrainHyperparametry':
        types = {
            'epoch_count': int,
            'batch_size': int,
            'betas': lambda s: tuple(float(b) for b in s.split(',')),
            'train_data_shuffle_disabled': int,
        }
        return dlfp.common.nt_from_args(TrainHyperparametry, arguments, types)


def get_hyperparameters(restored: Restored) -> tuple[bool, TrainHyperparametry, ModelHyperparametry]:
    train_config = (restored.extra or {}).get("train_config", {})
    train_hp_kwargs = train_config.get("train_hp", {})
    model_hp_kwargs = train_config.get("model_hp", {})
    ok = True
    if not train_hp_kwargs or not model_hp_kwargs:
        ok = False
    return ok, TrainHyperparametry(**train_hp_kwargs), ModelHyperparametry(**model_hp_kwargs)
