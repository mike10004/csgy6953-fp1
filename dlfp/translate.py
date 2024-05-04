#!/usr/bin/env python3

import torch
from torch import Tensor
from dlfp.utils import generate_square_subsequent_mask
from dlfp.tokens import Tokenage
from dlfp.models import Seq2SeqTransformer


class Translator:

    def __init__(self, model: Seq2SeqTransformer, tokenage: Tokenage, device):
        self.device = device
        self.model = model
        self.tokenage = tokenage

    def greedy_decode(self, src: Tensor, src_mask: Tensor, max_len: int):
        model = self.model
        start_symbol: int = self.tokenage.specials.indexes.bos
        src = src.to(self.device)
        src_mask = src_mask.to(self.device)

        memory = model.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.device)
        for i in range(max_len - 1):
            memory = memory.to(self.device)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0), device=self.device).type(torch.bool)).to(self.device)
            out = model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == self.tokenage.specials.indexes.eos:
                break
        return ys

    def translate(self, src_sentence: str) -> str:
        self.model.eval()
        SRC_LANGUAGE, TGT_LANGUAGE = self.tokenage.language_pair
        src = self.tokenage.text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(src, src_mask, max_len=num_tokens + 5).flatten()
        tgt_vocab = self.tokenage.vocab_transform[TGT_LANGUAGE]
        tgt_tokens = tgt_vocab.lookup_tokens(list(tgt_tokens.cpu().numpy()))
        return " ".join(tgt_tokens).replace("<bos>",      "").replace("<eos>", "")
