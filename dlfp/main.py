#!/usr/bin/env python3

import sys
from argparse import ArgumentParser

import torch

from dlfp.tokens import Tokenage
from dlfp.models import Seq2SeqTransformer


def create_tokenage() -> Tokenage:
    raise NotImplementedError()


def main() -> int:
    seed = 0
    torch.manual_seed(seed)

    tokenage = create_tokenage()
    SRC_LANGUAGE, TGT_LANGUAGE = tokenage.language_pair
    SRC_VOCAB_SIZE = len(tokenage.vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(tokenage.vocab_transform[TGT_LANGUAGE])
    BATCH_SIZE = 128
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_iter = Multi30k(split=runtype, language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    dataloader = DataLoader(data_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    return 0


if __name__ == '__main__':
    sys.exit(main())
