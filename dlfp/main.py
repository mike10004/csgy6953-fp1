#!/usr/bin/env python3

import sys
from argparse import ArgumentParser

import torch
from torch.utils.data.dataset import Dataset
from dlfp.tokens import Tokenage
from dlfp.models import Seq2SeqTransformer
from dlfp.train import create_model
from dlfp.train import Trainer

def create_tokenage(dataset: Dataset) -> Tokenage:
    raise NotImplementedError()


def main() -> int:
    seed = 0
    torch.manual_seed(seed)
    dataset = Multi30k(split=runtype, language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    tokenage = create_tokenage(dataset)
    SRC_LANGUAGE, TGT_LANGUAGE = tokenage.language_pair
    SRC_VOCAB_SIZE = len(tokenage.vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(tokenage.vocab_transform[TGT_LANGUAGE])
    BATCH_SIZE = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src_lang, tgt_lang = tokenage.language_pair
    transformer = create_model(
        src_vocab_size=len(tokenage.vocab_transform[src_lang]),
        tgt_vocab_size=len(tokenage.vocab_transform[tgt_lang]),
        DEVICE=device,
    )
    dataloader = DataLoader(data_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    return 0


if __name__ == '__main__':
    sys.exit(main())
