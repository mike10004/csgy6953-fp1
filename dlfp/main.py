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
    return 0


if __name__ == '__main__':
    sys.exit(main())
