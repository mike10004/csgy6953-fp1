#!/usr/bin/env python3

import sys
from argparse import ArgumentParser
from pathlib import Path

import tabulate
import torch
from torchtext.data.utils import get_tokenizer
from dlfp.tokens import Tokenage
from dlfp.train import TrainLoaders
from dlfp.train import create_model
from dlfp.train import Trainer
from dlfp.utils import Checkpointer
from dlfp.utils import EpochResult
import dlfp.utils


def main() -> int:
    parser = ArgumentParser(description="Run German-to-English translation demo")
    parser.add_argument("-m", "--mode", choices=("train", "eval"), default="train")
    parser.add_argument("-o", "--output", metavar="DIR", help="output root directory")
    parser.add_argument("-f", "--file", metavar="FILE", help="checkpoint file for eval mode")
    args = parser.parse_args()
    if args.mode != "train":
        print("'eval' mode not yet supported", file=sys.stderr)
        return 1
    seed = 0
    torch.manual_seed(seed)
    language_pair = "de", "en"
    tokenizers = {
        "de": get_tokenizer('spacy', language='de_core_news_sm'),
        "en": get_tokenizer('spacy', language='en_core_web_sm'),
    }
    train_dataset = dlfp.utils.multi30k_de_en(split='train')
    valid_dataset = dlfp.utils.multi30k_de_en(split='valid')
    print("loading vocab")
    tokenage = Tokenage.from_token_transform(language_pair, tokenizers, train_dataset)
    batch_size = 128
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    src_lang, tgt_lang = tokenage.language_pair
    model = create_model(
        src_vocab_size=len(tokenage.vocab_transform[src_lang]),
        tgt_vocab_size=len(tokenage.vocab_transform[tgt_lang]),
        DEVICE=device,
    )
    trainer = Trainer(model, pad_idx=tokenage.specials.indexes.pad, device=device)
    loaders = TrainLoaders.from_datasets(train_dataset, valid_dataset, collate_fn=tokenage.collate_fn, batch_size=batch_size)
    epoch_count = 10
    checkpoints_dir = Path(args.output or ".") / f"checkpoints/{dlfp.utils.timestamp()}"
    print(f"writing checkpoints to {checkpoints_dir}")
    checkpointer = Checkpointer(checkpoints_dir, model)
    results = trainer.train(loaders, epoch_count, callback=checkpointer.checkpoint)
    results_table = [
        (r.epoch_index, r.train_loss, r.valid_loss)
        for r in results
    ]
    print(tabulate.tabulate(results_table, headers=EpochResult._fields))
    return 0


if __name__ == '__main__':
    sys.exit(main())
