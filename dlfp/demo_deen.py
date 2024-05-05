#!/usr/bin/env python3

import sys
from argparse import ArgumentParser
from pathlib import Path

import tabulate
import torch
from dlfp.tokens import Biglot
from dlfp.tokens import Linguist
from dlfp.models import Seq2SeqTransformer
from dlfp.train import TrainLoaders
from dlfp.train import create_model
from dlfp.train import Trainer
from dlfp.utils import Checkpointer
from dlfp.utils import LanguageCache
from dlfp.utils import Restored
from dlfp.utils import EpochResult
from dlfp.utils import PhrasePairDataset
from dlfp.translate import Translator
import dlfp.utils



def print_translations(model: Seq2SeqTransformer, tokenage: Biglot, dataset: PhrasePairDataset, device, limit: int = 5):
    limit = limit or 5
    translator = Translator(model, tokenage, device)
    for index, (de_phrase, en_phrase) in enumerate(dataset):
        if index >= limit:
            break
        if index > 0:
            print()
        print(f"{index: 2d} de: {de_phrase}")
        print(f"{index: 2d} en: {en_phrase}")
        translation = translator.translate(de_phrase).strip()
        print(f"{index: 2d} mx: {translation}")

def main() -> int:
    parser = ArgumentParser(description="Run German-to-English translation demo")
    parser.add_argument("-m", "--mode", choices=("train", "eval"), default="train")
    parser.add_argument("-o", "--output", metavar="DIR", help="output root directory")
    parser.add_argument("-f", "--file", metavar="FILE", help="checkpoint file for eval mode")
    parser.add_argument("--limit", type=int, metavar="N", help="eval mode phrase limit")
    args = parser.parse_args()
    seed = 0
    torch.manual_seed(seed)
    train_dataset = dlfp.utils.multi30k_de_en(split='train')
    valid_dataset = dlfp.utils.multi30k_de_en(split='valid')
    cache = LanguageCache()
    src_ling = Linguist.from_language(cache.get(train_dataset, "de", "spacy", "de_core_news_sm"))
    tgt_ling = Linguist.from_language(cache.get(train_dataset, "en", "spacy", "en_core_web_sm"))
    biglot = Biglot(src_ling, tgt_ling)
    batch_size = 128
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = create_model(
        src_vocab_size=len(biglot.source.language.vocab),
        tgt_vocab_size=len(biglot.target.language.vocab),
        DEVICE=device,
    )
    if args.mode == "eval":
        checkpoint_file = args.file
        if not checkpoint_file:
            parser.error("checkpoint file must be specified")
            return 1
        restored = Restored.from_file(checkpoint_file, device=device)
        model.load_state_dict(restored.model_state_dict)
        print_translations(model, biglot, valid_dataset, device, limit=args.limit)
        return 0
    elif args.mode == "train":
        trainer = Trainer(model, pad_idx=biglot.source.language.specials.indexes.pad, device=device)
        loaders = TrainLoaders.from_datasets(train_dataset, valid_dataset, collate_fn=biglot.collate, batch_size=batch_size)
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
    else:
        parser.error("BUG unhandled mode")
        return 2


if __name__ == '__main__':
    sys.exit(main())
