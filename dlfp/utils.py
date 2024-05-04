#!/usr/bin/env python3

from pathlib import Path

import torch

def get_repo_root() -> Path:
    return Path(__file__).absolute().parent.parent


def generate_square_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

