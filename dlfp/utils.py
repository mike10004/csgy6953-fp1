#!/usr/bin/env python3

from pathlib import Path

def get_repo_root() -> Path:
    return Path(__file__).absolute().parent.parent