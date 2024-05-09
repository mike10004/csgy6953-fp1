#!/usr/bin/env python3

import json
from pathlib import Path
from random import Random
from typing import Dict
from typing import List
from typing import Tuple
from unittest import TestCase

import dlfp_tests.tools
from dlfp.datasets import DatasetResolver
from dlfp.baseline.cs import Words_Offline
from dlfp.baseline.cs import SuggestionDict
from dlfp.utils import PhrasePairDataset


SKIP_ALL = True

# Pattern is (initial_X, initial_Y), direction(D or A), length
# THE WORDS INSERTED SHOULD HAVE THEIR STARTING LETTER CAPITALIZED (For clue names)
# "": {"start":(), "direction":"", "length": },
CROSSWORD_GRID = {
    "__ of bad news": {"start":(0, 1), "direction":"D", "length": 6},
    "Posture problem": {"start":(0, 3), "direction":"D", "length": 5},
    "Loads": {"start":(0, 4), "direction":"D", "length": 6},
    "Laundry appliance": {"start":(0, 5), "direction":"D", "length": 5},
    "Lectured": {"start":(1, 0), "direction":"D", "length": 5},
    "One who weeps": {"start":(1, 2), "direction":"D", "length": 5},
    "Grassy clump": {"start":(0, 3), "direction":"A", "length": 3},
    "Pie chart portion": {"start":(1, 0), "direction":"A", "length": 6},
    "\"Scary Movie,\" e.g.": {"start":(2, 0), "direction":"A", "length": 6},
    "Maryland's state bird": {"start":(3, 0), "direction":"A", "length": 6},
    "Something worth saving": {"start":(4, 0), "direction":"A", "length": 6},
    "\"To __ is human\"": {"start":(5, 0), "direction":"A", "length": 3}
}


def build_clue_dict() -> Dict[str, int]:
    clues = dict()
    for clue in CROSSWORD_GRID:
        clues[clue] = CROSSWORD_GRID[clue]["length"]
    return clues


class SmallGinsberg:

    def __init__(self):
        self.rng = Random(123855)
        self.truncated_size = 600000  # use 10% of full dataset

    def load_dataset(self) -> PhrasePairDataset:
        phrase_pairs = []
        with open(DatasetResolver().data_root / "datasets" / "ginsberg" / "all-clues.txt", "r", encoding='latin-1') as ifile:
            for line in ifile:
                line = line.rstrip()
                parts = line.split(maxsplit=4)
                answer, clue = parts[0], parts[-1]
                phrase_pairs.append((clue, answer))
        if self.rng is not None:
            self.rng.shuffle(phrase_pairs)
        if self.truncated_size is not None:
            phrase_pairs = phrase_pairs[:self.truncated_size]
        return PhrasePairDataset("ginsberg", phrase_pairs, ("clue", "answer"))


def zip_safe(a, b):
    if len(a) != len(b):
        raise ValueError(f"sequence length mismatch: {len(a)} != {len(b)}")
    yield from zip(a, b)



class Words_OfflineTest(TestCase):

    def setUp(self):
        if SKIP_ALL:
            self.skipTest("SKIP_ALL=True")

    def test_fetch_all_small(self):
        dataset = SmallGinsberg().load_dataset()
        w = Words_Offline(dataset)
        self._test_all_solution(w, "cs-test-1.json")

    def _test_all_solution(self, w: Words_Offline, filename: str):
        candidate_dict = w.all_solution(build_clue_dict())
        expected_file = dlfp_tests.tools.get_testdata_dir() / filename
        # serialize(candidate_dict, expected_file)
        expected = json.loads(expected_file.read_text())
        self._compare(expected, candidate_dict)

    def _compare(self, a: SuggestionDict, b: SuggestionDict):
        for a_clue, b_clue in zip_safe(sorted(a.keys()), sorted(b.keys())):
            a_suggs, b_suggs = a[a_clue], b[b_clue]
            with self.subTest():
                for a_sugg, b_sugg in zip_safe(a_suggs, b_suggs):
                    self.assertAlmostEqual(a_sugg[0], b_sugg[0], delta=1e-6, msg=f"{a_sugg} != {b_sugg}")
                    self.assertEqual(a_sugg[1], b_sugg[1])
