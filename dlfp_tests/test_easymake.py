#!/usr/bin/env python3

from unittest import TestCase

from dlfp.datasets import Tokenized
from dlfp.easymake import PredicateSet


class PredicateSetTest(TestCase):

    def test_regex_match(self):
        p = PredicateSet(require_regex_match=r'^[a-z ]+$')
        good = {"foo bar", "blah", "a day of rest"}
        bad = {"21jumpstreet", "tea for 2", "=to", "movin'onup", "rachel\"", "b&orailroad"}
        for g in good:
            with self.subTest():
                self.assertTrue(p.regex_match(Tokenized(g, [])))
        for b in bad:
            with self.subTest():
                self.assertFalse(p.regex_match(Tokenized(b, [])))
