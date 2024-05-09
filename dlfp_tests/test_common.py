#!/usr/bin/env python3

import io
import json
import contextlib
from json import JSONDecodeError
from unittest import TestCase

from dlfp.common import Table

class TableTest(TestCase):

    def test_write(self):
        table = Table([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], ["foo", "bar", "baz"])
        for fmt in ["csv", "json", None, "simple_grid"]:
            with self.subTest(fmt=fmt):
                buffer = io.StringIO()
                table.write(buffer, fmt=fmt)
                text = buffer.getvalue()
                self._check_content(text, noheader=fmt == "json")
                if fmt == "json":
                    try:
                        items = json.loads(text)
                    except JSONDecodeError:
                        self.fail(f"text:\n{text}")
                    self.assertIn([1, 2, 3], items)

    def _check_content(self, text: str, noheader: bool = False):
        self.assertIn("5", text)
        if not noheader:
            self.assertIn("bar", text)

    def test_write_stdout(self):
        table = Table([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ], ["foo", "bar", "baz"])
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            table.write()
        text = buffer.getvalue()
        self._check_content(text)
