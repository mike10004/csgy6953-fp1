#!/usr/bin/env python3

import io
import json
import contextlib
from json import JSONDecodeError
from typing import NamedTuple
from typing import Optional
from unittest import TestCase

import dlfp.common
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


class ModuleMethodsTest(TestCase):

    def test_namedtuple_diff(self):
        class N(NamedTuple):

            w: str = "foo"
            x: int = 5
            y: float = 0.001
            z: Optional[str] = None

        q = N(y=0.0005, z="bar")
        d = dlfp.common.namedtuple_diff(N(), q)
        self.assertDictEqual({
            'y': q.y,
            'z': q.z,
        }, d)