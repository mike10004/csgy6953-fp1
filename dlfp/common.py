#!/usr/bin/env python3

import sys
import csv
import json
import contextlib
from datetime import datetime
from typing import ContextManager
from typing import TextIO
from typing import Any
from typing import Union
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import TypeVar
from pathlib import Path

import tabulate
T = TypeVar("T")
Pathish = Union[Path, str]


# noinspection PyUnusedLocal
def noop(*args, **kwargs):
    pass

def identity(x: T) -> T:
    return x


def get_repo_root() -> Path:
    return Path(__file__).absolute().parent.parent


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


@contextlib.contextmanager
def open_write(pathname: Optional[Pathish], **kwargs) -> ContextManager[TextIO]:
    if pathname:
        if not "mode" in kwargs:
            kwargs["mode"] = "w"
        with open(pathname, **kwargs) as ofile:
            yield ofile
    else:
        yield sys.stdout


class Table(NamedTuple):

    rows: Sequence[Sequence[Any]]
    headers: Sequence[str] = None

    def write_file(self, pathname: Optional[Pathish], fmt: str = None):
        with open_write(pathname) as ofile:
            self.write(ofile, fmt)

    def write(self, sink: Optional[TextIO] = None, fmt: str = None, **kwargs):
        fmt = fmt or "github"
        sink = sink or sys.stdout
        if fmt == "csv":
            csv_writer = csv.writer(sink, **kwargs)
            if self.headers:
                csv_writer.writerow(self.headers)
            csv_writer.writerows(self.rows)
        if fmt == "json":
            print("[", file=sink)
            for i, row in enumerate(self.rows):
                if i > 0:
                    print(",")
                print("  ", end="")
                json.dump(row, sink, **kwargs)
            print("]", file=sink)
        else:
            if not "headers" in kwargs:
                kwargs["headers"] = self.headers
            if not "tablefmt" in kwargs:
                kwargs["tablefmt"] = fmt
            content = tabulate.tabulate(self.rows, **kwargs)
            print(content, file=sink)

