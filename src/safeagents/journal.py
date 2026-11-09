# Copyright 2026-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Keeps every tool call the guard didn't simply pass, with its full arguments, so it
can be inspected later with `safeagents inspect`.

Arguments can hold secrets, so the file is readable by its owner only.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path

from .constants import DEFAULT_JOURNAL_PATH

# File locking has no portable standard-library API: Windows has msvcrt.locking,
# POSIX has fcntl.flock. Each module exists only on its own platform (see runs.py).
if os.name == "nt":
    import msvcrt
else:
    import fcntl


def read(path: Path = DEFAULT_JOURNAL_PATH) -> list[dict]:
    """The recorded calls, oldest first; unreadable lines are skipped."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return []
    entries = []
    for line in lines:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and isinstance(entry.get("id"), int):
            entries.append(entry)
    return entries


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Holds an exclusive lock on a sibling lock file for the critical section.

    Numbering and the append must be atomic across processes: every background proxy
    writes to the same journal by default, so two of them picking the next id from the
    file at the same time would otherwise hand out the same number.
    """
    lock_path = path.with_name(path.name + ".lock")
    file_descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        if os.name == "nt":
            msvcrt.locking(file_descriptor, msvcrt.LK_LOCK, 1)
        else:
            fcntl.flock(file_descriptor, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                msvcrt.locking(file_descriptor, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(file_descriptor, fcntl.LOCK_UN)
    finally:
        os.close(file_descriptor)


class Journal:
    """Appends calls to a JSON-lines file, numbering them after those already in it."""

    def __init__(self, path: Path = DEFAULT_JOURNAL_PATH) -> None:
        self.path = Path(path)
        # How far into the file this process has read, and the largest id seen there.
        # Ids are still taken from the file under the lock (another proxy may have
        # appended since), but only the bytes added since the last append are read.
        self._offset = 0
        self._max_id = 0

    def _catch_up(self) -> None:
        """Reads the lines appended since the last call; the whole file if it shrank."""
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            size = 0
        if size < self._offset:  # truncated or rotated: start over
            self._offset = self._max_id = 0
        if size == self._offset:
            return
        with open(self.path, "rb") as file:
            file.seek(self._offset)
            data = file.read(size - self._offset)
        complete = data.rfind(b"\n") + 1  # a torn last line is read again next time
        for raw in data[:complete].splitlines():
            try:
                entry = json.loads(raw)
            except ValueError:
                continue
            if isinstance(entry, dict) and isinstance(entry.get("id"), int):
                self._max_id = max(self._max_id, entry["id"])
        self._offset += complete

    def append(self, **entry: object) -> int:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # The next id is taken from the file under the lock, not from a counter that
        # goes stale the moment another process (or Journal) appends to the same file.
        with _locked(self.path):
            self._catch_up()
            call_id = self._max_id + 1
            line = json.dumps({"id": call_id, "time": time.time(), **entry}, ensure_ascii=False)
            file_descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(file_descriptor, "a+b") as file:
                # A writer that died mid-line leaves no newline; end its line first, or
                # this entry would merge into it, fail to parse, and its id be reused.
                end = file.seek(0, os.SEEK_END)
                if end:
                    file.seek(end - 1)
                    if file.read(1) != b"\n":
                        file.write(b"\n")  # O_APPEND: lands at the end whatever the position
                file.write((line + "\n").encode("utf-8"))
            self._max_id = call_id
        return call_id
