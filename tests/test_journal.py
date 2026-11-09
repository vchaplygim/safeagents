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
import multiprocessing
from pathlib import Path

from safeagents.journal import Journal, read


def test_single_journal_numbers_sequentially(tmp_path):
    journal = Journal(tmp_path / "calls.jsonl")
    assert [journal.append(verdict="blocked") for _ in range(3)] == [1, 2, 3]


def test_interleaved_journals_do_not_reuse_ids(tmp_path):
    # Two Journal objects on one file stand in for two proxies writing the default
    # journal. A cached counter made both pick the same next id (1, 2, 2, 3).
    path = tmp_path / "calls.jsonl"
    a, b = Journal(path), Journal(path)
    ids = [a.append(verdict="x"), b.append(verdict="y"), a.append(verdict="z"), b.append(verdict="w")]
    assert ids == [1, 2, 3, 4]
    assert sorted(entry["id"] for entry in read(path)) == [1, 2, 3, 4]


def _hammer(path: str, count: int) -> None:
    journal = Journal(Path(path))
    for _ in range(count):
        journal.append(verdict="blocked")


def test_concurrent_processes_assign_unique_ids(tmp_path):
    path = tmp_path / "calls.jsonl"
    workers, per_worker = 4, 25
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_hammer, args=(str(path), per_worker)) for _ in range(workers)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    ids = sorted(entry["id"] for entry in read(path))
    # Every append got a distinct id and the range is contiguous: no collisions, no gaps.
    assert ids == list(range(1, workers * per_worker + 1))
