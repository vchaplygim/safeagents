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
import asyncio
import time

from conftest import bash_request

from safeagents import Config, Guard


class SlowScanner:
    """A stand-in for atif-scan: `signals` is synchronous and blocks (CPU-bound)."""

    def __init__(self, delay: float) -> None:
        self.delay = delay

    def signals(self, steps, calls):
        time.sleep(self.delay)
        return []


def test_scan_does_not_stall_the_event_loop():
    delay = 0.3
    reviews = 4

    async def one() -> bool:
        guard = Guard(bash_request("hi"), Config(scanner=SlowScanner(delay)))
        return await guard.allow("bash", '{"command": "ls"}')

    async def go() -> tuple[float, list[bool]]:
        start = time.perf_counter()
        results = await asyncio.gather(*[one() for _ in range(reviews)])
        return time.perf_counter() - start, results

    elapsed, results = asyncio.run(go())
    assert all(results)  # no findings, so every call is allowed
    # Run off the loop, the blocking scans overlap in worker threads; left on the loop
    # they would serialise to reviews * delay. A generous bound keeps this non-flaky.
    assert elapsed < reviews * delay * 0.6
