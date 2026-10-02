"""Parallel event-scan runner.

The 409 (allocation event) table holds millions of rows; decoding them is pure
CPU work that a single Python process cannot parallelize (GIL). This module
shards the table by rowid and runs one worker per shard in forked processes.
Each worker opens its own read-only SQLite connection and reuses the parent's
already-decoded 408 context tables through fork inheritance, so no large
payload is pickled: only ``(db_path, kind, lo, hi)`` tuples cross the process
boundary, and small per-shard results come back.

Set ``WORKERS=1`` (or leave a single rowid chunk) for the serial path, which
runs through the same worker function.
"""
#     Copyright 2026 András Oravecz <info@oandras.hu>
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import os

from asdbtools.db import get_chunks


def default_workers():
    try:
        explicit = int(os.environ.get('WORKERS', '0'))
        if explicit > 0:
            return explicit
    except ValueError:
        pass
    return os.cpu_count() or 4


def run(db_path, kind, n, worker):
    """Run worker(lo, hi) over the rowid chunks; return the list of results."""
    import multiprocessing as mp
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    workers = n or default_workers()
    chunks = get_chunks(db_path, kind, workers)
    if len(chunks) <= 1:
        return [worker(*chunks[0])] if chunks else []
    with mp.Pool(min(len(chunks), workers)) as pool:
        return pool.starmap(worker, chunks)
