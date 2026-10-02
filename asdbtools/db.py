"""Read-only access to the .asdb SQLite database.

An `.asdb` capture is a SQLite database with (at least) a
``UnifiedEventsTable(StreamId, ProcessId, GroupId, Kind, CommandId, Timestamp,
Data)`` table. ``Data`` holds a serialized ``Common.Event``; the event payload
for a given kind lives in the ``oneof`` field numbered like the kind itself
(e.g. field 408 carries ``MEMORY_ALLOC_CONTEXTS``).
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

import sqlite3


def open_db(path):
    return sqlite3.connect(f'file:{path}?mode=ro', uri=True)


def kind_blobs(db, kind):
    return [r[0] for r in db.execute(
        'SELECT Data FROM UnifiedEventsTable WHERE Kind=?', (kind,))]


def rowid_span(db, kind):
    """(min_rowid, max_rowid, row_count) for one event kind, or None."""
    row = db.execute(
        'SELECT MIN(rowid), MAX(rowid), COUNT(*) FROM UnifiedEventsTable WHERE Kind=?',
        (kind,)).fetchone()
    if not row or not row[2]:
        return None
    return row


def chunk_blobs(db_path, kind, lo, hi):
    """Fetch the Data blobs of one [lo, hi) rowid span (fresh connection)."""
    db = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    try:
        return [row[0] for row in db.execute(
            'SELECT Data FROM UnifiedEventsTable WHERE Kind=? AND rowid >= ? AND rowid < ?',
            (kind, lo, hi))]
    finally:
        db.close()


def get_chunks(db_path, kind, n):
    """Split one event kind's rowid range into [lo, hi) spans."""
    span = rowid_span(open_db(db_path), kind)
    # NOTE: open_db connection intentionally left to GC; it is read-only and
    # short-lived here.
    if span is None:
        return []
    lo, hi, _ = span
    width = max(1, (hi - lo + 1 + n - 1) // n)
    chunks = []
    start = lo
    while start <= hi:
        chunks.append((start, min(start + width, hi + 1)))
        start += width
    return chunks
