"""Tests for scripts/asdb_timeline.py and scripts/asdb_rollup.py.

Builds a tiny synthetic `.asdb` (two kind-409 rows, one kind-408 contexts row)
and checks the timeline buckets and rollup groupings.
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

import contextlib
import io
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
import asdb_rollup
import asdb_timeline


def _varint(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _tag(field, wire):
    return _varint((field << 3) | wire)


def _fv(field, value):
    return _tag(field, 0) + _varint(value)


def _fb(field, data):
    return _tag(field, 2) + _varint(len(data)) + data


def _contexts():
    cls1 = _fv(1, 1) + _fb(2, b'Lcom/example/Foo;')
    cls2 = _fv(1, 2) + _fb(2, b'[B')
    method1 = _fv(1, 1) + _fb(2, b'Lcom/example/Foo;') + _fb(3, b'run') + _fb(4, b'Foo.kt') + _fv(5, 10)
    method2 = _fv(1, 2) + _fb(2, b'Ljava/lang/String;') + _fb(3, b'bar') + _fb(4, b'String.java') + _fv(5, 20)
    frame1 = _fv(1, 1) + _fv(2, 10)
    frame2 = _fv(1, 2) + _fv(2, 20)
    stack1 = _fv(1, 1) + _fb(3, _fb(1, frame1))
    stack2 = _fv(1, 2) + _fb(3, _fb(1, frame2))
    thread = _fv(2, 1) + _fb(3, b'main')
    batch = (_fv(1, 0) + _fb(2, cls1) + _fb(2, cls2) + _fb(3, method1) + _fb(3, method2) +
             _fb(4, stack1) + _fb(4, stack2) + _fb(5, thread))
    return _fb(408, _fb(1, batch))


def _event(class_id, size, stack_id, timestamp):
    alloc = (_fv(2, class_id) + _fv(3, size) + _fv(4, 0) + _fv(5, 1) +
             _fv(6, stack_id) + _fb(10, b'main'))
    batch = _fv(1, timestamp) + _fb(2, _fb(4, alloc))
    return _fb(409, _fb(1, batch))


def _make_db(path):
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE UnifiedEventsTable (StreamId INTEGER, ProcessId INTEGER, '
               'GroupId INTEGER, Kind INTEGER, CommandId INTEGER, Timestamp INTEGER, Data BLOB)')
    rows = [
        (408, 0, _contexts()),
        (409, 0, _event(1, 16, 1, 0)),
        (409, 0, _event(2, 8, 2, 0)),
        (409, 1_500_000_000, _event(1, 16, 1, 1_500_000_000)),
    ]
    for kind, ts, blob in rows:
        db.execute('INSERT INTO UnifiedEventsTable VALUES (1, 1, 0, ?, 0, ?, ?)',
                   (kind, ts, blob))
    db.commit()
    db.close()


def _run(module, argv):
    args = module.parse_args(argv)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        module.run(args)
    return out.getvalue()


class TimelineTest(unittest.TestCase):
    def test_buckets_and_needles(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'c.asdb')
            _make_db(path)
            text = _run(asdb_timeline, ['--db', path, '--bucket-ms', '1000',
                                        '--needle', 'Foo'])
        self.assertIn('3 allocations', text)
        lines = [ln for ln in text.splitlines() if ln.strip().startswith('0 ') or
                 ln.strip().startswith('1000 ')]
        self.assertTrue(any(ln.split()[1:2] == ['2'] for ln in lines))
        self.assertTrue(any(ln.split()[1:2] == ['1'] for ln in lines))
        self.assertIn('NEEDLE COUNTS', text)


class RollupTest(unittest.TestCase):
    def test_package_of(self):
        self.assertEqual(asdb_rollup.package_of('Ljava/lang/String;', 3), 'java/lang')
        self.assertEqual(asdb_rollup.package_of('Lcom/example/Foo;', 3), 'com/example')
        self.assertEqual(asdb_rollup.package_of('[B', 3), 'arrays')
        self.assertEqual(asdb_rollup.package_of('Foo', 3), '(default)')

    def test_group_by_package(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'c.asdb')
            _make_db(path)
            text = _run(asdb_rollup, ['--db', path, '--by', 'package'])
        self.assertIn('com/example', text)
        self.assertIn('arrays', text)

    def test_group_by_owner(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'c.asdb')
            _make_db(path)
            text = _run(asdb_rollup, ['--db', path, '--by', 'owner', '--app', 'com/example'])
        self.assertIn('Foo;.run(Foo.kt:10)', text)


if __name__ == '__main__':
    unittest.main()
