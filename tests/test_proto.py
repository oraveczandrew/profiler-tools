"""Self-tests for asdbtools. No capture needed; run from profiler-tools/."""
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
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from asdbtools import fm, get_chunks, ss, sub, sv, walk


def encode_varint(value):
    out = bytearray()
    while True:
        bits = value & 0x7F
        value >>= 7
        if value:
            out.append(bits | 0x80)
        else:
            out.append(bits)
            return bytes(out)


def encode_field(num, wt, payload):
    return encode_varint((num << 3) | wt) + payload


def encode_bytes_field(num, data):
    return encode_field(num, 2, encode_varint(len(data)) + data)


class WalkTest(unittest.TestCase):
    def test_varint_and_nested(self):
        inner = encode_varint((2 << 3) | 0) + encode_varint(150)
        blob = (
            encode_varint((1 << 3) | 0) + encode_varint(300)
            + encode_bytes_field(2, inner)
            + encode_bytes_field(3, b'hi')
        )
        fields = list(walk(blob))
        self.assertEqual(fields[0][:3], (1, 0, 300))
        self.assertEqual(fields[1][0], 2)
        nested = fm(fields[1][2])
        self.assertEqual(nested[2], [150])
        self.assertEqual(fm(blob)[3], [b'hi'])

    def test_helpers(self):
        blob = encode_bytes_field(7, b'ab') + encode_varint((8 << 3) | 0) + encode_varint(9)
        d = fm(blob)
        self.assertEqual(sub(blob, 7), b'ab')
        self.assertEqual(sv(d, 8), 9)
        self.assertEqual(sv(d, 99), 0)
        self.assertEqual(ss(d, 3), '')

    def test_empty_and_garbage(self):
        self.assertEqual(fm(b''), {})
        self.assertEqual(fm(None), {})
        self.assertEqual(sub(None, 1), None)

    def test_32bit_and_64bit(self):
        blob = encode_field(1, 5, b'\x01\x02\x03\x04') + encode_field(2, 1, b'\x00' * 8)
        fields = list(walk(blob))
        self.assertEqual((fields[0][0], fields[0][1], fields[0][4]), (1, 5, 4))
        self.assertEqual((fields[1][0], fields[1][1], fields[1][4]), (2, 1, 8))


class ChunksTest(unittest.TestCase):
    def test_get_chunks_covers_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 't.asdb')
            db = sqlite3.connect(path)
            db.execute('CREATE TABLE UnifiedEventsTable (Kind INTEGER, Data BLOB)')
            db.executemany('INSERT INTO UnifiedEventsTable VALUES (?, ?)',
                           [(409, b'x')] * 100 + [(411, b'y')] * 10)
            db.commit()
            db.close()
            chunks = get_chunks(path, 409, 4)
            self.assertTrue(chunks)
            covered = sum(hi - lo for lo, hi in chunks)
            self.assertEqual(covered, 100)
            self.assertEqual(get_chunks(path, 999, 4), [])
            self.assertEqual(get_chunks(path, 409, 1), [(chunks[0][0], chunks[-1][1])])


if __name__ == '__main__':
    unittest.main()
