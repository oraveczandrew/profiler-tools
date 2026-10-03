"""Self-tests for scripts/hprof_hist.py (parallel heap-span walk).

Builds a minimal hand-crafted HPROF fixture (no capture needed), runs the
CLI serially vs. sharded, and asserts exact output equality plus the known
per-class histogram. Also checks walk_span_range merge equivalence at the
dict level (exact, not spot checks).
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

import importlib.util
import os
import struct
import subprocess
import sys
import unittest

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts')
HPROF_PATH = os.path.join(REPO, 'tmp', 'hprof-hist-test.hprof')


def _load_module():
    spec = importlib.util.spec_from_file_location(
        'hprof_hist', os.path.join(SCRIPTS_DIR, 'hprof_hist.py'))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rec(tag, body):
    return struct.pack('>BII', tag, 0, len(body)) + body


def sid(v):
    return struct.pack('>I', v)


def class_dump(cid, isize):
    return (b'\x20' + sid(cid) + struct.pack('>I', 0) + sid(0) * 4 + sid(0) * 2
            + struct.pack('>I', isize) + struct.pack('>HHH', 0, 0, 0))


def inst(obj, cid, size):
    return b'\x21' + sid(obj) + struct.pack('>I', 0) + sid(cid) \
        + struct.pack('>I', size) + bytes(size)


def prim(obj, cnt, etype):
    return b'\x23' + sid(obj) + struct.pack('>I', 0) + struct.pack('>I', cnt) \
        + bytes([etype]) + bytes(cnt * 4)


def objarr(obj, cid, cnt):
    return b'\x22' + sid(obj) + struct.pack('>I', 0) + struct.pack('>I', cnt) \
        + sid(cid) + sid(0) * cnt


def build_hprof():
    blob = bytearray()
    blob += b'JAVA PROFILE 1.0.3\x00'
    blob += struct.pack('>I', 4)  # id_size
    blob += struct.pack('>Q', 0)  # timestamp
    blob += rec(0x01, sid(1) + b'Lcom/ex/Foo;')
    blob += rec(0x01, sid(2) + b'Lcom/ex/Bar;')
    blob += rec(0x02, struct.pack('>I', 1) + sid(100) + struct.pack('>I', 0) + sid(1))
    blob += rec(0x02, struct.pack('>I', 2) + sid(200) + struct.pack('>I', 0) + sid(2))
    seg1 = class_dump(100, 8) + inst(1001, 100, 8) + inst(1002, 100, 8) + prim(2001, 4, 10)
    seg2 = class_dump(200, 4) + inst(2002, 200, 4) + objarr(3001, 200, 3)
    blob += rec(0x0C, seg1)
    blob += rec(0x0C, seg2)
    return bytes(blob)


def run_hist(*args):
    proc = subprocess.run(
        [sys.executable, os.path.join(SCRIPTS_DIR, 'hprof_hist.py'),
         '--db', HPROF_PATH, *args],
        cwd=REPO, capture_output=True, text=True, timeout=600)
    return proc


class HprofHistTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(HPROF_PATH, 'wb') as f:
            f.write(build_hprof())
        cls.addClassCleanup(cls._cleanup)

    @classmethod
    def _cleanup(cls):
        try:
            os.remove(HPROF_PATH)
        except OSError:
            pass

    def test_known_histogram(self):
        proc = run_hist('--workers', '1')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = proc.stdout
        self.assertIn('top-level heap dumps: 2, heap bytes counted: 48', out)
        foo = [l for l in out.splitlines() if l.rstrip().endswith('Lcom.ex.Foo;')]
        bar = [l for l in out.splitlines() if l.rstrip().endswith('Lcom.ex.Bar;')]
        prim = [l for l in out.splitlines() if 'primitive-array[-10]' in l]
        self.assertEqual(len(foo), 1)
        self.assertTrue(foo[0].split()[0] == '2' and foo[0].split()[1] == '16')
        self.assertEqual(len(bar), 1)
        self.assertTrue(bar[0].split()[0] == '2' and bar[0].split()[1] == '16')
        self.assertEqual(len(prim), 1)
        self.assertTrue(prim[0].split()[0] == '1' and prim[0].split()[1] == '16')

    def test_serial_vs_parallel_stdout_identical(self):
        serial = run_hist('--workers', '1')
        self.assertEqual(serial.returncode, 0, serial.stderr)
        for w in ('2', '8'):
            parallel = run_hist('--workers', w)
            self.assertEqual(parallel.returncode, 0, parallel.stderr)
            self.assertEqual(parallel.stdout, serial.stdout)

    def test_dict_merge_exact(self):
        mod = _load_module()
        with open(HPROF_PATH, 'rb') as f:
            data = f.read()
        strings, load_classes, spans, n_heap = mod.scan_top_level(data, 4)
        self.assertEqual(n_heap, 2)
        full, full_total, full_skipped = mod.walk_span_range(data, spans, 4, 0, len(spans))
        half = len(spans) // 2
        left, lt, ls = mod.walk_span_range(data, spans, 4, 0, half)
        right, rt, rs = mod.walk_span_range(data, spans, 4, half, len(spans))
        merged = {}
        for part in (left, right):
            for cid in sorted(part):
                cnt, b = part[cid]
                e = merged.get(cid)
                if e is None:
                    merged[cid] = [cnt, b]
                else:
                    e[0] += cnt
                    e[1] += b
        self.assertEqual(merged, full)
        self.assertEqual(lt + rt, full_total)
        self.assertEqual(ls + rs, full_skipped)


if __name__ == '__main__':
    unittest.main()
