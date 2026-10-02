"""End-to-end test of the trace pipeline on a synthetic ART trace.

Generates a tiny cpu-art-*.trace with a known call tree, runs
trace_parse.py -> trace_analyze.py -> trace_deep.py as subprocesses, and
asserts the known inclusive/exclusive numbers. Exercises the parallel
stack walker through the real scripts (WORKERS=4).
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
import struct
import subprocess
import sys
import unittest

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trace')
TAG = 'synth-trace-test'
TRACE_PATH = os.path.join(REPO, 'tmp', TAG + '.trace')


def thread_rec(tid, name):
    name = name.encode()
    # separator: another special record follows
    return b'\x02' + struct.pack('<HH', tid, len(name)) + name + b'\x00\x00'


def method_rec(dex, cls, name, sig, fil):
    return b'\x01\x00\x00' + ('0x%x\t%s\t%s\t%s\t%s\n' % (dex, cls, name, sig, fil)).encode()


def data_rec(tid, dex, act, ts):
    return struct.pack('<HII', tid, (dex << 2) | act, ts)


def build_trace():
    blob = bytearray()
    blob += b'SLOW' + b'\x00' * 28
    blob += thread_rec(1, 'main')
    blob += thread_rec(2, 'worker')
    blob += method_rec(0, 'com.ex.A', 'a', '()V', 'A.java')
    blob += method_rec(1, 'com.ex.A', 'b', '()V', 'A.java')
    blob += method_rec(2, 'com.ex.B', 'c', '()V', 'B.java')
    # thread 1: a(0-100) containing b(10-40)
    blob += data_rec(1, 0, 0, 0)
    blob += data_rec(1, 1, 0, 10)
    blob += data_rec(1, 1, 1, 40)
    blob += data_rec(1, 0, 1, 100)
    # thread 2: c(0-50)
    blob += data_rec(2, 2, 0, 0)
    blob += data_rec(2, 2, 1, 50)
    blob += b'*version\nv=1\n*threads\n1 main\n2 worker\n*methods\n*end\n'
    return bytes(blob)


def run_script(name, *args):
    env = dict(os.environ, WORKERS='4')
    proc = subprocess.run(
        [sys.executable, os.path.join(TRACE_DIR, name), TRACE_PATH, *args],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    return proc


class TracePipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(TRACE_PATH, 'wb') as f:
            f.write(build_trace())
        cls.addClassCleanup(cls._cleanup)

    @classmethod
    def _cleanup(cls):
        for suffix in ('.trace', '.parsed.npz', '.stats.npz'):
            try:
                os.remove(os.path.join(REPO, 'tmp', TAG + suffix))
            except OSError:
                pass

    def test_parse(self):
        proc = run_script('trace_parse.py')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('records: 6', proc.stdout)
        self.assertIn('valid tid frac: 1.0 valid dex frac: 1.0', proc.stdout)

    def test_analyze(self):
        self.assertEqual(run_script('trace_parse.py').returncode, 0)
        proc = run_script('trace_analyze.py')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = proc.stdout
        self.assertIn('stack errors: empty=0 mismatch=0 backward=0', out)
        # inclusive: a=100, c=50, b=30
        a_line = [l for l in out.splitlines() if l.rstrip().endswith('com.ex.A#a()V')]
        c_line = [l for l in out.splitlines() if l.rstrip().endswith('com.ex.B#c()V')]
        b_line = [l for l in out.splitlines() if l.rstrip().endswith('com.ex.A#b()V')]
        self.assertEqual(len(a_line), 2)  # once in each TOP 20 list
        self.assertTrue(a_line[0].startswith('         100'))
        self.assertTrue(c_line[0].startswith('          50'))
        self.assertTrue(b_line[0].startswith('          30'))
        # exclusive: a=70
        self.assertTrue(a_line[1].startswith('          70'))

    def test_deep(self):
        self.assertEqual(run_script('trace_parse.py').returncode, 0)
        self.assertEqual(run_script('trace_analyze.py').returncode, 0)
        proc = run_script('trace_deep.py')
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('main tid: 1', proc.stdout)
        self.assertIn('tid=1  ', proc.stdout)


if __name__ == '__main__':
    unittest.main()
