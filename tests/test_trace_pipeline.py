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
import tempfile
import unittest

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trace')
# Hermetic fixture dir (auto-removed on exit); the old tests/../../../tmp
# layout only worked by accident of the local checkout path.
_TMP = tempfile.TemporaryDirectory(prefix='synth-trace-test-')
TAG = 'synth-trace-test'
TRACE_PATH = os.path.join(_TMP.name, TAG + '.trace')


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


def run_script_on(trace_path, name, *args):
    env = dict(os.environ, WORKERS='4')
    proc = subprocess.run(
        [sys.executable, os.path.join(TRACE_DIR, name), trace_path, *args],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    return proc


TAG2 = 'synth-trace-sharded'
TRACE_PATH2 = os.path.join(_TMP.name, TAG2 + '.trace')


def build_multigap_trace():
    """Same call tree as build_trace but with method records interleaved
    between data records, so the data spans ≥2 gap regions (the sharded
    parser must concatenate them back in file order)."""
    blob = bytearray()
    blob += b'SLOW' + b'\x00' * 28
    blob += thread_rec(1, 'main')
    blob += thread_rec(2, 'worker')
    blob += method_rec(0, 'com.ex.A', 'a', '()V', 'A.java')
    blob += data_rec(1, 0, 0, 0)
    blob += method_rec(1, 'com.ex.A', 'b', '()V', 'A.java')
    blob += method_rec(2, 'com.ex.B', 'c', '()V', 'B.java')
    blob += data_rec(1, 1, 0, 10)
    blob += data_rec(2, 2, 0, 5)
    blob += data_rec(1, 1, 1, 40)
    blob += data_rec(2, 2, 1, 50)
    blob += data_rec(1, 0, 1, 100)
    blob += b'*version\nv=1\n*threads\n1 main\n2 worker\n*methods\n*end\n'
    return bytes(blob)


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
                os.remove(os.path.join(_TMP.name, TAG + suffix))
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

    def test_parents(self):
        self.assertEqual(run_script('trace_parse.py').returncode, 0)
        self.assertEqual(run_script('trace_analyze.py').returncode, 0)
        outs = []
        for w in ('1', '4'):
            proc = run_script('trace_parents.py', 'com.ex', '--workers', w)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            outs.append(proc.stdout)
        self.assertIn('matched targets: 3', outs[0])
        self.assertIn('com.ex.A#b()V', outs[0])
        norm = [o.replace('workers=1', 'workers=N').replace('workers=4', 'workers=N')
                for o in outs]
        self.assertEqual(norm[0], norm[1])

    def test_parents_dict_equivalence(self):
        # walk_all_parents merge must be identical for any worker count.
        import importlib.util
        import numpy as np
        spec = importlib.util.spec_from_file_location(
            'trace_parents', os.path.join(TRACE_DIR, 'trace_parents.py'))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules['trace_parents'] = mod  # needed: mp pickles by reference
        spec.loader.exec_module(mod)
        walk_all_parents = mod.walk_all_parents
        self.assertEqual(run_script('trace_parse.py').returncode, 0)
        z = np.load(os.path.splitext(TRACE_PATH)[0] + '.parsed.npz')
        tid, idx, act, ts = z['tid'], z['dex'], z['act'], z['ts']
        targets = frozenset({0, 1, 2})
        serial = walk_all_parents(tid, idx, act, ts, targets, workers=1)
        parallel = walk_all_parents(tid, idx, act, ts, targets, workers=4)
        self.assertEqual(serial['parent'], parallel['parent'])
        self.assertEqual(serial['child'], parallel['child'])
        self.assertEqual(serial['threads'], parallel['threads'])


class ShardedParseTest(unittest.TestCase):
    """Multi-gap fixture: sharded gap-decode must equal serial exactly."""

    @classmethod
    def setUpClass(cls):
        with open(TRACE_PATH2, 'wb') as f:
            f.write(build_multigap_trace())
        cls.addClassCleanup(cls._cleanup)

    @classmethod
    def _cleanup(cls):
        for suffix in ('.trace', '.parsed.npz', '.stats.npz'):
            try:
                os.remove(os.path.join(_TMP.name, TAG2 + suffix))
            except OSError:
                pass

    def _parse_arrays(self, workers):
        import numpy as np
        proc = run_script_on(TRACE_PATH2, 'trace_parse.py', '--workers', workers)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc, dict(np.load(os.path.splitext(TRACE_PATH2)[0] + '.parsed.npz',
                                  allow_pickle=True))

    def test_multiple_gaps(self):
        proc, _ = self._parse_arrays('1')
        line = [l for l in proc.stdout.splitlines() if l.startswith('gap regions:')]
        self.assertEqual(len(line), 1)
        n_gaps = int(line[0].split()[2])
        self.assertGreaterEqual(n_gaps, 2)

    def test_sharded_vs_serial_identical(self):
        import numpy as np
        _, a = self._parse_arrays('1')
        _, b = self._parse_arrays('4')
        self.assertEqual(set(a.keys()), set(b.keys()))
        for k in a:
            self.assertTrue((np.asarray(a[k]) == np.asarray(b[k])).all(), k)


TAG3 = 'synth-trace-cached'
TRACE_PATH3 = os.path.join(_TMP.name, TAG3 + '.trace')
PARSED_PATH3 = os.path.splitext(TRACE_PATH3)[0] + '.parsed.npz'


class CachedTablesTest(unittest.TestCase):
    """trace_parse precomputes names/threads/gaps; downstream scripts reuse
    them instead of re-reading the multi-GB .trace."""

    @classmethod
    def setUpClass(cls):
        with open(TRACE_PATH3, 'wb') as f:
            f.write(build_multigap_trace())
        cls.addClassCleanup(cls._cleanup)

    @classmethod
    def _cleanup(cls):
        for suffix in ('.trace', '.parsed.npz', '.stats.npz'):
            try:
                os.remove(os.path.join(_TMP.name, TAG3 + suffix))
            except OSError:
                pass

    def _parse(self, workers='4'):
        if not os.path.exists(TRACE_PATH3):
            with open(TRACE_PATH3, 'wb') as f:
                f.write(build_multigap_trace())
        proc = run_script_on(TRACE_PATH3, 'trace_parse.py', '--workers', workers)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return proc

    def test_cached_keys_present(self):
        import numpy as np
        self._parse()
        z = dict(np.load(PARSED_PATH3, allow_pickle=True))
        for k in ('tid', 'dex', 'act', 'ts', 'valid',
                  'names', 'thread_ids', 'thread_names', 'gaps'):
            self.assertIn(k, z)
        names = [str(n) for n in z['names']]
        self.assertEqual(names, ['com.ex.A#a()V', 'com.ex.A#b()V', 'com.ex.B#c()V'])
        self.assertEqual(dict(zip([int(t) for t in z['thread_ids']],
                                  [str(n) for n in z['thread_names']])),
                         {1: 'main', 2: 'worker'})
        gaps = [(int(s), int(e)) for s, e in z['gaps'].tolist()]
        self.assertGreaterEqual(len(gaps), 2)
        for s, e in gaps:
            self.assertLess(s, e)

    def test_downstream_without_trace(self):
        # parse, then DELETE the .trace: analyze/deep/parents must still run
        # with identical outputs from the cache alone.
        self._parse()
        ref = {}
        for name, args in (('analyze', ('trace_analyze.py',)),
                           ('deep', ('trace_deep.py',)),
                           ('parents', ('trace_parents.py', 'com.ex'))):
            proc = run_script_on(TRACE_PATH3, *args)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            ref[name] = proc.stdout
        self.assertIn('stack errors: empty=0 mismatch=0 backward=0', ref['analyze'])
        self.assertIn('main tid: 1', ref['deep'])
        self.assertIn('matched targets: 3', ref['parents'])
        os.remove(TRACE_PATH3)
        for name, args in (('analyze', ('trace_analyze.py',)),
                           ('deep', ('trace_deep.py',)),
                           ('parents', ('trace_parents.py', 'com.ex'))):
            proc = run_script_on(TRACE_PATH3, *args)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            norm = lambda o: o.replace('(cached)', '').replace('workers=4', 'workers=N')
            self.assertEqual(norm(proc.stdout), norm(ref[name]), name)


if __name__ == '__main__':
    unittest.main()
