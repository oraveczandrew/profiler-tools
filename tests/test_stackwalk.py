"""Self-tests for trace/stackwalk.py. No capture needed."""
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
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trace'))

from stackwalk import walk_all, walk_thread


def ev(methods, acts, times):
    return (np.array(methods, dtype=np.int64),
            np.array(acts, dtype=np.int64),
            np.array(times, dtype=np.int64))


class WalkThreadTest(unittest.TestCase):
    def test_nested(self):
        # main calls a(0-100), a calls b(10-40): a excl=70, b incl=excl=30
        ii, aa, ss = ev([0, 1, 1, 0], [0, 0, 1, 1], [0, 10, 40, 100])
        incl, excl, calls, ee, em, eb, busy, nev = walk_thread(ii, aa, ss, 2)
        self.assertEqual(list(incl), [100, 30])
        self.assertEqual(list(excl), [70, 30])
        self.assertEqual(list(calls), [1, 1])
        self.assertEqual((ee, em, eb), (0, 0, 0))
        self.assertEqual(busy, 100)
        self.assertEqual(nev, 4)

    def test_empty_exit_counted_not_crashing(self):
        ii, aa, ss = ev([3, 0, 0], [1, 0, 1], [5, 0, 10])
        incl, excl, calls, ee, em, eb, busy, nev = walk_thread(ii, aa, ss, 4)
        self.assertEqual(ee, 1)
        self.assertEqual(list(incl), [10, 0, 0, 0])
        self.assertEqual(list(excl), [10, 0, 0, 0])
        self.assertEqual(list(calls), [1, 0, 0, 0])
        self.assertEqual(busy, 10)

    def test_mismatch_attributed_to_popped(self):
        ii, aa, ss = ev([0, 1], [0, 2], [0, 50])
        incl, excl, calls, ee, em, eb, busy, nev = walk_thread(ii, aa, ss, 2)
        self.assertEqual(em, 1)
        self.assertEqual(list(incl), [50, 0])
        self.assertEqual(busy, 50)

    def test_backward_clamped(self):
        ii, aa, ss = ev([0, 0], [0, 1], [100, 40])
        incl, excl, calls, ee, em, eb, busy, nev = walk_thread(ii, aa, ss, 1)
        self.assertEqual(eb, 1)
        self.assertEqual(list(incl), [0])
        self.assertEqual(busy, 0)


class WalkAllTest(unittest.TestCase):
    def _two_threads(self):
        # thread 1: a(0-100) with b(10-40); thread 2: c(0-50)
        tid = np.array([1, 1, 1, 1, 2, 2], dtype=np.int64)
        idx = np.array([0, 1, 1, 0, 2, 2], dtype=np.int64)
        act = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
        ts = np.array([0, 10, 40, 100, 0, 50], dtype=np.int64)
        return tid, idx, act, ts

    def test_serial(self):
        tid, idx, act, ts = self._two_threads()
        r = walk_all(tid, idx, act, ts, 3, workers=1)
        self.assertEqual(list(r['incl']), [100, 30, 50])
        self.assertEqual(list(r['excl']), [70, 30, 50])
        self.assertEqual(list(r['calls']), [1, 1, 1])
        self.assertEqual((r['err_empty'], r['err_mismatch'], r['err_back']), (0, 0, 0))
        self.assertEqual(r['events'], 6)
        rows = sorted(r['rows'])
        self.assertEqual(rows, [(50, 2, 2), (100, 4, 1)])

    def test_parallel_matches_serial(self):
        tid, idx, act, ts = self._two_threads()
        serial = walk_all(tid, idx, act, ts, 3, workers=1)
        parallel = walk_all(tid, idx, act, ts, 3, workers=4)
        self.assertEqual(list(serial['incl']), list(parallel['incl']))
        self.assertEqual(list(serial['excl']), list(parallel['excl']))
        self.assertEqual(list(serial['calls']), list(parallel['calls']))
        self.assertEqual(sorted(serial['rows']), sorted(parallel['rows']))
        self.assertEqual(
            (serial['err_empty'], serial['err_mismatch'], serial['err_back']),
            (parallel['err_empty'], parallel['err_mismatch'], parallel['err_back']))


if __name__ == '__main__':
    unittest.main()
