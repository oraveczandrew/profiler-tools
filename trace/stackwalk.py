"""Parallel per-thread stack walks over parsed ART traces.

The event loop of a stack walk is pure serial Python (stateful per thread),
so threads cannot help (GIL) — but threads of a trace are independent of each
other, so each thread's walk runs in its own forked process. Callers get back
merged inclusive/exclusive/call counters plus per-thread busy rows.

The walk semantics are exactly trace_analyze.py's: empty-stack exits are
counted and skipped, mismatched exits are attributed to the popped frame,
negative durations are clamped and counted. (trace_deep.py's thread section had
no empty-stack guard and crashed on unbalanced traces; it now shares this
walker, which is identical wherever the old code did not crash.)
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

import numpy as np


def default_workers():
    try:
        explicit = int(os.environ.get('WORKERS', '0'))
        if explicit > 0:
            return explicit
    except ValueError:
        pass
    return os.cpu_count() or 4


def walk_thread(ii, aa, ss, n_methods):
    """Walk ONE thread's (method, action, timestamp) events in order.

    Returns (incl, excl, calls, err_empty, err_mismatch, err_back, busy, nev)
    where incl/excl/calls are length-n_methods int64 arrays and busy is the
    thread's total exclusive time (== excl.sum()).
    """
    incl = np.zeros(n_methods, dtype=np.int64)
    excl = np.zeros(n_methods, dtype=np.int64)
    calls = np.zeros(n_methods, dtype=np.int64)
    err_empty = 0
    err_mismatch = 0
    err_back = 0
    stack = []  # [method_idx, enter_ts, child_sum]
    for method, a, s in zip(ii.tolist(), aa.tolist(), ss.tolist()):
        if a == 0:
            stack.append([method, s, 0])
            calls[method] += 1
        else:
            if not stack:
                err_empty += 1
                continue
            pm, es, ch = stack.pop()
            if pm != method:
                err_mismatch += 1
                # attribute to the popped frame anyway (unwind accounting)
                method = pm
            dur = s - es
            if dur < 0:
                err_back += 1
                dur = 0
            incl[method] += dur
            excl[method] += dur - ch
            if stack:
                stack[-1][2] += dur
    return incl, excl, calls, err_empty, err_mismatch, err_back, int(excl.sum()), len(ii)


def _walk_one(job):
    t, ii, aa, ss, n_methods = job
    incl, excl, calls, err_empty, err_mismatch, err_back, busy, nev = walk_thread(
        ii, aa, ss, n_methods)
    return t, incl, excl, calls, err_empty, err_mismatch, err_back, busy, nev


def walk_all(tid, idx, act, ts, n_methods, tids=None, workers=0):
    """Walk every requested thread in parallel; merge the results.

    tids defaults to all threads present in tid. Returns a dict with merged
    incl/excl/calls arrays, summed error counts, total events, and per-thread
    (busy, events, tid) rows.
    """
    import multiprocessing as mp
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    if tids is None:
        tids = [int(t) for t in np.unique(tid)]
    jobs = []
    for t in tids:
        m = tid == t
        jobs.append((t, idx[m], act[m], ts[m], n_methods))
    n_workers = workers or default_workers()
    if len(jobs) <= 1 or n_workers <= 1:
        parts = [_walk_one(job) for job in jobs]
    else:
        with mp.Pool(min(len(jobs), n_workers)) as pool:
            parts = pool.map(_walk_one, jobs)
    incl = np.zeros(n_methods, dtype=np.int64)
    excl = np.zeros(n_methods, dtype=np.int64)
    calls = np.zeros(n_methods, dtype=np.int64)
    err_empty = err_mismatch = err_back = 0
    rows = []
    for t, pi, pe, pc, ee, em, eb, busy, nev in parts:
        incl += pi
        excl += pe
        calls += pc
        err_empty += ee
        err_mismatch += em
        err_back += eb
        rows.append((busy, nev, t))
    return {
        'incl': incl, 'excl': excl, 'calls': calls,
        'err_empty': err_empty, 'err_mismatch': err_mismatch, 'err_back': err_back,
        'rows': rows, 'events': sum(r[1] for r in rows),
    }
