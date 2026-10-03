"""Parent/child attribution for suspicious frames + timeline.

Walks every thread's events (threads are independent) in forked workers and
merges the attribution tables in the parent. Deterministic: merged counters
are bit-identical to a serial run (sorted merge order).

usage:
    python3 trace/trace_parents.py <cpu-art-*.trace> <frame-substring> [...] [--workers N]

Workers default to $WORKERS or CPU count; 1 = serial path through the same
code. Unix only (fork).
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
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from stackwalk import default_workers, require_cached_tables


def match_targets(names, probes):
    """Map method idx -> full name for every name containing a probe."""
    targets = {}
    for i, n in enumerate(names):
        for t in probes:
            if t in n and i not in targets:
                targets[i] = n
    return targets


def walk_parents(ii, aa, ss, target_set):
    """Walk ONE thread's events; attribute parent/child edges for targets.

    Returns (parent_time, child_time) where both map (parent, child) ->
    [total_dur, count]. child_time only holds edges whose parent is a target.
    Durations clamp negative deltas to 0 (trace cut / clock skew).
    """
    parent_time = {}
    child_time = {}
    stack = []  # [method_idx, enter_ts, {child_idx: [dur, count]}]
    for method, a, s in zip(ii.tolist(), aa.tolist(), ss.tolist()):
        if a == 0:
            stack.append([method, s, {}])
        elif stack:
            pm, es, ch = stack.pop()
            if pm != method:
                method = pm
            dur = s - es
            if dur < 0:
                dur = 0
            if stack:
                p = stack[-1][0]
                key = (p, method)
                pt = parent_time.get(key)
                if pt is None:
                    parent_time[key] = [dur, 1]
                else:
                    pt[0] += dur
                    pt[1] += 1
                d = stack[-1][2].get(method)
                if d is None:
                    stack[-1][2][method] = [dur, 1]
                else:
                    d[0] += dur
                    d[1] += 1
            if method in target_set:
                for c, (d, cc) in ch.items():
                    key = (method, c)
                    g = child_time.get(key)
                    if g is None:
                        child_time[key] = [d, cc]
                    else:
                        g[0] += d
                        g[1] += cc
    return parent_time, child_time


def _parents_one(job):
    t, ii, aa, ss, target_tuple = job
    target_set = frozenset(target_tuple)
    par, chi = walk_parents(ii, aa, ss, target_set)
    return t, par, chi


def walk_all_parents(tid, idx, act, ts, target_set, tids=None, workers=0):
    """Walk every requested thread in parallel; merge attribution tables.

    tids defaults to all threads present in tid. Returns
    {'parent': {(p,c): [dur,count]}, 'child': {(p,c): [dur,count]},
     'threads': sorted list of walked tids}.
    Merge order is sorted by tid, so results are identical for any
    worker count.
    """
    if os.name != 'posix':
        sys.exit('trace_parents.py parallel walk needs fork (Unix only)')
    import multiprocessing as mp
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    if tids is None:
        tids = sorted(int(t) for t in np.unique(tid))
    else:
        tids = sorted(int(t) for t in tids)
    target_tuple = tuple(sorted(target_set))
    jobs = []
    for t in tids:
        m = tid == t
        jobs.append((t, idx[m], act[m], ts[m], target_tuple))
    n_workers = workers or default_workers()
    if len(jobs) <= 1 or n_workers <= 1:
        parts = [_parents_one(job) for job in jobs]
    else:
        with mp.Pool(min(len(jobs), n_workers)) as pool:
            parts = pool.map(_parents_one, jobs)
    parts.sort(key=lambda p: p[0])
    parent = {}
    child = {}
    for _, par, chi in parts:
        for k in sorted(par):
            d, c = par[k]
            g = parent.get(k)
            if g is None:
                parent[k] = [d, c]
            else:
                g[0] += d
                g[1] += c
        for k in sorted(chi):
            d, c = chi[k]
            g = child.get(k)
            if g is None:
                child[k] = [d, c]
            else:
                g[0] += d
                g[1] += c
    return {'parent': parent, 'child': child, 'threads': tids}


def parse_argv(argv):
    """Split argv into (trace_path, probes, workers)."""
    workers = 0
    rest = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--workers' and i + 1 < len(argv):
            try:
                workers = int(argv[i + 1])
            except ValueError:
                sys.exit('--workers needs an integer')
            i += 2
        elif a.startswith('--workers='):
            try:
                workers = int(a.split('=', 1)[1])
            except ValueError:
                sys.exit('--workers needs an integer')
            i += 1
        else:
            rest.append(a)
            i += 1
    traces = [a for a in rest if a.endswith('.trace')]
    if not traces:
        sys.exit(f'usage: trace_parents.py <cpu-art-*.trace> <frame-substring> [...] [--workers N]')
    probes = [a for a in rest if not a.endswith('.trace')]
    if not probes:
        sys.exit(f'usage: trace_parents.py <cpu-art-*.trace> <frame-substring> [frame-substring ...] [--workers N]')
    return traces[0], probes, workers


def main(argv=None):
    if os.name != 'posix':
        sys.exit('trace_parents.py parallel walk needs fork (Unix only)')
    args = sys.argv[1:] if argv is None else argv
    path, probes, workers = parse_argv(args)
    parsed_npz = os.path.splitext(path)[0] + '.parsed.npz'
    cached = require_cached_tables(parsed_npz)
    names = cached['names']
    n = len(names)

    threads = cached['threads']
    main_tid = next((t for t, nm in threads.items() if nm == 'main'), None)
    print('main tid:', main_tid, flush=True)
    print('=== ALL THREADS ===')
    for t in sorted(threads):
        print(f'  tid={t:<3d} {threads[t]}')

    z = np.load(parsed_npz)
    tid, idx, act, ts = z['tid'], z['dex'], z['act'], z['ts']
    ok = (idx >= 0) & (idx < n) & (act <= 2)
    tid, idx, act, ts = tid[ok], idx[ok], act[ok], ts[ok]

    targets = match_targets(names, probes)
    print('\nmatched targets:', len(targets))
    for i, nm in targets.items():
        print(f'  idx={i} {nm[:120]}')

    # Parent/child attribution over ALL threads (was: main only). Workers
    # inherit the filtered arrays via fork; only per-tid slices cross as
    # job args (same pattern as stackwalk.walk_all).
    res = walk_all_parents(tid, idx, act, ts, frozenset(targets), workers=workers)
    parent_time = res['parent']
    child_edges = res['child']
    print(f'\nwalked threads: {len(res["threads"])} (workers={workers or default_workers()})', flush=True)

    for t_idx, t_name in targets.items():
        print(f'\n=== {t_name[:130]} ===')
        par = [(p, v) for (p, c), v in parent_time.items() if c == t_idx]
        par.sort(key=lambda kv: -kv[1][0])
        print('  top parents (all threads; dur us, calls):')
        for p, (d, c) in par[:6]:
            print(f'    {d:<12d} x{c:<6d} {names[p][:120]}')
        ch = [(c, v) for (p, c), v in child_edges.items() if p == t_idx]
        ch.sort(key=lambda kv: -kv[1][0])
        if ch:
            print('  top children (all threads):')
            for c, (d, cc) in ch[:8]:
                print(f'    {d:<12d} x{cc:<6d} {names[c][:120]}')
        else:
            print('  (leaf: no java children — time is native/self)')

    # Timeline on main for the requested targets (first/last ts, ms since trace start)
    t0 = int(ts.min())
    print('\n=== timeline on main (ms since first event) ===')
    for probe in probes:
        pi = [i for i, nm in enumerate(names) if probe in nm]
        if not pi:
            continue
        sel = np.isin(idx, pi) & (tid == main_tid)
        if sel.sum() == 0:
            continue
        ent = ts[sel & (act == 0)]
        if len(ent) == 0:
            continue
        print(f'  {probe:22s} events={int(sel.sum()):<7d} first={(int(ent.min())-t0)/1000:8.1f} '
              f'last={(int(ent.max())-t0)/1000:8.1f} span={(int(ent.max())-int(ent.min()))/1000:.1f}ms')


if __name__ == '__main__':
    main()
