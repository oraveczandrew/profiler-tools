#!/usr/bin/env python3
"""Time-resolved allocation view for an .asdb capture.

Buckets kind-409 allocation events by time and reports, per bucket, the
allocation count, distinct stack count, thread mix and top allocating classes
and call sites. Also detects *stalls*: consecutive buckets where a single call
site dominates the bucket (a long-running loop repeating the same sites).

The batch timestamp is the allocation timestamp of the first event in its row,
so buckets are a lower bound on resolution (the writer emits roughly one row per
flush, typically tens of milliseconds apart).

usage:
    python3 scripts/asdb_timeline.py --db capture.asdb [--bucket-ms 1000] [--workers 8] [--output report.txt]
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

import argparse
import collections
import contextlib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from asdbtools import chunk_dated_blobs
from asdbtools import frame_str as _frame_str
from asdbtools import load_contexts, open_db
from asdbtools import run as run_parallel
from asdbtools import fm, min_timestamp, ss, sub, sv


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True, help='path to the .asdb capture')
    parser.add_argument('--bucket-ms', type=int, default=1000,
                        help='bucket width in milliseconds (default: 1000)')
    parser.add_argument('--top', type=int, default=3,
                        help='top classes/sites shown per detail bucket (default: 3)')
    parser.add_argument('--detail-buckets', type=int, default=20,
                        help='print per-bucket detail for the N busiest buckets (default: 20)')
    parser.add_argument('--stall-concentration', type=float, default=20.0,
                        help='a bucket is "repetitive" when allocations per distinct stack '
                             'reaches this value (default: 20)')
    parser.add_argument('--needle', action='append', default=[],
                        help='frame substring to count per bucket (repeatable); '
                             'forces a full-stack render per allocation')
    parser.add_argument('--workers', type=int, default=0,
                        help='parallel workers (default: CPU count; 1 = serial)')
    parser.add_argument('--output', default=None,
                        help='write the report to a file instead of stdout')
    return parser.parse_args(argv)


_CTX = None
_DB_PATH = ''
_T0 = 0
_BUCKET_NS = 1_000_000
_NEEDLES = ()
_CLASSES = {}
_THREADS = {}


def frame_str(mid, line):
    return _frame_str(_CTX, mid, line)


def stack_str(sid, depth=8):
    return _full_stack(sid, depth)


def _full_stack(sid, depth=None):
    frames = _CTX.stacks.get(sid, [])
    if depth is not None:
        frames = frames[:depth]
    return ' <- '.join(frame_str(m, l) for m, l in frames) or 'stack#%d?' % sid


def main(argv=None):
    args = parse_args(argv)
    out = open(args.output, 'w') if args.output else sys.stdout
    with contextlib.redirect_stdout(out):
        run(args)
    if args.output:
        out.close()


def _merge_counter_dict(dst, src):
    for key, counter in src.items():
        dst.setdefault(key, collections.Counter()).update(counter)


def _merge_set_dict(dst, src):
    for key, ids in src.items():
        dst.setdefault(key, set()).update(ids)


def run(args):
    global _CTX, _DB_PATH, _T0, _BUCKET_NS, _NEEDLES, _CLASSES, _THREADS
    _DB_PATH = args.db
    _BUCKET_NS = max(1, args.bucket_ms) * 1_000_000
    _NEEDLES = tuple(args.needle)

    db = open_db(args.db)
    _CTX = load_contexts(db)
    t0 = min_timestamp(db, 409)
    db.close()
    if t0 is None:
        print('no allocation events (kind 409) in %s' % args.db)
        return
    _T0 = t0
    _CLASSES, _THREADS = _CTX.classes, _CTX.threads

    results = run_parallel(args.db, 409, args.workers, _scan_chunk)

    counts = collections.Counter()
    distinct = {}
    classes = {}
    sites = {}
    threads = {}
    needles = {}
    total = 0
    for _counts, _distinct, _classes, _sites, _threads, _needles, _total in results:
        counts.update(_counts)
        _merge_set_dict(distinct, _distinct)
        _merge_counter_dict(classes, _classes)
        _merge_counter_dict(sites, _sites)
        _merge_counter_dict(threads, _threads)
        _merge_counter_dict(needles, _needles)
        total += _total

    if not counts:
        print('no allocation events (kind 409) in %s' % args.db)
        return

    last_ns = _T0 + (max(counts) + 1) * _BUCKET_NS
    print('timeline: %d allocations over %.1f ms (bucket %d ms), t0=%d'
          % (total, (last_ns - _T0) / 1e6, args.bucket_ms, _T0))
    print('buckets: %d (0..%d)' % (len(counts), max(counts)))

    print('\n================ ALLOCATIONS OVER TIME ================')
    print('concentration = allocations per distinct stack (high = repetitive hot loop)')
    print('%10s %9s %8s %7s %7s %6s %s'
          % ('start_ms', 'allocs', 'stacks', 'main%', 'conc', 'dom%', 'top class'))
    for b in sorted(counts):
        n = counts[b]
        tcount = threads.get(b, collections.Counter())
        main = 100.0 * tcount.get('main', 0) / n if n else 0.0
        stacks_n = len(distinct.get(b, ()))
        conc = n / stacks_n if stacks_n else 0.0
        site_totals = _site_totals(sites.get(b))
        dom_share = 100.0 * site_totals.most_common(1)[0][1] / n if (n and site_totals) else 0.0
        top_class = classes.get(b, collections.Counter()).most_common(1)
        top_class_name = top_class[0][0] if top_class else ''
        print('%10d %9d %8d %6.1f%% %7.1f %5.1f%% %s'
              % (b * args.bucket_ms, n, stacks_n, main, conc, dom_share, top_class_name))

    detail = set(sorted(counts, key=lambda b: -counts[b])[:max(1, args.detail_buckets)])
    print('\n================ PER-BUCKET DETAIL (busiest %d) ================' % len(detail))
    for b in sorted(detail):
        print('\n-- %6d..%-6d ms  (%d allocations) --'
              % (b * args.bucket_ms, (b + 1) * args.bucket_ms, counts[b]))
        print('  top classes:')
        for cname, c in classes.get(b, collections.Counter()).most_common(args.top):
            print('    %7d  %s' % (c, cname))
        print('  top sites:')
        for (cname, site), c in sites.get(b, collections.Counter()).most_common(args.top):
            print('    %7d  %s' % (c, site))

    _print_stalls(counts, distinct, sites, args)

    if _NEEDLES:
        print('\n================ NEEDLE COUNTS PER BUCKET ================')
        print('%10s %9s %s' % ('start_ms', 'allocs', '  '.join('%-24s' % n for n in _NEEDLES)))
        for b in sorted(counts):
            nc = needles.get(b, collections.Counter())
            print('%10d %9d %s'
                  % (b * args.bucket_ms, counts[b],
                     '  '.join('%-24d' % nc.get(n, 0) for n in _NEEDLES)))


def _site_totals(site_counter):
    totals = collections.Counter()
    if not site_counter:
        return totals
    for (_cname, site), c in site_counter.items():
        totals[site] += c
    return totals


def _print_stalls(counts, distinct, sites, args):
    print('\n================ REPETITIVE PHASES (concentration >= %.0f) ================'
          % args.stall_concentration)
    phases = []
    current = None
    for b in sorted(counts):
        n = counts[b]
        stacks_n = len(distinct.get(b, ()))
        conc = n / stacks_n if stacks_n else 0.0
        totals = _site_totals(sites.get(b))
        site = totals.most_common(1)[0][0] if totals else None
        if conc >= args.stall_concentration:
            if current and current['end'] == b - 1:
                current['end'] = b
                current['allocs'] += n
                current['concs'].append(conc)
            else:
                if current:
                    phases.append(current)
                current = {'start': b, 'end': b, 'allocs': n, 'concs': [conc], 'site': site}
            if site:
                current['site'] = site
        else:
            if current:
                phases.append(current)
                current = None
    if current:
        phases.append(current)
    if not phases:
        print('  (none)')
        return
    for p in phases:
        avg = sum(p['concs']) / len(p['concs'])
        print('  %6d..%-6d ms  %8d allocs  avg concentration %6.1f  %s'
              % (p['start'] * args.bucket_ms, (p['end'] + 1) * args.bucket_ms,
                 p['allocs'], avg, p['site']))


def scan_blobs(dated_blobs):
    counts = collections.Counter()
    distinct = {}
    classes = {}
    sites = {}
    threads = {}
    needles = {}
    total = 0
    for ts, blob in dated_blobs:
        b = (ts - _T0) // _BUCKET_NS
        pl = sub(blob, 409)
        if pl is None:
            continue
        for batch in fm(pl).get(1, []):
            for ev in fm(batch).get(2, []):
                ef = fm(ev)
                if 5 in ef:
                    continue  # deallocation
                ad = ef.get(4)
                if not ad:
                    continue
                af = fm(ad[0] if isinstance(ad[0], bytes) else b'')
                ctag = sv(af, 2)
                cname = _CLASSES.get(ctag, '?tag=%d' % ctag)
                sid = sv(af, 6)
                counts[b] += 1
                total += 1
                distinct.setdefault(b, set()).add(sid)
                classes.setdefault(b, collections.Counter())[cname] += 1
                sites.setdefault(b, collections.Counter())[(cname, stack_str(sid))] += 1
                tname = ss(af, 10) or _THREADS.get(sv(af, 5), 'tid=%d' % sv(af, 5))
                threads.setdefault(b, collections.Counter())[tname] += 1
                if _NEEDLES:
                    full = _full_stack(sid)
                    nc = needles.setdefault(b, collections.Counter())
                    for needle in _NEEDLES:
                        if needle in full:
                            nc[needle] += 1
    return counts, distinct, classes, sites, threads, needles, total


def _scan_chunk(lo, hi):
    return scan_blobs(chunk_dated_blobs(_DB_PATH, 409, lo, hi))


if __name__ == '__main__':
    main()
