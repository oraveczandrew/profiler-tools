#!/usr/bin/env python3
"""Break down every allocation under one app call site.

Selects allocations whose nearest app-owned frame equals --target and reports
them by class, thread, immediate allocator and full call chain.

usage:
    python3 scripts/asdb_site.py --db capture.asdb --target "SVGParserImpl;.parse(:360)"
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
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from asdbtools import chunk_blobs
from asdbtools import frame_str as _frame_str
from asdbtools import load_contexts, open_db
from asdbtools import run as run_parallel
from asdbtools import fm, ss, sub, sv


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True, help='path to the .asdb capture')
    parser.add_argument('--target', required=True,
                        help='call site rendered like "Cls;.method(file:line)"')
    parser.add_argument('--app', nargs='+', required=True,
                        help='class/file path markers that count as app-owned')
    parser.add_argument('--workers', type=int, default=0,
                        help='parallel workers (default: CPU count; 1 = serial)')
    parser.add_argument('--output', default=None,
                        help='write the report to a file instead of stdout')
    return parser.parse_args(argv)


_CTX = None
_DB_PATH = ''
_TARGET = ''
_APP = ()


def frame_str(mid, line):
    return _frame_str(_CTX, mid, line)


def is_app(fr):
    cname, _, fname_, _ = _CTX.methods.get(fr[0], ('', '', '', 0))
    haystack = cname + fname_
    return any(a in haystack for a in _APP)


def main(argv=None):
    args = parse_args(argv)
    import contextlib
    out = open(args.output, 'w') if args.output else sys.stdout
    with contextlib.redirect_stdout(out):
        run(args)
    if args.output:
        out.close()


def run(args):
    global _CTX, _DB_PATH, _TARGET, _APP
    global by_class, by_bytes, by_thread, by_immediate, by_chain, total
    global classes, methods, stacks, threads
    _DB_PATH, _TARGET = args.db, args.target
    _APP = tuple(args.app)
    db = open_db(args.db)
    _CTX = load_contexts(db)
    db.close()
    classes, methods, stacks, threads = _CTX.classes, _CTX.methods, _CTX.stacks, _CTX.threads

    by_class = collections.Counter()
    by_bytes = collections.Counter()
    by_thread = collections.Counter()
    by_immediate = collections.Counter()
    by_chain = collections.Counter()
    total = 0

    results = run_parallel(args.db, 409, args.workers, _scan_chunk)

    by_class = collections.Counter()
    by_bytes = collections.Counter()
    by_thread = collections.Counter()
    by_immediate = collections.Counter()
    by_chain = collections.Counter()
    total = 0

    for _by_class, _by_bytes, _by_thread, _by_immediate, _by_chain, _total in results:
        by_class.update(_by_class)
        by_bytes.update(_by_bytes)
        by_thread.update(_by_thread)
        by_immediate.update(_by_immediate)
        by_chain.update(_by_chain)
        total += _total

    print('target: %s' % _TARGET)
    print('matching allocations: %d of %d (%.2f%%)' % (
        sum(by_class.values()), total,
        (sum(by_class.values()) / total * 100) if total else 0,
    ))
    print('\n===== BY CLASS =====')
    for cname, count in by_class.most_common(40):
        print('  %7d  %9d B  %s' % (count, by_bytes[cname], cname))
    print('\n===== BY THREAD AND CLASS (top 30) =====')
    for (tname, cname), count in by_thread.most_common(30):
        print('  %7d  %-22s %s' % (count, tname, cname))
    print('\n===== IMMEDIATE ALLOCATOR BY CLASS (top 40) =====')
    for (cname, immediate), count in by_immediate.most_common(40):
        print('  %7d  %-28s %s' % (count, cname, immediate))
    print('\n===== FULL CHAINS (top 30) =====')
    for (cname, chain), count in by_chain.most_common(30):
        print('  %7d  %s\n      %s' % (count, cname, chain))


def scan_blobs(blobs):
    global total
    for blob in blobs:
        pl = sub(blob, 409)
        if pl is None:
            continue
        for batch in fm(pl).get(1, []):
            for ev in fm(batch).get(2, []):
                ef = fm(ev)
                if 5 in ef:
                    continue
                ad = ef.get(4)
                if not ad:
                    continue
                total += 1
                af = fm(ad[0] if isinstance(ad[0], bytes) else b'')
                cname = classes.get(sv(af, 2), '?tag=%d' % sv(af, 2))
                frames = stacks.get(sv(af, 6), [])
                target_index = None
                for i, fr in enumerate(frames):
                    if not is_app(fr):
                        continue
                    if frame_str(*fr) == _TARGET:
                        target_index = i
                        break
                    # Keep the nearest app frame only. Anything above TARGET belongs
                    # to a different app call path.
                    break
                if target_index is None:
                    continue
                size = sv(af, 3)
                tid = sv(af, 5)
                tname = ss(af, 10) or threads.get(tid, 'tid=%d' % tid)
                by_class[cname] += 1
                by_bytes[cname] += size
                by_thread[(tname, cname)] += 1
                chain = ' <- '.join(frame_str(*fr) for fr in frames[:target_index + 1])
                by_immediate[(cname, frame_str(*frames[0]))] += 1
                by_chain[(cname, chain)] += 1


def _scan_chunk(lo, hi):
    global by_class, by_bytes, by_thread, by_immediate, by_chain, total
    by_class = collections.Counter()
    by_bytes = collections.Counter()
    by_thread = collections.Counter()
    by_immediate = collections.Counter()
    by_chain = collections.Counter()
    total = 0
    scan_blobs(chunk_blobs(_DB_PATH, 409, lo, hi))
    return by_class, by_bytes, by_thread, by_immediate, by_chain, total


if __name__ == '__main__':
    main()
