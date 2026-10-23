#!/usr/bin/env python3
"""Group allocations by a chosen dimension.

Rolls up every kind-409 allocation event by one of:

  class    the exact allocated class name
  package  the class's package prefix (see --depth)
  thread   the allocating thread name
  owner    the nearest app-owned frame on the stack (see --app)

usage:
    python3 scripts/asdb_rollup.py --db capture.asdb --by package [--workers 8]
    python3 scripts/asdb_rollup.py --db capture.asdb --by owner --app com/example/app
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

from asdbtools import chunk_blobs
from asdbtools import frame_str as _frame_str
from asdbtools import load_contexts, open_db
from asdbtools import run as run_parallel
from asdbtools import fm, ss, sub, sv


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True, help='path to the .asdb capture')
    parser.add_argument('--by', required=True,
                        choices=('class', 'package', 'thread', 'owner'),
                        help='grouping dimension')
    parser.add_argument('--app', nargs='+', default=[],
                        help='class/file path markers for --by owner')
    parser.add_argument('--depth', type=int, default=3,
                        help='package path components kept for --by package (default: 3)')
    parser.add_argument('--top', type=int, default=30,
                        help='rows shown (default: 30)')
    parser.add_argument('--workers', type=int, default=0,
                        help='parallel workers (default: CPU count; 1 = serial)')
    parser.add_argument('--output', default=None,
                        help='write the report to a file instead of stdout')
    return parser.parse_args(argv)


_CTX = None
_DB_PATH = ''
_BY = 'class'
_APP = ()
_DEPTH = 3
_CLASSES = {}
_THREADS = {}


def frame_str(mid, line):
    return _frame_str(_CTX, mid, line)


def package_of(name, depth):
    """Package prefix of a class name ('arrays' for array types)."""
    if not name:
        return '(unknown)'
    n = name
    if n.startswith('L') and n.endswith(';'):
        n = n[1:-1]
    if n.startswith('[') or n.endswith('[]'):
        return 'arrays'
    parts = n.split('/')
    package = parts[:-1]  # drop the simple class name
    if not package:
        return '(default)'
    return '/'.join(package[:max(1, depth)])


def is_app(fr):
    c, _m, f, _ln = _CTX.methods.get(fr[0], ('', '', '', 0))
    return any(a in (c + f) for a in _APP)


def main(argv=None):
    args = parse_args(argv)
    out = open(args.output, 'w') if args.output else sys.stdout
    with contextlib.redirect_stdout(out):
        run(args)
    if args.output:
        out.close()


def run(args):
    global _CTX, _DB_PATH, _BY, _APP, _DEPTH, _CLASSES, _THREADS
    _DB_PATH = args.db
    _BY = args.by
    _APP = tuple(args.app)
    _DEPTH = max(1, args.depth)
    if _BY == 'owner' and not _APP:
        print('--by owner requires at least one --app marker', file=sys.stderr)
        return

    db = open_db(args.db)
    _CTX = load_contexts(db)
    db.close()
    _CLASSES, _THREADS = _CTX.classes, _CTX.threads

    results = run_parallel(args.db, 409, args.workers, _scan_chunk)

    counts = collections.Counter()
    sizes = collections.Counter()
    threads = collections.Counter()
    total = 0
    for _counts, _sizes, _threads, _total in results:
        counts.update(_counts)
        sizes.update(_sizes)
        threads.update(_threads)
        total += _total

    print('total allocations: %d   grouped by: %s%s'
          % (total, _BY, ' (%s)' % ', '.join(_APP) if _BY == 'owner' else ''))
    print('\n================ TOP %d BY %s ================' % (args.top, _BY.upper()))
    print('%9s %13s %7s  %s' % ('allocs', 'bytes', 'share', _BY))
    for key, c in counts.most_common(args.top):
        print('%9d %13d %6.1f%%  %s' % (c, sizes[key], 100.0 * c / total if total else 0.0, key))

    if _BY not in ('thread',):
        print('\n================ ALLOCATING THREADS ================')
        for t, c in threads.most_common(20):
            print('%9d %6.1f%%  %s' % (c, 100.0 * c / total if total else 0.0, t))


def key_of(cname, sid, tname):
    if _BY == 'class':
        return cname
    if _BY == 'package':
        return package_of(cname, _DEPTH)
    if _BY == 'thread':
        return tname
    for fr in _CTX.stacks.get(sid, []):  # owner
        if is_app(fr):
            return frame_str(*fr)
    return '(no app frame)'


def scan_blobs(blobs):
    counts = collections.Counter()
    sizes = collections.Counter()
    threads = collections.Counter()
    total = 0
    for blob in blobs:
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
                size = sv(af, 3)
                sid = sv(af, 6)
                tname = ss(af, 10) or _THREADS.get(sv(af, 5), 'tid=%d' % sv(af, 5))
                key = key_of(cname, sid, tname)
                counts[key] += 1
                sizes[key] += size
                threads[tname] += 1
                total += 1
    return counts, sizes, threads, total


def _scan_chunk(lo, hi):
    return scan_blobs(chunk_blobs(_DB_PATH, 409, lo, hi))


if __name__ == '__main__':
    main()
