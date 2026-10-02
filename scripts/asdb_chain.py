#!/usr/bin/env python3
"""Find allocation stacks containing all of the given frame substrings.

Prints every distinct full call chain (with allocation class and count) whose
rendered frames each contain the corresponding needle.

usage:
    python3 scripts/asdb_chain.py --db capture.asdb <substr> [<substr> ...]
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
from asdbtools import fm, sub, sv


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True, help='path to the .asdb capture')
    parser.add_argument('needles', nargs='+', help='frame substrings, all must match')
    parser.add_argument('--workers', type=int, default=0,
                        help='parallel workers (default: CPU count; 1 = serial)')
    parser.add_argument('--output', default=None,
                        help='write the report to a file instead of stdout')
    return parser.parse_args(argv)


_CTX = None
_DB_PATH = ''
_NEEDLES = []


def frame_str(mid, line):
    return _frame_str(_CTX, mid, line)


def main(argv=None):
    args = parse_args(argv)
    import contextlib
    out = open(args.output, 'w') if args.output else sys.stdout
    with contextlib.redirect_stdout(out):
        run(args)
    if args.output:
        out.close()


def run(args):
    global _CTX, _DB_PATH, _NEEDLES
    global classes, methods, stacks, threads
    global chains, total
    _DB_PATH, _NEEDLES = args.db, args.needles
    db = open_db(args.db)
    _CTX = load_contexts(db)
    db.close()
    classes, methods, stacks, threads = _CTX.classes, _CTX.methods, _CTX.stacks, _CTX.threads

    chains = collections.Counter()
    total = 0

    results = run_parallel(args.db, 409, args.workers, _scan_chunk)

    chains = collections.Counter()
    total = 0

    for _chains, _total in results:
        chains.update(_chains)
        total += _total

    print('matching allocations: %d' % total)
    for (cname, chain), count in chains.most_common(25):
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
                af = fm(ad[0] if isinstance(ad[0], bytes) else b'')
                cname = classes.get(sv(af, 2), '?')
                frames = stacks.get(sv(af, 6), [])
                rendered = [frame_str(*fr) for fr in frames]
                if all(any(n in f for f in rendered) for n in _NEEDLES):
                    total += 1
                    chains[(cname, ' <- '.join(rendered))] += 1


def _scan_chunk(lo, hi):
    global chains, total
    chains = collections.Counter()
    total = 0
    scan_blobs(chunk_blobs(_DB_PATH, 409, lo, hi))
    return chains, total


if __name__ == '__main__':
    main()
