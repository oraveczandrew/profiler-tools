#!/usr/bin/env python3
"""Compare exact per-class allocation counts across captures.

For a given watchlist of classes, prints counts, deltas and byte deltas with
the first capture as the baseline. Use it to measure an optimization: capture
before/after under comparable workloads and diff the table.

usage:
    python3 scripts/asdb_compare.py --db before.asdb after.asdb --watch Ljava/lang/String; [B --workers 8
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

from asdbtools import chunk_blobs, load_contexts, open_db
from asdbtools import run as run_parallel
from asdbtools import fm, sub, sv


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', nargs='+', required=True,
                        help='captures to compare (first is the baseline)')
    parser.add_argument('--watch', nargs='+', required=True,
                        help='classes to tabulate (exact descriptor names, e.g. Ljava/lang/String; [B)')
    parser.add_argument('--workers', type=int, default=0,
                        help='parallel workers (default: CPU count; 1 = serial)')
    parser.add_argument('--output', default=None,
                        help='write the report to a file instead of stdout')
    return parser.parse_args(argv)


_P_PATH = ''
_P_CLASSES = {}
_P_WATCH = set()


def _scan_events(blobs, classes, watch_ids):
    count = collections.Counter()
    size = collections.Counter()
    total = 0
    nostack = 0
    for blob in blobs:
        pl = sub(blob, 409)
        if pl is None:
            continue
        for batch in fm(pl).get(1, []):
          for e in fm(batch).get(2, []):
            ef = fm(e)
            if 5 in ef:
                continue
            ad = ef.get(4)
            if not ad:
                continue
            af = fm(ad[0] if isinstance(ad[0], bytes) else b'')
            total += 1
            tag = sv(af, 2)
            if not isinstance(tag, int) or tag not in watch_ids:
                continue
            if not af.get(6):
                nostack += 1
            count[tag] += 1
            size_v = sv(af, 3)
            size[tag] += size_v if isinstance(size_v, int) else 0
    return count, size, total, nostack


def _scan_chunk(lo, hi):
    return _scan_events(chunk_blobs(_P_PATH, 409, lo, hi), _P_CLASSES, _P_WATCH)


def analyze(path, workers, watch):
    db = open_db(path)
    ctx = load_contexts(db)
    db.close()
    classes = ctx.classes
    watch_ids = {cid for cid, n in classes.items() if n in watch}
    global _P_PATH, _P_CLASSES, _P_WATCH
    _P_PATH, _P_CLASSES, _P_WATCH = path, classes, watch_ids
    count = collections.Counter()
    size = collections.Counter()
    total = 0
    nostack = 0
    for _count, _size, _total, _nostack in run_parallel(path, 409, workers, _scan_chunk):
        count.update(_count)
        size.update(_size)
        total += _total
        nostack += _nostack
    return {
        'total': total,
        'nostack': nostack,
        'rows': {classes[t]: (count[t], size[t]) for t in count},
    }


def main(argv=None):
    args = parse_args(argv)
    import contextlib
    out = open(args.output, 'w') if args.output else sys.stdout
    with contextlib.redirect_stdout(out):
        run(args)
    if args.output:
        out.close()


def run(args):
    results = [(p, analyze(p, args.workers, args.watch)) for p in args.db]

    for i, (path, r) in enumerate(results):
        print('=' * 78)
        print(f'{path}   total allocations: {r["total"]}')
    if len(results) > 1:
        base_path, base = results[0]
        for path, other in results[1:]:
            print('=' * 108)
            print(f'{base_path} -> {path}')
            print(f'{"class":50} {base_path[-13:-9]:>10} {path[-13:-9]:>10} {"delta":>9} {"pct":>8}   {"bytes delta":>12}')
            print('-' * 108)
            for name in args.watch:
                ca, sa = base['rows'].get(name, (0, 0))
                cb, sb = other['rows'].get(name, (0, 0))
                if not ca and not cb:
                    continue
                d = cb - ca
                pct = (d / ca * 100) if ca else float('inf')
                print(f'{name:50} {ca:10} {cb:10} {d:+9} {pct:+7.1f}%   {sb - sa:+12}')
            print('-' * 108)
            d_total = other["total"] - base["total"]
            pct_total = (d_total / base["total"] * 100) if base["total"] else float('inf')
            print(f'{"TOTAL allocations":52} {base["total"]:12} {other["total"]:12} '
                  f'{d_total:+10} {pct_total:+7.1f}%')


if __name__ == '__main__':
    main()
