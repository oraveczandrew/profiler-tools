#!/usr/bin/env python3
"""Attribute every allocation to its nearest app-owned frame.

A frame counts as app-owned when its class or file path contains one of the
--app markers. Each allocation event is charged to the nearest such frame
on its stack.

usage:
    python3 scripts/asdb_owned.py --db capture.asdb --app com/example/app MainActivity [--workers 8] [--output report.txt]
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

_APP = ()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True, help='path to the .asdb capture')
    parser.add_argument('--app', nargs='+', required=True,
                        help='class/file path markers that count as app-owned')
    parser.add_argument('--workers', type=int, default=0,
                        help='parallel workers (default: CPU count; 1 = serial)')
    parser.add_argument('--output', default=None,
                        help='write the report to a file instead of stdout')
    return parser.parse_args(argv)


_CTX = None
_DB_PATH = ''


def frame_str(mid, line):
    return _frame_str(_CTX, mid, line)


def get_frames(sid):
    return [(m, l) for m, l in _CTX.stacks.get(sid, [])]


def is_app(fr):
    c, m, f, ln = _CTX.methods.get(fr[0], ('', '', '', 0))
    return any(a in (c + f) for a in _APP)


def main(argv=None):
    args = parse_args(argv)
    import contextlib
    out = open(args.output, 'w') if args.output else sys.stdout
    with contextlib.redirect_stdout(out):
        run(args)
    if args.output:
        out.close()


def run(args):
    global _CTX, _DB_PATH, _APP
    global classes, methods, stacks, threads
    global own, own_b, own_thread, top_class, nostack_cls, total
    _DB_PATH = args.db
    _APP = tuple(args.app)
    db = open_db(args.db)
    _CTX = load_contexts(db)
    db.close()
    classes, methods, stacks, threads = _CTX.classes, _CTX.methods, _CTX.stacks, _CTX.threads

    own = collections.Counter()          # (cls, appframe) -> count
    own_b = collections.Counter()        # (cls, appframe) -> bytes
    own_thread = collections.Counter()   # (thread, cls, appframe)
    top_class = collections.Counter()    # (cls, topframe)
    nostack_cls = collections.Counter()
    total = 0

    results = run_parallel(args.db, 409, args.workers, _scan_chunk)

    own = collections.Counter()
    own_b = collections.Counter()
    own_thread = collections.Counter()
    top_class = collections.Counter()
    nostack_cls = collections.Counter()
    total = 0

    for _own, _own_b, _own_thread, _top_class, _nostack_cls, _total in results:
        own.update(_own)
        own_b.update(_own_b)
        own_thread.update(_own_thread)
        top_class.update(_top_class)
        nostack_cls.update(_nostack_cls)
        total += _total

    print('total allocations: %d   with app frame: %d'
          % (total, sum(own.values())))
    print('\n===== ALLOCATIONS ATTRIBUTED TO APP FRAMES (%s) =====' % ', '.join(_APP))
    for (cname, af), c in own.most_common(35):
        print('  %7d  %9d B  %-28s %s' % (c, own_b[(cname, af)], cname, af))

    print('\n===== app-attributed by thread (top 20) =====')
    for (t, cname, af), c in own_thread.most_common(20):
        print('  %7d  %-22s %-24s %s' % (c, t, cname, af))

    print('\n===== java.lang.String: immediate (top) frame, top 20 =====')
    n = 0
    for (cname, tf), c in top_class.most_common():
        if cname == 'Ljava/lang/String;' and n < 20:
            print('  %7d  %s' % (c, tf))
            n += 1

    print('\n===== [Ljava/lang/Object; immediate frame, top 12 =====')
    n = 0
    for (cname, tf), c in top_class.most_common():
        if cname == '[Ljava/lang/Object;' and n < 12:
            print('  %7d  %s' % (c, tf))
            n += 1

    print('\n===== [B immediate frame, top 8 =====')
    n = 0
    for (cname, tf), c in top_class.most_common():
        if cname == '[B' and n < 8:
            print('  %7d  %s' % (c, tf))
            n += 1

    print('\n===== all thread names (%d) =====' % len(threads))
    for tid, nm in sorted(threads.items()):
        print('   %-6d %r' % (tid, nm))


def scan_blobs(blobs):
    global total
    for blob in blobs:
        pl = sub(blob, 409)
        if pl is None:
            continue
        for batch in fm(pl).get(1, []):
            for ev in fm(batch).get(2, []):
                ef = fm(ev)
                ad = ef.get(4)
                if not ad:
                    continue
                total += 1
                af = fm(ad[0] if isinstance(ad[0], bytes) else b'')
                ctag = sv(af, 2)
                cname = classes.get(ctag, '?tag=%d' % ctag)
                size = sv(af, 3)
                sid = sv(af, 6)
                fr = get_frames(sid)
                tname = ss(af, 10) or threads.get(sv(af, 5), 'tid=%d' % sv(af, 5))
                if not fr:
                    nostack_cls[cname] += 1
                if fr:
                    top_class[(cname, frame_str(*fr[0]))] += 1
                appf = None
                for x in fr:
                    if is_app(x):
                        appf = frame_str(*x)
                        break
                if appf:
                    own[(cname, appf)] += 1
                    own_b[(cname, appf)] += size
                    own_thread[(tname, cname, appf)] += 1


def _scan_chunk(lo, hi):
    global own, own_b, own_thread, top_class, nostack_cls, total
    own = collections.Counter()
    own_b = collections.Counter()
    own_thread = collections.Counter()
    top_class = collections.Counter()
    nostack_cls = collections.Counter()
    total = 0
    scan_blobs(chunk_blobs(_DB_PATH, 409, lo, hi))
    return own, own_b, own_thread, top_class, nostack_cls, total


if __name__ == '__main__':
    main()
