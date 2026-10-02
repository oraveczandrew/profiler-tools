#!/usr/bin/env python3
"""Top allocating classes and call sites in an .asdb capture.

Decodes kind-408 contexts (classes, methods, stacks, threads) and scans every
kind-409 allocation event, attributing each to its allocating call stack.

usage:
    python3 scripts/asdb_top.py --db capture.asdb [--workers 8] [--output report.txt]
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
from asdbtools import ss as sstr
from asdbtools import stack_str as _stack_str
from asdbtools import sub as payload
from asdbtools import fm, sub, sv


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True, help='path to the .asdb capture')
    parser.add_argument('--workers', type=int, default=0,
                        help='parallel workers (default: CPU count; 1 = serial)')
    parser.add_argument('--output', default=None,
                        help='write the report to a file instead of stdout')
    return parser.parse_args(argv)


_CTX = None


def frame_str(mid, line):
    return _frame_str(_CTX, mid, line)


def stack_str(sid, depth=6):
    return _stack_str(_CTX, sid, depth)


def main(argv=None):
    args = parse_args(argv)
    import contextlib
    out = open(args.output, 'w') if args.output else sys.stdout
    with contextlib.redirect_stdout(out):
        run(args)
    if args.output:
        out.close()


def run(args):
    global _CTX, _DB_PATH
    global classes, methods, stacks, threads
    global cnt, bys, site, site_any, thread_of, str_len, alloc_n, free_n, nostack
    _DB_PATH = args.db
    db = open_db(args.db)
    _CTX = load_contexts(db)
    db.close()
    classes, methods, stacks, threads = _CTX.classes, _CTX.methods, _CTX.stacks, _CTX.threads

    print('classes: %d  methods: %d  stacks: %d  threads: %d'
          % (len(classes), len(methods), len(stacks), len(threads)))
    print('thread names:', sorted(set(threads.values()))[:20])

    cnt = collections.Counter()
    bys = collections.Counter()
    site = collections.Counter()          # (class_name, stack_str)
    site_any = collections.Counter()      # (stack_str)
    thread_of = collections.Counter()     # (thread_name, class_name)
    str_len = collections.Counter()
    alloc_n = free_n = 0
    nostack = 0

    results = run_parallel(args.db, 409, args.workers, _scan_chunk)

    cnt = collections.Counter()
    bys = collections.Counter()
    site = collections.Counter()
    site_any = collections.Counter()
    thread_of = collections.Counter()
    str_len = collections.Counter()
    alloc_n = free_n = 0
    nostack = 0

    for _cnt, _bys, _site, _site_any, _thread_of, _str_len, _alloc_n, _free_n, _nostack in results:
        cnt.update(_cnt)
        bys.update(_bys)
        site.update(_site)
        site_any.update(_site_any)
        thread_of.update(_thread_of)
        str_len.update(_str_len)
        alloc_n += _alloc_n
        free_n += _free_n
        nostack += _nostack

    print('allocations: %d   deallocations: %d   allocations without stack: %d'
          % (alloc_n, free_n, nostack))

    print('\n================ TOP 20 CLASSES ================')
    for k, c in cnt.most_common(20):
        print('  %8d  %10d B   %s' % (c, bys[k], classes.get(k, '?tag=%d' % k)))

    print('\n================ STRING FAMILY ================')
    tot = 0
    for k, c in cnt.most_common():
        nm = classes.get(k, '')
        if 'String' in nm or nm in ('[C', '[B', 'char[]'):
            tot += c
            print('  %8d  %10d B   %s' % (c, bys[k], nm))
    print('  String-family total: %d (%.1f%% of %d allocations)'
          % (tot, 100.0 * tot / alloc_n, alloc_n))

    print('\n================ java.lang.String CALL SITES ================')
    for (cname, s), c in site.most_common():
        if cname == 'Ljava/lang/String;':
            print('  %8d  %s' % (c, s))
            if c < 200:
                continue
            break
    print('  ... all String sites, top 15:')
    n = 0
    for (cname, s), c in site.most_common():
        if cname == 'Ljava/lang/String;' and n < 15:
            print('  %8d  %s' % (c, s))
            n += 1

    print('\n================ TOP 15 ALLOCATION SITES (any class) ================')
    for s, c in site_any.most_common(15):
        print('  %8d  %s' % (c, s))

    print('\n================ String length distribution ================')
    for (nm, ln), c in sorted(str_len.items(), key=lambda t: -t[1])[:15]:
        print('  %-24s len=%-5d %d' % (nm, ln, c))

    print('\n================ top threads for String ================')
    for (t, cname), c in thread_of.most_common(40):
        if 'String' in cname:
            print('  %8d  %-22s %s' % (c, t, cname))


def scan_blobs(blobs):
    global alloc_n, free_n, nostack
    for b in blobs:
        pl = payload(b, 409)
        if pl is None:
            continue
        for batch in fm(pl).get(1, []):
            for ev in fm(batch).get(2, []):
                ef = fm(ev)
                if 5 in ef:
                    free_n += 1
                    continue
                ad = ef.get(4)
                if not ad:
                    continue
                alloc_n += 1
                af = fm(ad[0] if isinstance(ad[0], bytes) else b'')
                ctag = sv(af, 2)
                size = sv(af, 3)
                cnt[ctag] += 1
                bys[ctag] += size
                cname = classes.get(ctag, '?tag=%d' % ctag)
                tid = sv(af, 5)
                tname = sstr(af, 10) or threads.get(tid, 'tid=%d' % tid)
                thread_of[(tname, cname)] += 1
                sid = sv(af, 6)
                if not sid:
                    nostack += 1
                    sstr_ = '(no stack)'
                else:
                    sstr_ = stack_str(sid)
                site[(cname, sstr_)] += 1
                site_any[sstr_] += 1
                if 'String' in cname or cname in ('[C', '[B'):
                    str_len[(cname, sv(af, 4))] += 1


def _scan_chunk(lo, hi):
    global cnt, bys, site, site_any, thread_of, str_len, alloc_n, free_n, nostack
    from asdbtools import chunk_blobs as _blobs
    from asdbtools.db import open_db as _open
    _ = _open, _blobs  # referenced for clarity; DB_PATH passed below
    cnt = collections.Counter()
    bys = collections.Counter()
    site = collections.Counter()
    site_any = collections.Counter()
    thread_of = collections.Counter()
    str_len = collections.Counter()
    alloc_n = free_n = 0
    nostack = 0
    scan_blobs(chunk_blobs(_DB_PATH, 409, lo, hi))
    return cnt, bys, site, site_any, thread_of, str_len, alloc_n, free_n, nostack


_DB_PATH = ''


if __name__ == '__main__':
    main()
