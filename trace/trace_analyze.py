"""Stack-walk analysis of parsed ART streaming trace.

usage:
    python3 trace/trace_analyze.py <cpu-art-*.trace> [--workers N]
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stackwalk import require_cached_tables, walk_all

import numpy as np

if len(sys.argv) < 2 or sys.argv[1].startswith('--'):
    sys.exit(f'usage: {os.path.basename(sys.argv[0])} <cpu-art-*.trace> [--workers N]')
PATH = sys.argv[1]
PARSED_NPZ = os.path.splitext(PATH)[0] + '.parsed.npz'
STATS_NPZ = os.path.splitext(PATH)[0] + '.stats.npz'


def take_single(flag):
    vals = [a.split('=', 1)[1] if a.startswith(flag + '=') else None for a in sys.argv[2:]]
    for i, a in enumerate(sys.argv[2:]):
        if a == flag and i + 1 < len(sys.argv[2:]):
            try:
                return int(sys.argv[2:][i + 1])
            except ValueError:
                sys.exit(f'{flag} needs an integer')
    for v in vals:
        if v is not None:
            try:
                return int(v)
            except ValueError:
                sys.exit(f'{flag} needs an integer')
    return 0


WORKERS = take_single('--workers')

# Method names come precomputed from trace_parse's .parsed.npz cache
# (downstream scripts never re-read the multi-GB .trace).
cached = require_cached_tables(PARSED_NPZ)
names = cached['names']
N = len(names)
print('methods in table:', sum(1 for n in names if n), '(cached)', flush=True)

z = np.load(PARSED_NPZ)
tid, idx, act, ts = z['tid'], z['dex'], z['act'], z['ts']
n = len(tid)
ok = (idx >= 0) & (idx < N) & ((act == 0) | (act == 1) | (act == 2))
print(f'records: {n}, usable: {ok.sum()} ({ok.mean():.3f})', flush=True)
tid, idx, act, ts = tid[ok], idx[ok], act[ok], ts[ok]

res = walk_all(tid, idx, act, ts, N, workers=WORKERS)
incl, excl, calls = res['incl'], res['excl'], res['calls']
err_empty, err_mismatch, err_back = res['err_empty'], res['err_mismatch'], res['err_back']
# NOTE: the old per-thread thread_total dict was computed but never printed
# or saved, so it is gone; unclosed frames are still ignored (trace cut).
print('stack errors: empty=%d mismatch=%d backward=%d' % (err_empty, err_mismatch, err_back), flush=True)

np.savez_compressed(STATS_NPZ, incl=incl, excl=excl, calls=calls,
                     names=np.array(names, dtype=object))
print(f'saved {STATS_NPZ}', flush=True)

order = np.argsort(incl)[::-1]
print('\n=== TOP 20 inclusive (us) ===')
for i in order[:20]:
    print(f'{incl[i]:12d}  x{calls[i]:<8d} {names[i][:150]}')
print('\n=== TOP 20 exclusive (us) ===')
order2 = np.argsort(excl)[::-1]
for i in order2[:20]:
    print(f'{excl[i]:12d}  x{calls[i]:<8d} {names[i][:150]}')
