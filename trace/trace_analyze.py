"""Stack-walk analysis of parsed ART streaming trace."""
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stackwalk import mmap_trace, walk_all

import numpy as np

if len(sys.argv) < 2:
    sys.exit(f'usage: {os.path.basename(sys.argv[0])} <cpu-art-*.trace>')
PATH = sys.argv[1]
PARSED_NPZ = os.path.splitext(PATH)[0] + '.parsed.npz'
STATS_NPZ = os.path.splitext(PATH)[0] + '.stats.npz'

data = mmap_trace(PATH)  # zero-copy; only the method table is scanned here

# method table keyed by method id (multiples of 4); gaps stay ''
pat = re.compile(rb'\x01..0x([0-9a-f]+)\t([^\t]*)\t([^\t]*)\t([^\t]*)\t[^\n]*\n')
methods = {}
for m in pat.finditer(data):
    dex = int(m.group(1), 16)
    if dex not in methods:
        cls = m.group(2).decode('utf-8', 'replace')
        nm = m.group(3).decode('utf-8', 'replace')
        sig = m.group(4).decode('utf-8', 'replace')
        methods[dex] = f'{cls}#{nm}{sig}'
N = max(methods) + 1
names = [methods.get(i, '') for i in range(N)]
print('methods in table:', len(methods), flush=True)

z = np.load(PARSED_NPZ)
tid, idx, act, ts = z['tid'], z['dex'], z['act'], z['ts']
n = len(tid)
ok = (idx >= 0) & (idx < N) & ((act == 0) | (act == 1) | (act == 2))
print(f'records: {n}, usable: {ok.sum()} ({ok.mean():.3f})', flush=True)
tid, idx, act, ts = tid[ok], idx[ok], act[ok], ts[ok]

res = walk_all(tid, idx, act, ts, N)
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
