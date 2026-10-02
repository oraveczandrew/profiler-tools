"""Deep cuts: threads, main-thread frames, callers, namespace rollup."""
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stackwalk import walk_all

import numpy as np

if len(sys.argv) < 2:
    sys.exit(f'usage: {os.path.basename(sys.argv[0])} <cpu-art-*.trace> [--ns MARKER ...] [--group SUB ...]')
PATH = sys.argv[1]
PARSED_NPZ = os.path.splitext(PATH)[0] + '.parsed.npz'
STATS_NPZ = os.path.splitext(PATH)[0] + '.stats.npz'


def take_flag(flag):
    """Collect values following `flag` in argv (stops at the next flag)."""
    vals, i, argv = [], 0, sys.argv[2:]
    while i < len(argv):
        if argv[i] == flag:
            i += 1
            while i < len(argv) and not argv[i].startswith('--'):
                vals.append(argv[i])
                i += 1
        else:
            i += 1
    return vals


NS_MARKERS = take_flag('--ns')
GROUP_SUBS = take_flag('--group')

with open(PATH, 'rb') as f:
    data = f.read()

pat = re.compile(rb'\x01..0x[0-9a-f]+\t([^\t]*)\t([^\t]*)\t([^\t]*)\t[^\n]*\n')
names = []
for m in pat.finditer(data):
    names.append(
        m.group(1).decode('utf-8', 'replace') + '#' +
        m.group(2).decode('utf-8', 'replace') + m.group(3).decode('utf-8', 'replace'))
N = len(names)
name_of = {i: n for i, n in enumerate(names)}
idx_of = {}
for i, n in enumerate(names):
    idx_of.setdefault(n, i)

# threads: full-file scan (late-created threads are recorded mid-stream)
import struct


def thread_at(buf, off):
    if buf[off] != 0x02 or off + 7 > len(buf):
        return None
    t, nlen = struct.unpack('<HH', buf[off + 1:off + 5])
    if t > 1000 or nlen < 2 or nlen > 120 or off + 5 + nlen + 2 > len(buf):
        return None
    nm = buf[off + 5:off + 5 + nlen]
    if not all(32 <= b < 127 for b in nm):
        return None
    if buf[off + 5 + nlen:off + 5 + nlen + 2] != b'\x00\x00':
        return None
    return t, nm.decode('utf-8', 'replace')


threads = {}
pos = data.find(b'\x02')
while pos != -1:
    r = thread_at(data, pos)
    if r is not None:
        threads.setdefault(r[0], r[1])
        pos = data.find(b'\x02', pos + 1)
    else:
        pos = data.find(b'\x02', pos + 1)
MAIN = next((t for t, n in threads.items() if n == 'main'), None)
print('main tid:', MAIN, flush=True)

z = np.load(PARSED_NPZ)
tid, idx, act, ts = z['tid'], z['dex'], z['act'], z['ts']
ok = (idx >= 0) & (idx < N) & (act <= 2)
tid, idx, act, ts = tid[ok], idx[ok], act[ok], ts[ok]
st = np.load(STATS_NPZ)
incl, excl, calls = st['incl'], st['excl'], st['calls']

print('=== THREADS: events + exclusive busy (ms) ===')
res = walk_all(tid, idx, act, ts, N)
rows = [(busy / 1000, ev, t, threads.get(int(t), '?')) for busy, ev, t in res['rows']]
for busy_ms, ev, t, nm in sorted(rows, reverse=True)[:15]:
    print(f'{busy_ms:12.1f}ms  ev={ev:<9d} tid={t:<3d} {nm}')

print(f'\n=== MAIN tid={MAIN} top inclusive (us) ===')
mi = np.where((tid == MAIN))[0]
midx = idx[tid == MAIN]
from collections import Counter
c = Counter()
s = st['incl'], st['excl']
# per-method incl on main: recompute via stack walk limited to main
mm_idx, mm_act, mm_ts = idx[tid == MAIN], act[tid == MAIN], ts[tid == MAIN]
mi_incl = {}
mi_excl = {}
mi_calls = Counter()
stack = []
for method, a, tss in zip(mm_idx.tolist(), mm_act.tolist(), mm_ts.tolist()):
    if a == 0:
        stack.append([method, tss, 0])
        mi_calls[method] += 1
    elif stack:
        pm, es, ch = stack.pop()
        if pm != method:
            method = pm
        dur = max(0, tss - es)
        mi_incl[method] = mi_incl.get(method, 0) + dur
        mi_excl[method] = mi_excl.get(method, 0) + dur - ch
        if stack:
            stack[-1][2] += dur
top = sorted(mi_incl.items(), key=lambda kv: -kv[1])[:25]
for mth, v in top:
    print(f'{v:12d}  excl={mi_excl.get(mth,0):<10d} x{mi_calls[mth]:<7d} {name_of[mth][:140]}')

print('\n=== doFrame distribution on main (ms) ===')
df = idx_of.get('android.view.Choreographer#doFrame(JILandroid/view/DisplayEventReceiver$VsyncEventData;)V')
durs = []
stack = []
for method, a, tss in zip(mm_idx.tolist(), mm_act.tolist(), mm_ts.tolist()):
    if method == df and a == 0:
        stack.append(tss)
    elif method == df and stack:
        es = stack.pop()
        durs.append((tss - es) / 1000)
durs = sorted(durs)
print('frames:', len(durs))
if durs:
    import statistics
    print('min/med/p90/max:', round(durs[0],1), round(durs[len(durs)//2],1),
          round(durs[int(len(durs)*0.9)-1 if len(durs)>1 else 0],1), round(durs[-1],1))
    print('all:', [round(x,1) for x in durs])

if NS_MARKERS or GROUP_SUBS:
    print('\n=== namespace rollup (global, us) ===')
    ns_idx = [i for i, n in enumerate(names)
              if not NS_MARKERS or any(m in n for m in NS_MARKERS)]
    ni = sum(int(incl[i]) for i in ns_idx)
    ne = sum(int(excl[i]) for i in ns_idx)
    nc = sum(int(calls[i]) for i in ns_idx)
    print(f'methods={len(ns_idx)} incl={ni} excl={ne} calls={nc}')
    for group in GROUP_SUBS:
        gi = [i for i in ns_idx if group in name_of[i]]
        print(f'  {group:10s} incl={sum(int(incl[i]) for i in gi):<12d} excl={sum(int(excl[i]) for i in gi):<12d}')
    print('  top incl:')
    for i in sorted(ns_idx, key=lambda i: -incl[i])[:15]:
        print(f'    {int(incl[i]):<10d} excl={int(excl[i]):<10d} x{int(calls[i]):<7d} {name_of[i][:130]}')
