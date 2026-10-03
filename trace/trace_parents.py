"""Parent/child attribution for suspicious frames + timeline."""
import re
import struct
import sys
import os

import numpy as np

TRACES = [a for a in sys.argv[1:] if a.endswith('.trace')]
if not TRACES:
    sys.exit(f'usage: {os.path.basename(sys.argv[0])} <cpu-art-*.trace> [frame-substring ...]')
PATH = TRACES[0]
PARSED_NPZ = os.path.splitext(PATH)[0] + '.parsed.npz'

with open(PATH, 'rb') as f:
    data = f.read()

pat = re.compile(rb'\x01..0x([0-9a-f]+)\t([^\t]*)\t([^\t]*)\t([^\t]*)\t[^\n]*\n')
methods = {}
for m in pat.finditer(data):
    dex = int(m.group(1), 16)
    if dex not in methods:
        methods[dex] = (
            m.group(2).decode('utf-8', 'replace') + '#' +
            m.group(3).decode('utf-8', 'replace') + m.group(4).decode('utf-8', 'replace'))
# Sparse table keyed by method id (ids are multiples of 4); gaps stay ''.
N = max(methods) + 1
names = [methods.get(i, '') for i in range(N)]

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
MAIN = next((t for t, n in threads.items() if n == 'main'), None)
print('main tid:', MAIN, flush=True)
print('=== ALL THREADS ===')
for t in sorted(threads):
    print(f'  tid={t:<3d} {threads[t]}')

z = np.load(PARSED_NPZ)
tid, idx, act, ts = z['tid'], z['dex'], z['act'], z['ts']
ok = (idx >= 0) & (idx < N) & (act <= 2)
tid, idx, act, ts = tid[ok], idx[ok], act[ok], ts[ok]

ARGS = [a for a in sys.argv[1:] if not a.endswith('.trace')]
if not ARGS:
    sys.exit(f'usage: {os.path.basename(sys.argv[0])} <cpu-art-*.trace> <frame-substring> [frame-substring ...]')
TARGETS = ARGS
targets = {}
for i, n in enumerate(names):
    for t in TARGETS:
        if t in n and i not in targets:
            targets[i] = n
print('\nmatched targets:', len(targets))
for i, n in targets.items():
    print(f'  idx={i} {n[:120]}')

# single main-thread walk collecting parent->child edges for targets
m = tid == MAIN
ii, aa, ss = idx[m], act[m], ts[m]
parent_time = {}   # (parent_idx, child_idx) -> [total_dur, count]
child_time = {}    # target_idx -> {child_idx: [dur,count]}
stack = []
for method, a, s in zip(ii.tolist(), aa.tolist(), ss.tolist()):
    if a == 0:
        stack.append([method, s, {}])
    elif stack:
        pm, es, ch = stack.pop()
        if pm != method:
            method = pm
        dur = max(0, s - es)
        if stack:
            p = stack[-1][0]
            key = (p, method)
            pt = parent_time.get(key, [0, 0])
            pt[0] += dur
            pt[1] += 1
            parent_time[key] = pt
            stack[-1][2][method] = stack[-1][2].get(method, [0, 0])
            stack[-1][2][method][0] += dur
            stack[-1][2][method][1] += 1
        if method in targets:
            child_time[method] = ch

for t_idx, t_name in targets.items():
    print(f'\n=== {t_name[:130]} ===')
    par = [(p, v) for (p, c), v in parent_time.items() if c == t_idx]
    par.sort(key=lambda kv: -kv[1][0])
    print('  top parents (incl us, calls):')
    for p, (d, c) in par[:6]:
        print(f'    {d:<12d} x{c:<6d} {names[p][:120]}')
    ch = child_time.get(t_idx, {})
    if ch:
        print('  top children:')
        for c, (d, cc) in sorted(ch.items(), key=lambda kv: -kv[1][0])[:8]:
            print(f'    {d:<12d} x{cc:<6d} {names[c][:120]}')
    else:
        print('  (leaf: no java children — time is native/self)')

# Timeline on main for the requested targets (first/last ts, ms since trace start)
t0 = int(ts.min())
print('\n=== timeline on main (ms since first event) ===')
for probe in TARGETS:
    pi = [i for i, n in enumerate(names) if probe in n]
    if not pi:
        continue
    sel = np.isin(idx, pi) & (tid == MAIN)
    if sel.sum() == 0:
        continue
    t = ts[sel]
    ent = ts[sel & (act == 0)]
    print(f'  {probe:22s} events={int(sel.sum()):<7d} first={(int(ent.min())-t0)/1000:8.1f} '
          f'last={(int(ent.max())-t0)/1000:8.1f} span={(int(ent.max())-int(ent.min()))/1000:.1f}ms')
