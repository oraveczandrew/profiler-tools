"""Parser for Android Studio ART streaming method traces (cpu-art-*.trace).

Empirically reverse-engineered layout (checked against AOSP
platform/art/runtime/trace.{h,cc} + tools/stream-trace-converter.py):
  header: 'SLOW' + misc (32 bytes, offsetToData=0x20); version 0xf2 =
  streaming single-clock, record size u16 at offset 16 (10 or 14 bytes)
  thread records: 02 <id:u16> <len:u16> <name> 00 00
  method records: 01 <u16 len> '0x<index>' \\t class \\t name \\t sig \\t file \\n (+pad 00s)
    - method key = small sequential index (multiples of 4), NOT a pointer:
      data records carry <index|action>, so mask the low 2 bits (NO shift)
  data records: <tid:u16> <index|action:u32> <ts_us:u32> (+ second ts for dual)
    - action: 0=enter, 1/2=exit; ts = wall microseconds (deltas per thread are exact)
  footer: text key=value pairs + *threads/*methods/*end markers
Data and method records are interleaved (streaming flushes).
"""
import re
import struct
import sys
import os

import numpy as np

if len(sys.argv) < 2:
    sys.exit(f'usage: {os.path.basename(sys.argv[0])} <cpu-art-*.trace>')
PATH = sys.argv[1]
OUT = os.path.splitext(PATH)[0] + '.parsed.npz'

with open(PATH, 'rb') as f:
    data = f.read()
print('file size:', len(data), flush=True)
if not data:
    sys.exit('empty trace file (still recording?)')

# --- threads: full-file scan (late-created threads are recorded mid-stream) ---
# Grammar: `02 <id:u16> <len:u16> <name>` core, optionally followed by a `00 00`
# separator when another special (thread/method) record follows immediately.
# When DATA follows, there is no separator at all.
def thread_at(data, off):
    if data[off] != 0x02 or off + 7 > len(data):
        return None
    tid, nlen = struct.unpack('<HH', data[off + 1:off + 5])
    if tid > 1000 or nlen < 2 or nlen > 120 or off + 5 + nlen > len(data):
        return None
    name = data[off + 5:off + 5 + nlen]
    if not all(32 <= b < 127 for b in name):
        return None
    end = off + 5 + nlen
    if data[end:end + 2] == b'\x00\x00' and end + 2 < len(data) and data[end + 2] in (0x01, 0x02):
        end += 2
    return tid, name.decode('utf-8', 'replace'), end - off

threads = {}
tspans = []
pos = data.find(b'\x02')
while pos != -1:
    r = thread_at(data, pos)
    if r is not None:
        tid, name, ln = r
        threads.setdefault(tid, []).append(name)
        tspans.append((pos, pos + ln))
        pos = data.find(b'\x02', pos + ln)
    else:
        pos = data.find(b'\x02', pos + 1)
print('thread records:', len(tspans), 'unique tids:', len(threads), flush=True)
for tid in sorted(threads):
    print(f'  tid={tid:<4d} {threads[tid]}', flush=True)

# --- methods (union of framings, keyed by dex idx) ---
pat = re.compile(rb'\x01..0x([0-9a-f]+)\t([^\t]*)\t([^\t]*)\t([^\t]*)\t([^\n]*)\n')
methods = {}
spans = []
for m in pat.finditer(data):
    dex = int(m.group(1), 16)
    if dex not in methods:
        cls = m.group(2).decode('utf-8', 'replace')
        nm = m.group(3).decode('utf-8', 'replace')
        sig = m.group(4).decode('utf-8', 'replace')
        methods[dex] = f'{cls}#{nm}{sig}'
    spans.append((m.start(), m.end()))
print('methods:', len(methods), flush=True)

# --- footer bounds ---
foot_start = data.find(b'*version\n')
print('footer at:', hex(foot_start), flush=True)
print(data[foot_start:].decode('utf-8', 'replace')[:600], flush=True)

# --- data gaps: everything not covered by known spans ---
spans.append((0x0, 0x20))  # header
spans.extend(tspans)  # thread records (incl. mid-stream)
spans.append((foot_start, len(data)))
spans.sort()
merged = []
for s, e in spans:
    if merged and s <= merged[-1][1]:
        merged[-1][1] = max(merged[-1][1], e)
    else:
        merged.append([s, e])
gaps = []
prev = 0
for s, e in merged:
    if s > prev:
        gaps.append((prev, s))
    prev = max(prev, e)
gap_bytes = sum(e - s for s, e in gaps)
print('gap regions:', len(gaps), 'gap bytes:', gap_bytes,
      'lost to 10B align:', gap_bytes % 10, flush=True)

# --- parse records, gap by gap (no giant join: peak RAM stays ~2x file) ---
# dtypes are minimal: tid/ma u32, ts u64 (wall clock exceeds u32), act u8.
tid_parts, ma_parts, ts_parts = [], [], []
n_records = 0
for s, e in gaps:
    ln = (e - s) // 10 * 10
    if not ln:
        continue
    w16 = np.frombuffer(data, dtype=np.uint16, count=ln // 2, offset=s).reshape(-1, 5)
    tid16 = w16[:, 0].astype(np.uint32)
    ma = w16[:, 1].astype(np.uint32) | (w16[:, 2].astype(np.uint32) << 16)
    ts = w16[:, 3].astype(np.uint64) | (w16[:, 4].astype(np.uint64) << 16)
    tid_parts.append(tid16)
    ma_parts.append(ma)  # narrowed to key/act below after keying decision
    ts_parts.append(ts)
    n_records += len(w16)
    del w16, tid16, ma, ts
print('records:', n_records, flush=True)
tid_all = np.concatenate(tid_parts) if tid_parts else np.empty(0, dtype=np.uint32)
ma_all = np.concatenate(ma_parts) if ma_parts else np.empty(0, dtype=np.uint32)
ts_all = np.concatenate(ts_parts) if ts_parts else np.empty(0, dtype=np.uint64)
del tid_parts, ma_parts, ts_parts

dex_shift = ma_all >> np.uint32(2)
act = (ma_all & np.uint32(3)).astype(np.uint8)
# tid==0 chunks are special records (method/thread/summary markers), not events
valid_tid = np.isin(tid_all, list(threads.keys())) & (tid_all != 0)
mkeys = np.array(sorted(methods.keys()), dtype=np.uint32)
valid_shift = np.isin(dex_shift, mkeys)
valid_mask = np.isin(ma_all & np.uint32(0xFFFFFFFC), mkeys)
# Method keys are small sequential indexes (mask low 2 bits); ancient files
# reportedly used pointer>>2. Pick whichever keys hit the table, warn if low.
if valid_mask.mean() >= valid_shift.mean():
    dex, valid_dex = (ma_all & np.uint32(0xFFFFFFFC)), valid_mask
    print('keying: mask (index|action)', flush=True)
else:
    dex, valid_dex = dex_shift, valid_shift
    print('keying: shift (pointer>>2|action)', flush=True)
del ma_all, dex_shift
print('valid tid frac:', valid_tid.mean(), 'valid dex frac:', valid_dex.mean(), flush=True)
data_mask = tid_all != 0  # tid==0 chunks are special records, not events
data_dex_frac = valid_dex[data_mask].mean() if data_mask.any() else 1.0
print('data records:', int(data_mask.sum()), 'dex hit among data:', data_dex_frac, flush=True)
if data_dex_frac < 0.9:
    print('WARNING: low dex hit rate, attributions suspect', flush=True)
print('action histogram:', np.bincount(act, minlength=4), flush=True)

np.savez_compressed(OUT,
                     tid=tid_all, dex=dex, act=act, ts=ts_all,
                     valid=valid_tid & valid_dex)
print('saved', OUT, flush=True)
