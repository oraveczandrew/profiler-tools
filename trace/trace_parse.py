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

# --- parse records ---
bufs = []
for s, e in gaps:
    ln = (e - s) // 10 * 10
    if ln:
        bufs.append(data[s:s + ln])
blob = b''.join(bufs)
w16 = np.frombuffer(blob, dtype=np.uint16).reshape(-1, 5)
tid16 = w16[:, 0].astype(np.int64)
ma = (w16[:, 1].astype(np.int64) | (w16[:, 2].astype(np.int64) << 16))
ts = (w16[:, 3].astype(np.int64) | (w16[:, 4].astype(np.int64) << 16))
print('records:', len(w16), flush=True)

dex_shift = ma >> 2
act = ma & 3
# tid==0 chunks are special records (method/thread/summary markers), not events
valid_tid = np.isin(tid16, list(threads.keys())) & (tid16 != 0)
mset = set(methods.keys())
valid_shift = np.array([d in mset for d in dex_shift])
valid_mask = np.array([(d & ~3) in mset for d in ma])
# Method keys are small sequential indexes (mask low 2 bits); ancient files
# reportedly used pointer>>2. Pick whichever keys hit the table, warn if low.
if valid_mask.mean() >= valid_shift.mean():
    dex, valid_dex = (ma & ~3), valid_mask
    print('keying: mask (index|action)', flush=True)
else:
    dex, valid_dex = dex_shift, valid_shift
    print('keying: shift (pointer>>2|action)', flush=True)
print('valid tid frac:', valid_tid.mean(), 'valid dex frac:', valid_dex.mean(), flush=True)
data_mask = tid16 != 0  # tid==0 chunks are special records, not events
data_dex_frac = valid_dex[data_mask].mean() if data_mask.any() else 1.0
print('data records:', int(data_mask.sum()), 'dex hit among data:', data_dex_frac, flush=True)
if data_dex_frac < 0.9:
    print('WARNING: low dex hit rate, attributions suspect', flush=True)
print('action histogram:', np.bincount(act, minlength=4), flush=True)

np.savez_compressed(OUT,
                     tid=tid16, dex=dex, act=act, ts=ts,
                     valid=valid_tid & valid_dex)
print('saved', OUT, flush=True)
