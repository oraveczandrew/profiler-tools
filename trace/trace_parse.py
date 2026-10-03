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

Parallelism: the thread/method table scans stay single (C-speed regex/find),
while gap-decode + all three `isin` validations shard by gap-span ranges
across forked workers. The parent mmaps the file once; workers inherit it
CoW and never re-read. Only `(shard_id, lo, hi)` gap-index tuples cross the
process boundary; sharded arrays come back and are concatenated in gap
order, so output arrays are identical for any worker count. The final
`savez_compressed` stays single (zip compression is inherently serial).

usage:
    python3 trace/trace_parse.py <cpu-art-*.trace> [--workers N]
"""
import mmap
import re
import struct
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from stackwalk import default_workers


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


# Fork-inherited shard inputs (set by the parent before Pool creation;
# workers must NOT re-read the file). Job args are just (shard_id, lo, hi)
# gap-index ranges, per the run_parallel pattern.
_DATA = None
_GAPS = None
_THREAD_KEYS = None
_MKEYS = None


def _parse_shard(shard_id, lo, hi):
    """Decode gaps [lo, hi) + validate; pure function of (shard_id, lo, hi)."""
    assert _DATA is not None and _GAPS is not None
    assert _THREAD_KEYS is not None and _MKEYS is not None
    data, gaps = _DATA, _GAPS
    tid_parts, ma_parts, ts_parts = [], [], []
    for gi in range(lo, hi):
        s, e = gaps[gi]
        ln = (e - s) // 10 * 10
        if not ln:
            continue
        w16 = np.frombuffer(data, dtype=np.uint16, count=ln // 2, offset=s).reshape(-1, 5)
        tid_parts.append(w16[:, 0].astype(np.uint32))
        ma_parts.append(w16[:, 1].astype(np.uint32) | (w16[:, 2].astype(np.uint32) << 16))
        ts_parts.append(w16[:, 3].astype(np.uint64) | (w16[:, 4].astype(np.uint64) << 16))
        del w16
    if not tid_parts:
        z = np.empty(0, dtype=np.uint32)
        return {'tid': z, 'ma': z.copy(), 'ts': np.empty(0, dtype=np.uint64),
                'act': np.empty(0, dtype=np.uint8),
                'tid_valid': np.empty(0, dtype=bool),
                'mask_valid': np.empty(0, dtype=bool),
                'shift_valid': np.empty(0, dtype=bool),
                'n': 0, 'mask_hits': 0, 'shift_hits': 0}
    tid_s = np.concatenate(tid_parts)
    ma_s = np.concatenate(ma_parts)
    ts_s = np.concatenate(ts_parts)
    del tid_parts, ma_parts, ts_parts
    act_s = (ma_s & np.uint32(3)).astype(np.uint8)
    tid_valid_s = np.isin(tid_s, _THREAD_KEYS) & (tid_s != 0)
    mask_valid_s = np.isin(ma_s & np.uint32(0xFFFFFFFC), _MKEYS)
    shift_valid_s = np.isin(ma_s >> np.uint32(2), _MKEYS)
    return {'tid': tid_s, 'ma': ma_s, 'ts': ts_s, 'act': act_s,
            'tid_valid': tid_valid_s, 'mask_valid': mask_valid_s,
            'shift_valid': shift_valid_s, 'n': len(tid_s),
            'mask_hits': int(mask_valid_s.sum()), 'shift_hits': int(shift_valid_s.sum())}


def _split_gaps(gaps, n_workers):
    """Split gap indexes into contiguous ranges with ~equal record counts."""
    counts = [((e - s) // 10) for s, e in gaps]
    total = sum(counts)
    if n_workers <= 1 or len(gaps) <= 1 or total == 0:
        return [(0, len(gaps))]
    target = max(1, total // n_workers)
    ranges, lo, acc = [], 0, 0
    for gi, c in enumerate(counts):
        acc += c
        if acc >= target and lo < gi and len(ranges) < n_workers - 1:
            ranges.append((lo, gi + 1))
            lo, acc = gi + 1, 0
    ranges.append((lo, len(gaps)))
    return ranges


def parse_argv(argv):
    workers = 0
    rest = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--workers' and i + 1 < len(argv):
            try:
                workers = int(argv[i + 1])
            except ValueError:
                sys.exit('--workers needs an integer')
            i += 2
        elif a.startswith('--workers='):
            try:
                workers = int(a.split('=', 1)[1])
            except ValueError:
                sys.exit('--workers needs an integer')
            i += 1
        else:
            rest.append(a)
            i += 1
    if not rest or rest[0].startswith('--'):
        sys.exit(f'usage: {os.path.basename(sys.argv[0])} <cpu-art-*.trace> [--workers N]')
    return rest[0], workers


def main(argv=None):
    if os.name != 'posix':
        sys.exit('trace_parse.py parallel decode needs fork (Unix only)')
    args = sys.argv[1:] if argv is None else argv
    path, workers_arg = parse_argv(args)
    out = os.path.splitext(path)[0] + '.parsed.npz'

    f = open(path, 'rb')
    data = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    print('file size:', len(data), flush=True)
    if not len(data):
        sys.exit('empty trace file (still recording?)')

    # --- threads: full-file scan (late-created threads are recorded mid-stream) ---
    # Grammar: `02 <id:u16> <len:u16> <name>` core, optionally followed by a `00 00`
    # separator when another special (thread/method) record follows immediately.
    # When DATA follows, there is no separator at all.
    threads = {}
    tspans = []
    pos = data.find(b'\x02')
    while pos != -1:
        r = thread_at(data, pos)
        if r is not None:
            t, name, ln = r
            threads.setdefault(t, []).append(name)
            tspans.append((pos, pos + ln))
            pos = data.find(b'\x02', pos + ln)
        else:
            pos = data.find(b'\x02', pos + 1)
    print('thread records:', len(tspans), 'unique tids:', len(threads), flush=True)
    for t in sorted(threads):
        print(f'  tid={t:<4d} {threads[t]}', flush=True)

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

    # --- parallel gap-decode + isin validation, sharded by gap spans ---
    # dtypes are minimal: tid/ma u32, ts u64 (wall clock exceeds u32), act u8.
    global _DATA, _GAPS, _THREAD_KEYS, _MKEYS
    _DATA, _GAPS = data, gaps
    _THREAD_KEYS = sorted(threads.keys())
    _MKEYS = np.array(sorted(methods.keys()), dtype=np.uint32)

    import multiprocessing as mp
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    n_workers = workers_arg or default_workers()
    ranges = _split_gaps(gaps, n_workers)
    jobs = [(sid, lo, hi) for sid, (lo, hi) in enumerate(ranges)]
    if len(jobs) <= 1 or n_workers <= 1:
        parts = [_parse_shard(sid, lo, hi) for sid, lo, hi in jobs]
    else:
        with mp.Pool(min(len(jobs), n_workers)) as pool:
            parts = pool.starmap(_parse_shard, jobs)
    # Gap ranges are contiguous and ordered: concatenation preserves file order.
    n_records = sum(p['n'] for p in parts)
    print('records:', n_records, flush=True)
    tid_all = np.concatenate([p['tid'] for p in parts]) if n_records else np.empty(0, dtype=np.uint32)
    ma_all = np.concatenate([p['ma'] for p in parts]) if n_records else np.empty(0, dtype=np.uint32)
    ts_all = np.concatenate([p['ts'] for p in parts]) if n_records else np.empty(0, dtype=np.uint64)
    act = np.concatenate([p['act'] for p in parts]) if n_records else np.empty(0, dtype=np.uint8)
    valid_tid = np.concatenate([p['tid_valid'] for p in parts]) if n_records else np.empty(0, dtype=bool)
    mask_valid = np.concatenate([p['mask_valid'] for p in parts]) if n_records else np.empty(0, dtype=bool)
    shift_valid = np.concatenate([p['shift_valid'] for p in parts]) if n_records else np.empty(0, dtype=bool)
    mask_hits = sum(p['mask_hits'] for p in parts)
    shift_hits = sum(p['shift_hits'] for p in parts)
    for p in parts:
        for k in ('tid', 'ma', 'ts', 'act', 'tid_valid', 'mask_valid', 'shift_valid'):
            del p[k]
    del parts

    # Method keys are small sequential indexes (mask low 2 bits); ancient files
    # reportedly used pointer>>2. Pick whichever keys hit the table, warn if low.
    # Integer hit-count compare == mean compare (same denominator), so this
    # matches the serial decision exactly.
    if mask_hits >= shift_hits:
        dex = (ma_all & np.uint32(0xFFFFFFFC))
        valid_dex = mask_valid
        print('keying: mask (index|action)', flush=True)
    else:
        dex = ma_all >> np.uint32(2)
        valid_dex = shift_valid
        print('keying: shift (pointer>>2|action)', flush=True)
    del ma_all
    del mask_valid, shift_valid
    # tid==0 chunks are special records (method/thread/summary markers), not events
    print('valid tid frac:', valid_tid.mean(), 'valid dex frac:', valid_dex.mean(), flush=True)
    data_mask = tid_all != 0  # tid==0 chunks are special records, not events
    data_dex_frac = valid_dex[data_mask].mean() if data_mask.any() else 1.0
    print('data records:', int(data_mask.sum()), 'dex hit among data:', data_dex_frac, flush=True)
    if data_dex_frac < 0.9:
        print('WARNING: low dex hit rate, attributions suspect', flush=True)
    print('action histogram:', np.bincount(act, minlength=4), flush=True)

    # --- precomputed tables for downstream scripts (additive .npz keys) ---
    # analyze/deep/parents need the method-name table, the thread table and
    # the data-gap boundaries; all are known here after the full-file scan,
    # so they are cached in the .npz and downstream scripts never re-read
    # the multi-GB .trace. Existing keys/dtypes are untouched (old readers
    # ignore extra keys; missing keys are a hard error downstream).
    n_methods = max(methods) + 1 if methods else 0
    names = [methods.get(i, '') for i in range(n_methods)]
    sorted_tids = sorted(threads)
    np.savez_compressed(out,
                        tid=tid_all, dex=dex, act=act, ts=ts_all,
                        valid=valid_tid & valid_dex,
                        names=np.array(names, dtype=object),
                        thread_ids=np.array(sorted_tids, dtype=np.int64),
                        thread_names=np.array([threads[t][0] for t in sorted_tids],
                                              dtype=object),
                        gaps=np.array(gaps, dtype=np.int64) if gaps
                        else np.empty((0, 2), dtype=np.int64))
    print('saved', out, flush=True)
    data.close()
    f.close()


if __name__ == '__main__':
    main()
