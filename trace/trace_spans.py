"""Pair atrace B/E sections from a raw Perfetto trace (no trace_processor needed).

The kernel ftrace print buffer carries userspace markers as
``B|<pid>|<name>`` (begin) and ``E`` (end). This script walks the raw
protobuf with a dependency-free field walker, pairs begins with ends
per writer thread, and prints per-span durations plus a work-vs-wall
summary (handy for single-thread vs parallel questions).

Layout assumptions (verified against Perfetto v50 traces; field numbers
are stable wire identifiers, not names):
  Trace.packet = 1 -> raw ftrace bundle = 1 -> event = 2 ->
  FtraceEvent{timestamp = 1, pid = 2, print = 3} -> Print{buf = 2}

usage:
    python3 trace/trace_spans.py <perfetto.trace> [NAME] [--tids TID,...]

    NAME limits output to one section (exact match); without NAME the
    script lists distinct section names instead. --tids restricts pairing
    to writer tids. Timestamps are raw ftrace nanoseconds; durations are
    also shown in milliseconds.
"""
import sys
from collections import defaultdict


def read_varint(buf, pos):
    res = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        res |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return res, pos


def parse(buf):
    """Minimal protobuf walk: {field_number: [(wire_type, value)]}."""
    pos = 0
    n = len(buf)
    out = {}
    while pos < n:
        key, pos = read_varint(buf, pos)
        f, w = key >> 3, key & 7
        if w == 0:
            v, pos = read_varint(buf, pos)
        elif w == 1:
            v = bytes(buf[pos:pos + 8])
            pos += 8
        elif w == 2:
            ln, pos = read_varint(buf, pos)
            v = bytes(buf[pos:pos + ln])
            pos += ln
        elif w in (4, 5):
            # fixed32 (wire 4 is not valid protobuf, but tolerate it
            # inside unknown ftrace payloads instead of dying)
            v = bytes(buf[pos:pos + 4])
            pos += 4
        else:
            raise ValueError((f, w))
        out.setdefault(f, []).append((w, v))
    return out


def iter_prints(top):
    """Yield (ts_ns, writer_tid, text) for every ftrace print line."""
    for _w, v in top.get(1, []):
        try:
            pkt = parse(v)
        except ValueError:
            continue
        if 1 not in pkt:
            continue
        for _ww, bundle in pkt[1]:
            try:
                b = parse(bundle)
            except ValueError:
                continue
            for _ew, ev in b.get(2, []):
                try:
                    e = parse(ev)
                except ValueError:
                    continue
                if 1 not in e or 2 not in e or 3 not in e:
                    continue
                ts = e[1][0][1]
                pid = e[2][0][1]
                for _bw, print_msg in e[3]:
                    try:
                        inner = parse(bytes(print_msg))
                    except ValueError:
                        continue
                    for _cw, buf in inner.get(2, []):
                        yield ts, pid, bytes(buf).rstrip(b'\n').decode('utf-8', 'replace')


def collect(path, name_filter=None, tids=None):
    data = open(path, 'rb').read()
    top = parse(data)
    stacks = defaultdict(list)  # writer tid -> [(name, begin_ts)]
    spans = []  # (name, tid, begin_ts, end_ts)
    b_count = 0
    names = defaultdict(int)
    for ts, pid, s in iter_prints(top):
        if s.startswith('B|'):
            parts = s.split('|', 2)
            if len(parts) != 3:
                continue
            secname = parts[2]
            names[secname] += 1
            if (name_filter is None or secname == name_filter) and (tids is None or pid in tids):
                stacks[pid].append((secname, ts))
                b_count += 1
        elif s == 'E' or s.startswith('E|'):
            if stacks.get(pid):
                secname, bts = stacks[pid].pop()
                spans.append((secname, pid, bts, ts))
    unclosed = sum(len(v) for v in stacks.values())
    return spans, b_count, unclosed, names


def main(argv):
    if not argv or argv[0].startswith('-'):
        sys.exit('usage: trace_spans.py <perfetto.trace> [NAME] [--tids TID,...]')
    path = argv[0]
    name_filter = None
    tids = None
    for a in argv[1:]:
        if a.startswith('--tids='):
            tids = {int(x) for x in a.split('=', 1)[1].split(',') if x}
        elif not a.startswith('-') and name_filter is None:
            name_filter = a
        else:
            sys.exit(f'unknown arg: {a}')
    spans, b_count, unclosed, names = collect(path, name_filter, tids)
    if name_filter is None:
        print(f'--- distinct section names: {len(names)} ---')
        for k, v in sorted(names.items(), key=lambda x: -x[1])[:60]:
            print(f'{v:6d}  {k[:100]}')
        return
    print(f'B events for {name_filter}: {b_count}, paired spans: {len(spans)}, unclosed: {unclosed}')
    for secname, pid, bts, ets in sorted(spans, key=lambda x: x[2]):
        print(f'{secname} tid={pid} begin={bts} end={ets} dur_ns={ets - bts} dur_ms={(ets - bts) / 1e6:.3f}')
    if spans:
        total = sum(ets - bts for _, _, bts, ets in spans)
        wall = max(ets for _, _, _, ets in spans) - min(bts for _, _, bts, _ in spans)
        print(f'TOTAL work_ms={total / 1e6:.1f} wall_ms={wall / 1e6:.1f} speedup={total / wall:.2f}x')


if __name__ == '__main__':
    main(sys.argv[1:])
