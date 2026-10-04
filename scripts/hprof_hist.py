#!/usr/bin/env python3
"""Shallow heap histogram from an HPROF (heap dump) file.

Parses STRING / LOAD_CLASS / CLASS_DUMP / INSTANCE_DUMP /
OBJECT_ARRAY_DUMP / PRIMITIVE_ARRAY_DUMP records and prints per-class
instance counts and shallow bytes. Retained (dominator) analysis is out
of scope — use Android Studio / MAT for that.

usage:
    python3 scripts/hprof_hist.py --db capture.hprof [--top 30] [--filter substr]
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
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

HEAP_DUMP = 0x0C
HEAP_DUMP_SEGMENT = 0x1C
CLASS_DUMP = 0x20
INSTANCE_DUMP = 0x21
OBJECT_ARRAY_DUMP = 0x22
PRIMITIVE_ARRAY_DUMP = 0x23

PRIM_SIZES = {4: 1, 5: 2, 6: 4, 7: 8, 8: 1, 9: 2, 10: 4, 11: 8}


def field_size(t, id_size):
    """Size of a constant/static field value of ART hprof type tag t."""
    if t == 2:
        return id_size
    return PRIM_SIZES.get(t, 4)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True, help='path to the .hprof file')
    parser.add_argument('--top', type=int, default=30,
                        help='rows to print (default: 30)')
    parser.add_argument('--filter', default=None,
                        help='only classes containing this substring')
    parser.add_argument('--workers', type=int, default=0,
                        help='parallel workers (default: CPU count; 1 = serial)')
    return parser.parse_args(argv)


def u32_at(buf, off):
    return struct.unpack_from('>I', buf, off)[0]


def default_workers():
    try:
        explicit = int(os.environ.get('WORKERS', '0'))
        if explicit > 0:
            return explicit
    except ValueError:
        pass
    return os.cpu_count() or 4


# Fork-inherited shard inputs (set by the parent before Pool creation; the
# 127 MB file is read once and workers see it CoW). Job args are just
# (shard_id, lo, hi) span-index ranges.
_HDATA = None
_HSPANS = None
_HID_SIZE = 4


def walk_span_range(data, spans, id_size, lo, hi):
    """Walk heap spans [lo, hi); returns (counts, total_bytes, skipped).

    Pure function of its arguments (workers call it on inherited globals).
    counts maps class_id -> [instances, bytes]; primitive arrays key by
    -etype. skipped counts segments aborted on unknown sub-records.
    """
    counts = {}
    total = [0]

    def bump(cid, nbytes):
        e = counts.get(cid)
        if e is None:
            counts[cid] = [1, nbytes]
        else:
            e[0] += 1
            e[1] += nbytes
        total[0] += nbytes

    def walk_segment(body, bend):
        """Walk one heap segment; True if fully walked, False on unknown tag."""
        p = body
        while p < bend:
            sub = data[p]
            if sub == CLASS_DUMP:
                # Stepped for framing only; sizes come from instance records.
                p += 1
                p += id_size
                p += 4 + id_size * 4  # stack, superclass, loader, signer, prot domain
                p += id_size * 2  # two reserved ids
                p += 4  # instance size
                # constant pool
                ncp = struct.unpack_from('>H', data, p)[0]
                p += 2
                for _ in range(ncp):
                    p += 2  # index
                    t = data[p]
                    p += 1
                    p += field_size(t, id_size)
                # static fields
                nsf = struct.unpack_from('>H', data, p)[0]
                p += 2
                for _ in range(nsf):
                    p += id_size  # name id
                    t = data[p]
                    p += 1
                    p += field_size(t, id_size)
                # instance fields: name id + type each
                nif = struct.unpack_from('>H', data, p)[0]
                p += 2 + nif * (id_size + 1)
            elif sub == INSTANCE_DUMP:
                p += 1 + id_size + 4  # obj id + stack
                cid = u32_at(data, p) if id_size == 4 else int.from_bytes(data[p:p + 8], 'big')
                p += id_size
                # ART reserves a u4 length here (0x77777777 placeholder,
                # patched with the real size): step AND count by it, so no
                # class-size map (and no forward-reference problem) is needed.
                size = u32_at(data, p)
                p += 4
                bump(cid, size)
                p += size
            elif sub == OBJECT_ARRAY_DUMP:
                p += 1 + id_size + 4  # obj id + stack
                cnt = u32_at(data, p)  # num elements
                p += 4
                cid = u32_at(data, p) if id_size == 4 else int.from_bytes(data[p:p + 8], 'big')
                p += id_size  # array class id
                bump(cid, cnt * id_size)
                p += cnt * id_size
            elif sub == PRIMITIVE_ARRAY_DUMP:
                p += 1 + id_size + 4
                p += 4
                cnt = u32_at(data, p - 4)
                etype = data[p]
                p += 1
                esize = PRIM_SIZES.get(etype, 1)
                    # primitive arrays share one synthetic class per type; key by -etype
                bump(-etype, cnt * esize)
                p += cnt * esize
            else:
                at = p
                p = skip_root(data, p, bend, id_size)
                if p < 0:
                    print(f'unknown heap sub-record 0x{sub:02x} at {at}, stopping segment', flush=True)
                    return False
        return True

    skipped = 0
    for body, bend in spans[lo:hi]:
        if not walk_segment(body, bend):
            skipped += 1
    return counts, total[0], skipped


def _walk_shard(shard_id, lo, hi):
    """Worker entry: pure function of (shard_id, lo, hi) on fork globals."""
    assert _HDATA is not None and _HSPANS is not None
    return walk_span_range(_HDATA, _HSPANS, _HID_SIZE, lo, hi)


def _split_spans(spans, n_workers):
    """Split span indexes into contiguous ranges with ~equal byte counts."""
    sizes = [e - s for s, e in spans]
    total = sum(sizes)
    if n_workers <= 1 or len(spans) <= 1 or total == 0:
        return [(0, len(spans))]
    target = max(1, total // n_workers)
    ranges, lo, acc = [], 0, 0
    for i, sz in enumerate(sizes):
        acc += sz
        if acc >= target and lo < i and len(ranges) < n_workers - 1:
            ranges.append((lo, i + 1))
            lo, acc = i + 1, 0
    ranges.append((lo, len(spans)))
    return ranges


def scan_top_level(data, id_size):
    """Scan top-level records; returns (strings, load_classes, heap_spans, n_heap).

    Single cheap pass; tables are inherited CoW by forked workers.
    """
    strings = {}
    load_classes = {}  # class_id -> name string id
    off, n = 31, len(data)
    # Records don't always start right after the fixed header (seen: 17
    # mystery bytes, likely an OEM header extension). Find the first offset
    # that chains several valid records instead of assuming one.
    VALID_TAGS = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                  0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x1C}
    found = -1
    for start in range(off, min(off + 64, n)):
        o, ok = start, 0
        while ok < 4:
            if o + 9 > n:
                break
            if data[o] not in VALID_TAGS:
                break
            ln = u32_at(data, o + 5)
            if ln > n:
                break
            ok += 1
            o += 9 + ln
        if ok >= 4:
            found = start
            break
    if found < 0:
        sys.exit('no chained records found after header')
    if found != off:
        print(f'first record at {found} ({found - off} header extension bytes)', flush=True)
    off = found
    n_top = n_heap = 0
    heap_spans = []
    while off + 9 <= n:
        tag = data[off]
        length = u32_at(data, off + 5)
        body, bend = off + 9, off + 9 + length
        if bend > n:
            print(f'truncated record at {off}, stopping', flush=True)
            break
        if tag == 0x01:  # STRING: id + bytes
            if id_size == 4:
                sid = u32_at(data, body)
                strings[sid] = data[body + 4:bend].decode('utf-8', 'replace')
            else:
                sid = int.from_bytes(data[body:body + 8], 'big')
                strings[sid] = data[body + 8:bend].decode('utf-8', 'replace')
        elif tag == 0x02:  # LOAD_CLASS: serial, class_id, stack, name_id
            p = body + 4
            cid = u32_at(data, p) if id_size == 4 else int.from_bytes(data[p:p + 8], 'big')
            p += id_size
            p += 4  # stack trace serial
            nid = u32_at(data, p) if id_size == 4 else int.from_bytes(data[p:p + 8], 'big')
            load_classes[cid] = nid
        elif tag in (HEAP_DUMP, HEAP_DUMP_SEGMENT):
            n_heap += 1
            heap_spans.append((body, bend))
        off = bend
    return strings, load_classes, heap_spans, n_heap


def run(args):
    with open(args.db, 'rb') as f:
        data = f.read()
    print(f'file size: {len(data)}', flush=True)
    if data[:18] != b'JAVA PROFILE 1.0.3' or data[18] != 0:
        sys.exit('not an HPROF file')
    id_size = u32_at(data, 19)
    if id_size not in (4, 8):
        sys.exit(f'unsupported id size: {id_size}')
    print(f'id size: {id_size}', flush=True)

    strings = {}
    load_classes = {}  # class_id -> name string
    # NOTE: per-class (counts, bytes) are produced by the parallel heap-span
    # walk below (merged in shard order) into `counts` / `total_bytes`.

    strings, load_classes, heap_spans, n_heap = scan_top_level(data, id_size)

    # --- parallel heap walk, sharded by top-level heap-dump spans ---
    # strings/load_classes are built above (single cheap pass) and inherited
    # CoW; workers only return small (counts, bytes) dicts for the parent
    # to merge in shard order (deterministic = serial-identical).
    if os.name != 'posix':
        sys.exit('hprof_hist.py parallel walk needs fork (Unix only)')
    global _HDATA, _HSPANS, _HID_SIZE
    _HDATA, _HSPANS, _HID_SIZE = data, heap_spans, id_size
    import multiprocessing as mp
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    n_workers = args.workers or default_workers()
    ranges = _split_spans(heap_spans, n_workers)
    jobs = [(sid, lo, hi) for sid, (lo, hi) in enumerate(ranges)]
    if len(jobs) <= 1 or n_workers <= 1:
        parts = [_walk_shard(sid, lo, hi) for sid, lo, hi in jobs]
    else:
        with mp.Pool(min(len(jobs), n_workers)) as pool:
            parts = pool.starmap(_walk_shard, jobs)
    counts = {}
    total_bytes = 0
    skipped = 0
    for part_counts, part_total, part_skipped in parts:
        total_bytes += part_total
        skipped += part_skipped
        for cid in sorted(part_counts):
            cnt, b = part_counts[cid]
            e = counts.get(cid)
            if e is None:
                counts[cid] = [cnt, b]
            else:
                e[0] += cnt
                e[1] += b
    del parts
    if skipped:
        print(f'{skipped} segments skipped (unknown tags)', flush=True)
    print(f'top-level heap dumps: {n_heap}, heap bytes counted: {total_bytes}', flush=True)

    def name_of(cid):
        if cid < 0:
            return f'primitive-array[{cid}]'
        nid = load_classes.get(cid)
        s = strings.get(nid, f'?id={cid}') if nid is not None else f'?class={cid}'
        return s.replace('/', '.')

    rows = sorted(counts.items(), key=lambda kv: -kv[1][1])
    if args.filter:
        rows = [(c, v) for c, v in rows if args.filter in name_of(c)]
    print(f'{"instances":>10} {"bytes":>12}  class', flush=True)
    for cid, (cnt, b) in rows[:args.top]:
        print(f'{cnt:10d} {b:12d}  {name_of(cid)}', flush=True)


ROOT_SIZES = None


def skip_root(data, p, bend, id_size):
    """Skip a GC-root sub-record at p; returns new offset or -1 if unknown.

    Layouts from platform/art/runtime/hprof/hprof.cc (Hprof::MarkRootObject):
    single ids, JNI global (2 ids), local/monitor/java-frame (id + 2x u4),
    native-stack/thread-block (id + u4), thread-object (id + 2x u4),
    HEAP_DUMP_INFO (u4 heap type + string id).
    """
    sub = data[p]
    if sub == 0xFE:  # HEAP_DUMP_INFO: u4 heap type + string id
        return p + 1 + 4 + id_size
    table = {
        0xFF: (1, 0), 0x05: (1, 0), 0x07: (1, 0),
        0x89: (1, 0), 0x8B: (1, 0), 0x8D: (1, 0),
        0x01: (2, 0),
        0x02: (1, 8), 0x8E: (1, 8), 0x03: (1, 8),
        0x04: (1, 4), 0x06: (1, 4),
        0x08: (1, 8),
    }
    spec = table.get(sub)
    if spec is None:
        return -1
    nids, extra = spec
    return p + 1 + nids * id_size + extra


def main(argv=None):
    args = parse_args(argv)
    run(args)


if __name__ == '__main__':
    main()
