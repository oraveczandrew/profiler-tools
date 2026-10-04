#!/usr/bin/env python3
"""Who holds instances of one class alive (incoming references) in an .hprof.

Shallow histograms show *what* is retained; this shows *by whom*: for every
instance of --target it aggregates the referring objects by their class, plus
one example referrer pair per class and (with --chain) one example chain
toward a GC root. Full MAT-style dominators are deliberately out of scope.

usage:
    python3 scripts/hprof_referrers.py --db capture.hprof \\
        --target PaintConfiguration [--chain 5] [--workers 16]

Two streaming passes over the heap (layouts+targets, then edges), so memory
stays flat regardless of heap size. Chain hops are extra passes, one per hop.
"""
#     Copyright 2026 András Oravecz <info@oandras.hu>
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hprof_hist import (CLASS_DUMP, HEAP_DUMP, HEAP_DUMP_SEGMENT,
                        INSTANCE_DUMP, OBJECT_ARRAY_DUMP, PRIM_SIZES,
                        _split_spans, default_workers, field_size, scan_top_level,
                        skip_root, u32_at)

PRIMITIVE_ARRAY_DUMP = 0x23

# hprof type tags by primitive array descriptor.
_PRIM_ETYPE = {'[Z': 4, '[C': 5, '[F': 6, '[D': 7,
               '[B': 8, '[S': 9, '[I': 10, '[J': 11}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', required=True, help='path to the .hprof file')
    parser.add_argument('--target', required=True,
                        help='class name substring, e.g. PaintConfiguration')
    parser.add_argument('--chain', type=int, default=0,
                        help='example chain hops toward a root (default: 0 = off)')
    parser.add_argument('--workers', type=int, default=0,
                        help='parallel workers (default: CPU count; 1 = serial)')
    return parser.parse_args(argv)


def read_id(data, p, id_size):
    if id_size == 4:
        return u32_at(data, p)
    return int.from_bytes(data[p:p + 8], 'big')


def collect_layouts(data, spans, id_size, strings):
    """Serial framing walk; returns raw {class_id: [super_id, own_fields,
    statics]} where own_fields/static entries are (name_id, type[, value]).

    Only CLASS_DUMP bodies are parsed in detail; everything else is stepped
    over by its framed size. Synthetic `$class$...` static fields (ART VM
    metadata: vtable, iftable, shadow klass/monitor) are dropped: they point
    at VM structures, never at app objects.
    """
    raw = {}
    skipped = 0
    for body, bend in spans:
        p = body
        ok = True
        while p < bend:
            sub = data[p]
            if sub == CLASS_DUMP:
                q = p + 1
                cid = read_id(data, q, id_size)
                q += id_size
                q += 4  # stack
                super_id = read_id(data, q, id_size)
                q += id_size
                q += id_size * 3  # loader, signer, prot domain
                q += id_size * 2  # reserved
                q += 4  # instance size
                ncp = struct.unpack_from('>H', data, q)[0]
                q += 2
                for _ in range(ncp):
                    q += 2
                    t = data[q]
                    q += 1
                    q += field_size(t, id_size)
                nsf = struct.unpack_from('>H', data, q)[0]
                q += 2
                statics = []
                for _ in range(nsf):
                    nid = read_id(data, q, id_size)
                    q += id_size
                    t = data[q]
                    q += 1
                    if t == 2:
                        if not strings.get(nid, '').startswith('$class$'):
                            statics.append((nid, read_id(data, q, id_size)))
                    q += field_size(t, id_size)
                nif = struct.unpack_from('>H', data, q)[0]
                q += 2
                own = []
                for _ in range(nif):
                    nid = read_id(data, q, id_size)
                    q += id_size
                    t = data[q]
                    q += 1
                    own.append((nid, t))
                if cid not in raw:
                    raw[cid] = [super_id, own, statics]
                p = q
            elif sub == INSTANCE_DUMP:
                q = p + 1 + id_size + 4
                q += id_size  # class id
                size = u32_at(data, q)
                p = q + 4 + size
            elif sub == OBJECT_ARRAY_DUMP:
                q = p + 1 + id_size + 4
                cnt = u32_at(data, q)
                q += 4 + id_size
                p = q + cnt * id_size
            elif sub == PRIMITIVE_ARRAY_DUMP:
                q = p + 1 + id_size + 4
                cnt = u32_at(data, q)
                q += 4
                esize = PRIM_SIZES.get(data[q], 1)
                p = q + 1 + cnt * esize
            else:
                at = p
                p = skip_root(data, p, bend, id_size)
                if p < 0:
                    print(f'unknown heap sub-record 0x{sub:02x} at {at},'
                          ' stopping segment', flush=True)
                    skipped += 1
                    ok = False
                    break
        if not ok:
            continue
    return raw, skipped


def full_layouts(raw, id_size):
    """Resolve field order: {class_id: (ref_offsets, off_to_name, data_size)}.

    ref_offsets = [(byte_offset, name_id)] for reference-typed fields only.
    Verified against ground truth (ArrayList/size-constraint over all 3 659
    instances, PathRenderNode total == measured): ART stores subclass-own
    fields FIRST in CLASS_DUMP declaration order, then the superclass block,
    with java.lang.Object's klass/monitor words LAST — i.e. leaf-to-root,
    the reverse of the JVM-spec order.
    """
    memo = {}

    def size_of(t):
        return id_size if t == 2 else PRIM_SIZES.get(t, 4)

    def build(cid, seen):
        if cid in memo:
            return memo[cid]
        entry = raw.get(cid)
        if entry is None or cid in seen:
            memo[cid] = ([], {}, [], 0)
            return memo[cid]
        seen = seen | {cid}
        super_id, own, _statics = entry
        refs, names, off = [], {}, 0
        for nid, t in own:
            if t == 2:
                refs.append((off, nid))
                names[off] = nid
            off += size_of(t)
        statics = [(nid, v) for nid, v in _statics]
        if super_id in raw:
            s_refs, s_names, s_statics, s_size = build(super_id, seen)
            for o, n in s_refs:
                refs.append((off + o, n))
                names[off + o] = n
            off += s_size
        out = (refs, names, statics, off)
        memo[cid] = out
        return out

    return {cid: build(cid, frozenset()) for cid in raw}


# Fork-inherited shard inputs.
_RDATA = None
_RSPANS = None
_RID = 4
_RLAYOUTS = None
_RTARGETS = None
_RMODE = 'edges'
_RHOP = None


def _walk_shard(sid, lo, hi):
    """Shard walk; mode 'edges' aggregates referrers, mode 'hop' finds the
    first referrer of a single id. Returns small mergeable tuples."""
    data, spans, id_size = _RDATA, _RSPANS, _RID
    layouts, targets = _RLAYOUTS, _RTARGETS
    hop = _RMODE == 'hop'
    counts = {}
    examples = {}
    n_targets = 0
    hop_hit = None

    def note(referrer_cid, referrer_oid, target_oid, field):
        nonlocal hop_hit
        if hop:
            if hop_hit is None:
                hop_hit = (referrer_cid, referrer_oid, target_oid, field)
            return True  # found; caller stops the record, not the shard
        e = counts.get(referrer_cid)
        if e is None:
            counts[referrer_cid] = 1
            examples[referrer_cid] = (referrer_oid, target_oid, field)
        else:
            counts[referrer_cid] = e + 1
        return False

    def walk_segment(body, bend):
        p = body
        while p < bend:
            sub = data[p]
            if sub == CLASS_DUMP:
                q = p + 1
                cid = read_id(data, q, id_size)
                lay = layouts.get(cid)
                statics = lay[2] if lay is not None else []
                for nid, v in statics:
                    if v in targets:
                        if note(cid, cid, v, ('static', nid)):
                            return True
                # frame through the record generically
                q += id_size + 4 + id_size * 4 + id_size * 2 + 4
                ncp = struct.unpack_from('>H', data, q)[0]
                q += 2
                for _ in range(ncp):
                    q += 2
                    t = data[q]
                    q += 1
                    q += field_size(t, id_size)
                nsf = struct.unpack_from('>H', data, q)[0]
                q += 2
                for _ in range(nsf):
                    q += id_size
                    t = data[q]
                    q += 1
                    q += field_size(t, id_size)
                nif = struct.unpack_from('>H', data, q)[0]
                p = q + 2 + nif * (id_size + 1)
            elif sub == INSTANCE_DUMP:
                q = p + 1
                oid = read_id(data, q, id_size)
                q += id_size + 4
                cid = read_id(data, q, id_size)
                q += id_size
                size = u32_at(data, q)
                q += 4
                fbase = q
                lay = layouts.get(cid)
                if lay is not None:
                    refs = lay[0]
                    if refs and refs[-1][0] + id_size <= size:
                        for off, nid in refs:
                            v = read_id(data, fbase + off, id_size)
                            if v != 0 and v in targets:
                                if note(cid, oid, v, ('field', nid)):
                                    break
                p = fbase + size
            elif sub == OBJECT_ARRAY_DUMP:
                q = p + 1
                oid = read_id(data, q, id_size)
                q += id_size + 4
                cnt = u32_at(data, q)
                q += 4
                cid = read_id(data, q, id_size)
                q += id_size
                for i in range(cnt):
                    v = read_id(data, q, id_size)
                    q += id_size
                    if v != 0 and v in targets:
                        if note(cid, oid, v, ('index', i)):
                            break
                p = q
            elif sub == PRIMITIVE_ARRAY_DUMP:
                q = p + 1 + id_size + 4
                cnt = u32_at(data, q)
                q += 4
                esize = PRIM_SIZES.get(data[q], 1)
                p = q + 1 + cnt * esize
            else:
                at = p
                p = skip_root(data, p, bend, id_size)
                if p < 0:
                    return False
        return True

    stopped = 0
    for body, bend in spans[lo:hi]:
        if not walk_segment(body, bend):
            stopped += 1
        if hop and hop_hit is not None:
            break
    if _RMODE == 'targets':
        return None
    return counts, examples, stopped, hop_hit


def _run_shards(spans, n_workers):
    import multiprocessing as mp
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    ranges = _split_spans(spans, n_workers)
    jobs = [(sid, lo, hi) for sid, (lo, hi) in enumerate(ranges)]
    if len(jobs) <= 1 or n_workers <= 1:
        return [_walk_shard(sid, lo, hi) for sid, lo, hi in jobs]
    with mp.Pool(min(len(jobs), n_workers)) as pool:
        return pool.starmap(_walk_shard, jobs)


def run(args):
    global _RDATA, _RSPANS, _RID, _RLAYOUTS, _RTARGETS, _RMODE
    with open(args.db, 'rb') as f:
        data = f.read()
    print(f'file size: {len(data)}', flush=True)
    if data[:18] != b'JAVA PROFILE 1.0.3' or data[18] != 0:
        sys.exit('not an HPROF file')
    id_size = u32_at(data, 19)
    if id_size not in (4, 8):
        sys.exit(f'unsupported id size: {id_size}')

    strings, load_classes, heap_spans, n_heap = scan_top_level(data, id_size)
    print(f'heap spans: {len(heap_spans)}', flush=True)

    def name_of(cid):
        nid = load_classes.get(cid)
        s = strings.get(nid, f'?id={cid}') if nid is not None else f'?class={cid}'
        return s.replace('/', '.')

    matches = sorted(
        cid for cid, nid in load_classes.items()
        if args.target in strings.get(nid, '')
    )
    # Primitive arrays (e.g. '[I') have no LOAD_CLASS entry: match by etype.
    prim_etype = _PRIM_ETYPE.get(args.target)
    if prim_etype is not None:
        matches = []
    if not matches and prim_etype is None:
        sys.exit(f'no class matching {args.target!r}')
    if len(matches) > 1:
        # Class names use slashes but array names use dots
        # ('java/lang/Foo' vs 'java.lang.Object[]'): accept either spelling.
        dotted = args.target.replace('.', '/')
        exact = [c for c in matches
                 if strings.get(load_classes[c], '') in (args.target, dotted)]
        if len(exact) == 1:
            matches = exact
    if len(matches) > 1:
        outer = [c for c in matches
                 if '$' not in strings.get(load_classes[c], '')]
        if len(outer) == 1:
            matches = outer
    if not matches and prim_etype is None:
        sys.exit(f'no class matching {args.target!r}')
    if len(matches) > 1:
        names = [strings.get(load_classes[c], f'?{c}') for c in matches]
        sys.exit(f'ambiguous target {args.target!r}: {names}')
    target_cid = matches[0] if matches else None
    print(f'target: {args.target if prim_etype is not None else name_of(target_cid)}',
          flush=True)

    print('pass 1: layouts...', flush=True)
    raw, skipped = collect_layouts(data, heap_spans, id_size, strings)
    if skipped:
        print(f'{skipped} segments skipped in pass 1', flush=True)
    layouts = full_layouts(raw, id_size)
    print(f'classes with layouts: {len(layouts)}', flush=True)

    if os.name != 'posix':
        sys.exit('parallel walk needs fork (Unix only)')
    _RDATA, _RSPANS, _RID = data, heap_spans, id_size
    _RLAYOUTS = layouts
    n_workers = args.workers or default_workers()

    # Pass 2a: collect target instance ids (serial framing walk is enough;
    # instances are found by class id while stepping records).
    print('pass 2: target instances...', flush=True)
    target_ids = set()
    skipped = 0
    for body, bend in heap_spans:
        p = body
        ok = True
        while p < bend:
            sub = data[p]
            if sub == CLASS_DUMP:
                # frame only; layouts already captured
                q = p + 1 + id_size + 4 + id_size * 4 + id_size * 2 + 4
                ncp = struct.unpack_from('>H', data, q)[0]
                q += 2
                for _ in range(ncp):
                    q += 2
                    t = data[q]
                    q += 1
                    q += field_size(t, id_size)
                nsf = struct.unpack_from('>H', data, q)[0]
                q += 2
                for _ in range(nsf):
                    q += id_size
                    t = data[q]
                    q += 1
                    q += field_size(t, id_size)
                nif = struct.unpack_from('>H', data, q)[0]
                p = q + 2 + nif * (id_size + 1)
            elif sub == INSTANCE_DUMP:
                q = p + 1
                oid = read_id(data, q, id_size)
                q += id_size + 4
                cid = read_id(data, q, id_size)
                q += id_size
                size = u32_at(data, q)
                if cid == target_cid:
                    target_ids.add(oid)
                p = q + 4 + size
            elif sub == OBJECT_ARRAY_DUMP:
                q = p + 1
                aoid = read_id(data, q, id_size)
                q += id_size + 4
                cnt = u32_at(data, q)
                q += 4
                acid = read_id(data, q, id_size)
                q += id_size
                if acid == target_cid:
                    target_ids.add(aoid)
                p = q + cnt * id_size
            elif sub == PRIMITIVE_ARRAY_DUMP:
                q = p + 1
                parr = read_id(data, q, id_size)
                q += id_size + 4
                cnt = u32_at(data, q)
                q += 4
                etype = data[q]
                if prim_etype is not None and etype == prim_etype:
                    target_ids.add(parr)
                esize = PRIM_SIZES.get(etype, 1)
                p = q + 1 + cnt * esize
            else:
                at = p
                p = skip_root(data, p, bend, id_size)
                if p < 0:
                    print(f'unknown heap sub-record 0x{sub:02x} at {at},'
                          ' stopping segment', flush=True)
                    skipped += 1
                    ok = False
                    break
        if not ok:
            continue
    print(f'target instances: {len(target_ids)}', flush=True)
    if not target_ids:
        return

    # Pass 3: referrer aggregation (parallel).
    print('pass 3: referrers...', flush=True)
    _RTARGETS = target_ids
    _RMODE = 'edges'
    counts = {}
    examples = {}
    stopped = 0
    for part in _run_shards(heap_spans, n_workers):
        pc, pe, ps, _ = part
        stopped += ps
        for cid, c in pc.items():
            counts[cid] = counts.get(cid, 0) + c
        for cid, ex in pe.items():
            examples.setdefault(cid, ex)
    if stopped:
        print(f'{stopped} segments stopped early in pass 3', flush=True)

    def field_name(nid):
        return strings.get(nid, f'?name={nid}').split('/')[-1]

    total_refs = sum(counts.values())
    print(f'incoming references: {total_refs}', flush=True)
    print(f'{"count":>10}  referrer', flush=True)
    for cid, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        roid, toid, field = examples[cid]
        kind, detail = field
        if kind == 'static':
            where = f'static {field_name(detail)}'
        elif kind == 'index':
            where = f'element[{detail}]'
        else:
            where = f'.{field_name(detail)}'
        print(f'{c:10d}  {name_of(cid)}  e.g. obj@{roid:x}{where}'
              f' -> obj@{toid:x}', flush=True)

    # Optional example chain toward a root (one iterative pass per hop).
    if args.chain > 0 and examples:
        first_cid = max(counts, key=counts.get)
        current = examples[first_cid][1]
        seen = {current}
        print(f'chain from obj@{current:x}:', flush=True)
        _RMODE = 'hop'
        for hop in range(args.chain):
            _RTARGETS = {current}
            hit = None
            for part in _run_shards(heap_spans, n_workers):
                _, _, _, h = part
                if h is not None:
                    hit = h
                    break
            if hit is None:
                print(f'  [{hop}] obj@{current:x}: no referrer found'
                      ' (root or untracked)', flush=True)
                break
            rcid, roid, _, field = hit
            kind, detail = field
            if kind == 'static':
                print(f'  [{hop}] static {name_of(rcid)}.'
                      f'{field_name(detail)} -> obj@{current:x} (root)',
                      flush=True)
                break
            if kind == 'index':
                where = f'element[{detail}]'
            else:
                where = f'.{field_name(detail)}'
            print(f'  [{hop}] obj@{roid:x} ({name_of(rcid)}){where}'
                  f' -> obj@{current:x}', flush=True)
            if roid in seen:
                print('  (cycle, stopping)', flush=True)
                break
            seen.add(roid)
            current = roid


def main(argv=None):
    args = parse_args(argv)
    run(args)


if __name__ == '__main__':
    main()
