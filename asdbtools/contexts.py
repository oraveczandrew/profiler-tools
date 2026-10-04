"""Decoding of kind-408 (MEMORY_ALLOC_CONTEXTS) payloads.

Schema (from the Android Studio profiler plugin protobuf):

    408 -> MemoryAllocContextsData{1 = BatchAllocationContexts}
    BatchAllocationContexts{1 timestamp, 2 classes, 3 methods,
                            4 encoded_stacks, 5 thread_infos, 6 memory_map}
    AllocatedClass{1 class_id, 2 class_name, 3 class_loader_id}
    StackFrame{1 method_id, 2 class_name, 3 method_name, 4 file_name, 5 line}
    AllocationStack{1 stack_id, 2 full_stack, 3 encoded_stack}
    ThreadInfo{2 thread_id, 3 thread_name}

    Encoded stacks carry (method_id, line) pairs (fields 1, 2); the captures
    seen so far are encoded-only, and the methods table carries no lines, so
    every printed :line comes from encoded field 2 (verified sane).
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

from asdbtools.db import kind_blobs
from asdbtools.proto import fm, ss, sub, sv


class Contexts:
    """Decoded 408 tables: classes, methods, stacks and threads."""

    def __init__(self):
        # class tag -> 'Ljava/lang/String;' style name
        self.classes = {}
        # method id -> (class, method, file, line)
        self.methods = {}
        # stack id -> [(method_id, line)]
        self.stacks = {}
        # thread id -> name
        self.threads = {}


def load_contexts(db):
    """Decode every 408 row of an open database into a Contexts."""
    ctx = Contexts()
    for blob in kind_blobs(db, 408):
        ctxs = sub(blob, 408)
        batch = sub(ctxs, 1) if ctxs is not None else None
        if batch is None:
            continue
        cm = fm(batch)
        for c in cm.get(2, []):
            cf = fm(c)
            raw = cf.get(2)
            ctx.classes[sv(cf, 1)] = raw[0].decode('utf-8', 'replace') if raw else ''
        for m in cm.get(3, []):
            mf = fm(m)
            ctx.methods[sv(mf, 1)] = (ss(mf, 2), ss(mf, 3), ss(mf, 4), sv(mf, 5))
        for st in cm.get(4, []):
            sf = fm(st)
            sid = sv(sf, 1)
            frames = []
            full = [x for x in sf.get(2, []) if isinstance(x, bytes)]
            encoded = [x for x in sf.get(3, []) if isinstance(x, bytes)]
            source = full[0] if full else (encoded[0] if encoded else None)
            if source:
                for fr in fm(source).get(1, []):
                    ff = fm(fr)
                    frames.append((sv(ff, 1), sv(ff, 2 if encoded and not full else 5)))
            ctx.stacks[sid] = frames
        for t in cm.get(5, []):
            tf = fm(t)
            ctx.threads[sv(tf, 2)] = ss(tf, 3)
    return ctx


def frame_str(ctx, mid, line):
    if mid in ctx.methods:
        c, m, f, ln = ctx.methods[mid]
        loc = '%s:%s' % (f.rsplit('/', 1)[-1], ln or line)
        return '%s.%s(%s)' % (c.rsplit('/', 1)[-1], m, loc)
    return 'method#%d:%d' % (mid, line)


def stack_str(ctx, sid, depth=6):
    frames = ctx.stacks.get(sid, [])
    return ' <- '.join(frame_str(ctx, m, l) for m, l in frames[:depth]) or 'stack#%d?' % sid
