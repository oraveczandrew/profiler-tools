# profiler-tools

Offline analysis toolkit for **Android Studio memory captures (`.asdb`)** —
the allocation-tracking files behind *Profiler → Memory → Record allocations*.

An `.asdb` file is a SQLite database whose `UnifiedEventsTable` holds serialized
`Common.Event` protobufs. This toolkit decodes the two allocation event kinds
without Android Studio:

| Kind | Meaning | Payload |
|------|---------|---------|
| 408 | `MEMORY_ALLOC_CONTEXTS` | classes, methods, encoded stacks, threads |
| 409 | `MEMORY_ALLOC_EVENTS` | per-allocation records (class, size, thread, stack) |

(The event payload lives in the `Common.Event` oneof field numbered like the
kind itself — e.g. field 408 carries the 408 payload.)

## Layout

```
profiler-tools/
  asdbtools/      importable package (proto walker, db access, contexts, runner)
  scripts/        CLI entry points (thin wrappers around asdbtools)
  trace/          ART CPU-trace helpers (numpy; unrelated to .asdb)
  tests/          self-tests (no capture needed)
```

## Requirements

- Python 3.10+ (developed and measured on 3.14; Unix only — the parallel
  runner uses `fork`)
- standard library only for the `.asdb` tools; `numpy` for `trace/`

```bash
pip install -r requirements.txt  # only needed for trace/
```

No installation is needed for the `.asdb` tools: the scripts add their parent
directory to `sys.path`, so run them from anywhere:

```bash
python3 scripts/asdb_top.py --db capture.asdb
```

## Scripts

| Script | Purpose |
|--------|---------|
| `asdb_top.py` | top allocating classes, String-family totals, call sites, threads |
| `asdb_owned.py` | attribute every allocation to its nearest app-owned frame (`--app` markers) |
| `asdb_site.py` | break down everything under one call site (`--target` + `--app` markers) |
| `asdb_chain.py` | show full stacks containing all given frame substrings |
| `asdb_compare.py` | exact per-class counts across captures (first = baseline; `--watch` list) |
| `hprof_hist.py` | shallow per-class heap histogram from an ART `.hprof` dump (`--db`, `--top`, `--filter`) |
| `find_no_jvmfield.py` | static source check: class-level properties without `@JvmField` (`--src <kotlin-src-root>`) |

Common flags: `--db` (required, repeatable for compare), `--workers N`
(default: CPU count; `1` = serial path through the same code), `--output`.
`--app` (owned, site) and `--watch` (compare) take space-separated values.

```bash
# single capture, 8 workers
python3 scripts/asdb_owned.py --db before.asdb --app com/example/app MainActivity --workers 8 --output before.txt

# before/after comparison of an optimization
python3 scripts/asdb_compare.py --db before.asdb after.asdb --watch Ljava/lang/String; '[B' --workers 8
```

**Workload caveat:** two captures are only comparable per unit of work
(e.g. per created screen, per parsed document). Absolute counts scale with
session length; normalize before concluding anything.

## CPU traces (ART streaming)

`trace/` handles Android Studio CPU captures (`cpu-art-*.trace`) — the
binary ART streaming method traces behind *Profiler → CPU → Record*.
Pipeline: parse once, then analyze the parsed arrays:

```bash
python3 trace/trace_parse.py <capture>.trace  # -> <capture>.parsed.npz (next to the trace)
python3 trace/trace_analyze.py <capture>.trace # inclusive/exclusive stack walk (+ <capture>.stats.npz)
python3 trace/trace_deep.py <capture>.trace --ns com.example .app. --group .parser. --group .render.
python3 trace/trace_parents.py <capture>.trace <frame-substring> [...]
```

| Script | Purpose |
|--------|---------|
| `trace_parse.py` | decode the binary trace into `tid`/`dex`/`act`/`ts_us` numpy arrays (`.parsed.npz`) |
| `trace_analyze.py` | stack-walk analysis: inclusive/exclusive/call counts per method (`.stats.npz`) |
| `trace_deep.py` | thread breakdown, main-thread frames, caller lists, namespace rollup (`--ns`, `--group`) |
| `trace_parents.py` | parent/child attribution for the given frame substrings |
| `stackwalk.py` | shared library (not a CLI): parallel per-thread stack walker used by the above |

All take the `.trace` path as a required argument (usage error otherwise). Requires `numpy`. The binary layout is reverse-engineered
(see the `trace_parse.py` docstring: `SLOW` header, thread/method/data
records, `0=enter, 1/2=exit` actions with wall-microsecond timestamps).

## Parallelism

Decoding millions of events is CPU-bound single-threaded Python (GIL), so the
409 table is sharded by `rowid` and scanned by one forked worker per shard.
Each worker opens its own read-only SQLite connection and inherits the decoded
408 contexts through the fork — only `(lo, hi)` tuples cross the process
boundary. Measured ~2.4x at 8 workers on a 106 MB capture (serial 43s →
parallel 18s); beyond that shared-file reads and result merging dominate.

## App attribution

`asdb_owned.py` (and the `--app` filter in `asdb_site.py`) charges each
allocation to the nearest stack frame whose class or file contains one of
the `--app` markers, e.g. `--app com/example/app MainActivity`. Vendor/ROM
frames (e.g. OEM font or canvas hooks) then show up as costs *triggered
by* your call sites — which is usually where the fix belongs.

## Tests

```bash
python3 -m pytest tests/   # or: python3 -m unittest discover -s tests
```
