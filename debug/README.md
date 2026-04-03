# Standalone C++ Programs

This directory contains standalone C++ programs for smoke-testing and profiling
the stochtree C++ API.  They compile against the same library objects as the R
and Python packages, so they are the fastest way to iterate on C++ changes
without going through a language binding.

## Programs

| Target | Source | Purpose |
|---|---|---|
| `debug_bart` | `bart.cpp` | Identity and probit BART smoke tests; optional wall-time table |

## Building and running

`debug_bart` serves double duty depending on how it is built:

| Build type | CMake flag | Use case |
|---|---|---|
| `Debug` | `-DCMAKE_BUILD_TYPE=Debug` | Step-through debugging (`lldb`/`gdb`) |
| `RelWithDebInfo` | `-DCMAKE_BUILD_TYPE=RelWithDebInfo` | Profiling (optimized + symbols) |

```bash
# For a debugging session: unoptimized, full debug symbols.
cmake -DBUILD_DEBUG_TARGETS=ON -DCMAKE_BUILD_TYPE=Debug -B build
cmake --build build --target debug_bart
./build/debug_bart          # smoke tests only — runs in seconds

# For profiling: optimized with debug symbols for readable stack traces.
cmake -DBUILD_DEBUG_TARGETS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -B build
cmake --build build --target debug_bart
./build/debug_bart --timing # smoke tests + wall-time table
```

Output sections:

1. **Identity BART smoke test** — continuous outcome, RMSE and posterior mean sigma2
2. **Probit BART smoke test** — binary outcome, class calibration check
3. **Wall-time table** — BARTFit wall time across scaling scenarios (`--timing` only)

---

## Profiling on macOS (Xcode Instruments)

The recommended approach is **Xcode Instruments → Time Profiler**.  It gives a
call tree with per-symbol wall time and integrates with the Xcode source viewer.

### Option A: command-line launch (xctrace)

```bash
# Record a Time Profiler trace.
xctrace record \
    --template 'Time Profiler' \
    --output bart_profile.trace \
    --launch -- ./build/debug_bart

# Open the trace in Instruments.
open bart_profile.trace
```

In Instruments, select the **Time Profiler** instrument, click the call tree,
and use **Invert Call Tree** + **Hide System Libraries** to isolate hot paths
in stochtree code.

### Option B: attach from Instruments GUI

1. Open Instruments (`xcrun instruments` or via Xcode → Open Developer Tool → Instruments).
2. Choose **Time Profiler**.
3. Click the target selector → **Choose Target…** → browse to `./build/debug_bart`.
4. Press **Record**.

### Option C: `sample` (quick, no Xcode required)

`sample` is a lightweight macOS command that captures call stacks at 1 ms
intervals.  Useful for a quick look without opening Instruments.

```bash
# Launch the program in the background, then sample it.
./build/debug_bart &
PID=$!
sample $PID -wait -file bart_sample.txt
# Or, sample for 10 seconds at 1ms intervals:
sample $PID 10 1 -file bart_sample.txt
cat bart_sample.txt
```

### Compilation flags for profiling

For useful symbol information, build with `RelWithDebInfo` (see above).
Avoid `Debug` for profiling — unoptimized code does not reflect real hotspots.

If you need line-level resolution in Instruments, add `-fno-omit-frame-pointer`
to the CMake build:

```bash
cmake -DBUILD_DEBUG_TARGETS=ON \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer" \
      -B build
cmake --build build --target debug_bart
```

---

## Profiling on Linux (perf)

### Quick summary with `perf stat`

```bash
perf stat ./build/debug_bart
```

Reports cache misses, branch mispredictions, instructions per cycle, and wall
time — useful for a first pass without drilling into the call tree.

### Call-graph profiling with `perf record` / `perf report`

```bash
# Record with call graphs (frame-pointer unwinding — fast, needs -fno-omit-frame-pointer).
perf record -g --call-graph fp ./build/debug_bart
perf report --stdio | head -80

# Or use DWARF unwinding (more accurate, works without -fno-omit-frame-pointer,
# but higher overhead):
perf record -g --call-graph dwarf ./build/debug_bart
perf report
```

`perf report` opens an interactive TUI.  Press `/` to search for a symbol,
`Enter` to drill into callers/callees, `q` to quit.

### Callgrind (Valgrind) + KCachegrind

Callgrind counts instructions exactly (no sampling noise) but runs 10–50×
slower than native.  Good for comparing two implementations precisely.

```bash
valgrind --tool=callgrind --callgrind-out-file=callgrind.out \
         ./build/debug_bart
kcachegrind callgrind.out
```

`kcachegrind` shows a source-annotated call graph.  Install via your package
manager (`apt install kcachegrind` / `dnf install kcachegrind`).

### Compilation flags for Linux profiling

```bash
cmake -DBUILD_DEBUG_TARGETS=ON \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer" \
      -B build
cmake --build build --target debug_bart
```

`-fno-omit-frame-pointer` is required for frame-pointer-based unwinding
(`perf record -g --call-graph fp`); DWARF unwinding works without it.

---

## Adding a new profiling scenario

Edit `debug/bart.cpp` and add a row to the `scenarios` vector in
`run_timing_table()`:

```cpp
{"My new scenario", n, p, num_trees, num_gfr, num_mcmc, num_chains,
 StochTree::LinkFunction::Identity},
```

Rebuild and rerun:

```bash
cmake --build build --target debug_bart && ./build/debug_bart
```
