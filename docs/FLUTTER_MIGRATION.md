# Porting Bitmapper to a Flutter app

A plan for turning the Python reference implementation in `bitmapper/` into a
Flutter/Dart app. Written against the code as of this commit: ~900 lines of
source across 10 modules, 196 tests, no dependencies beyond NumPy and Pillow.

This is a plan, not a spec — the sequencing and the open decisions at the end
matter more than the specific file names.

## 1. The property that makes this port easy

The expensive stage runs on the **grid**, not the **canvas**.

A typical configuration downsamples a 1200x1200 image to a 150x150 grid,
quantizes and dithers *there*, then replicates blocks back up to full size.
That's 22,500 cells doing the hard work versus 1,440,000 pixels doing
block-fill. Measured in the Python reference:

| Work | Time |
|---|---|
| Whole pipeline, `dither="none"` (resize + downsample + upscale) | 0.076s |
| Same, plus Floyd-Steinberg error diffusion on the grid | 0.237s |

So the per-pixel-sequential part of the algorithm — the part that can't be
vectorized or trivially parallelized — touches a few tens of thousands of
cells. Everything that touches the full canvas is either a block-fill or a
row-wise multiply.

This is why the recommendation below is "just write it in Dart" rather than
anything involving FFI or shaders. Keep this property in mind if the design
ever changes: **moving quantization or dithering to full canvas resolution
would invalidate the entire performance argument.**

## 2. Where the pixel work should run

**Recommendation: pure Dart, on a background isolate.**

| Option | Verdict |
|---|---|
| **Pure Dart in an isolate** | **Recommended.** ~900 lines of array math with no exotic dependencies. Dart AOT compiles the nested loops to native code, which is exactly what the slow stage needs. One toolchain, one language, trivially debuggable, runs everywhere Flutter runs. |
| FFI to Rust/C | Defer. Would be faster still, but adds per-platform build config, a second language, and cross-compilation to the release pipeline — a large tax to remove a cost that is already small once compiled. Revisit only if profiling on a real device says otherwise. |
| GPU fragment shaders | Wrong shape for the core work. Error diffusion is inherently sequential (each cell depends on error propagated from its neighbors), so it can't be expressed as an independent per-fragment computation. *Could* make sense later for the full-canvas tail (upscale, grid gap, scanlines), which is embarrassingly parallel — see §8. |

The Python slowness in the notebook is an *interpreter* artifact. The same
nested loop in AOT-compiled Dart is a different animal. Do not port the
Python performance characteristics as if they were algorithmic facts; measure
on a device early (§7, Milestone 2) and let that drive any further work.

## 3. Data representation

| Python / NumPy | Dart |
|---|---|
| `np.ndarray` (H, W, 3) `uint8` | `Uint8List` of length `H*W*3`, row-major, with explicit `(y*W + x)*3 + c` indexing |
| `np.ndarray` float64 working buffer | `Float64List` — Dart's `double` is IEEE-754 binary64, identical to NumPy's `float64` |
| Palette (K, 3) `uint8` | `Uint8List` of length `K*3`, or a `List<int>` of packed ARGB |
| `PIL.Image.open` / `np.array` | `package:image` for decode, or `dart:ui` `instantiateImageCodec` + `toByteData` |
| Displaying the result | `ui.decodeImageFromPixels` → `RawImage`, avoiding a PNG round-trip for previews |

Do **not** reach for a Dart NumPy-alike. The operations in use are narrow
(elementwise math, block averaging, argmin over a small palette) and each is
clearer as an explicit loop over a typed list than as a generic array
abstraction.

A note that will save time: the vectorized NumPy stages are the ones that
need the most rewriting, because their brevity hides loops that Dart makes
explicit. The already-looping error-diffusion code is nearly a line-for-line
transcription. Budget effort accordingly — it's the inverse of what the
Python line counts suggest.

## 4. Stage-by-stage port

Ordered roughly as they should be built. "Effort" is relative, not absolute.

| Stage | Source | Effort | Notes |
|---|---|---|---|
| Nearest-color lookup | `quantize.py` | Low | The single hottest primitive — everything else calls it. Port `nearest_index` first and keep it the only implementation of "closest palette entry," as in Python. Chunking exists purely to bound NumPy's intermediate allocation; **Dart needs no chunking at all**, since a hand-written loop allocates nothing. Ties must go to the lowest index. |
| Grid downsample / upscale | `grid.py` | Low | Block boundaries come from `_splits` (evenly spread, *not* `np.array_split`); replicate exactly (§6). Upscale is a block-fill plus optional gutter. |
| Fixed palettes | `palettes.py` | Low | Pure data. Consider generating the Dart table from Python to avoid transcription typos in ~500 hand-typed RGB triples. |
| Adjustments | `adjustments.py` | Low | Three elementwise passes. Watch the uint8 cast semantics (§6). |
| Effects (scanlines) | `effects.py` | Low | Row-wise multiply at canvas resolution. |
| Ordered / random dither | `dither.py` | Low–Med | Bayer matrix is built recursively; easy. `random` needs a portable PRNG decision (§6). |
| Error-diffusion dither | `dither.py` | Medium | Closest to a direct transcription — already a nested loop. Keep the kernel table as data, exactly as `_DIFFUSION_KERNELS` is, so adding a kernel stays a one-line change. |
| Median cut | `palette_gen.py` | Medium | Recursive bucket splitting with a sort per split. Deterministic, so it goldens cleanly. |
| K-means | `palette_gen.py` | Medium | Deterministic *except* its random initialization (§6 — needs a decision). |
| Pipeline + config | `pipeline.py`, `presets.py` | Medium | Config validation should live in the same place conceptually (`__post_init__` → a Dart factory/assert). Presets are plain data. |
| Canvas resize | `pipeline.py` | **Decision needed** | See §6 — this is the one stage that cannot be made bit-exact, and it may not need to exist at all. |

## 5. How we know the port is correct

The reference implementation is deterministic (two exceptions in §6), which
makes golden-file testing the highest-leverage verification available. Use it
instead of trying to eyeball parity.

**Proposed fixture generator** (`scripts/export_goldens.py`, to write when the
port starts):

1. Pick a small set of source images — a photo, a synthetic gradient, a hard
   edge case (flat color, single pixel, extreme aspect ratio).
2. Cross them with a set of configs covering each palette mode, each dither
   method, and the effect knobs at non-default values.
3. For each pair, write: the source image as PNG, the config as JSON, and the
   expected `result.output` and `result.grid` as **raw** `.bin` dumps (not
   PNG — PNG encoders differ and would test the encoder, not the filter).
4. Commit that directory; have the Dart test suite load each case, run the
   port, and assert byte equality.

This gives an unambiguous pass/fail per stage rather than "looks about right,"
and it catches the subtle divergences in §6 immediately rather than after
someone notices a preview looks slightly off.

Two practical notes:

- Golden **per stage**, not just end-to-end. An end-to-end mismatch tells you
  something is wrong; a per-stage golden tells you *which* stage.
- Keep the fixture set small enough to commit comfortably. Raw dumps of a
  64x64 grid are a few KB; a 2000x2000 canvas is not. Golden the grid at full
  fidelity and the canvas at a modest size.

## 6. Portability gotchas

These are verified against the current code, not speculative.

**Bit-exactness is achievable everywhere except the canvas resize.**

1. **Canvas resize is not reproducible — and is mostly wasted work.**
   `_resize_to_canvas` uses Pillow's Lanczos filter. `package:image` has no
   Lanczos, and resampling filters differ in their kernels and edge handling,
   so this stage will not match byte-for-byte no matter how carefully it's
   written.

   There is a second problem worth solving at the same time: for the common
   case (source smaller than the output canvas) this stage *upscales* the
   source only for the very next stage to average it back down to a small
   grid. Measured: upscaling 400x400 → 2000x2000 and then downsampling costs
   ~0.095s versus ~0.005s downsampling the source directly — ~19x the work
   for a result that differs only by interpolation artifacts.

   **Recommendation:** in the Dart port, downsample to the grid directly from
   the source image and drop the intermediate canvas resize. The canvas size
   then only governs the upscale at the end, which is what it's really for.
   Note this *changes output slightly* versus Python, so decide deliberately
   and, if adopted, change the Python reference in the same commit so the two
   stay golden-comparable. Do not let them silently diverge.

2. **Block splitting: evenly spread, not `np.array_split`.** Length `L`
   splits into `n` parts at bounds `i * L // n`, so sizes differ by at most 1
   and the larger parts are spread across the image, e.g. `L=100, n=7` →
   `[14, 14, 14, 15, 14, 14, 15]`. Both `grid.downsample` and `grid.upscale`
   depend on this, and `upscale`'s gutter positions come from the same split
   so gaps land exactly on block edges. Get this wrong and everything is
   subtly misaligned.

   *Resolved:* this used to be `np.array_split`, which gives every leftover
   element to the first parts. That squeezes the start of the image into its
   cells and stretches the rest, e.g. 1024 px in 120 columns → 64 cells of
   9 px, then 56 of 8. It showed as visible "stretched to the right and
   bottom" distortion in the Dart app, which downsamples straight from the
   source (§6.1). Python hid it because the default canvas (2000) divides by
   the default grid (200). Both codebases changed together, and
   `tests/test_grid.py` shares expected values with the Dart grid tests.

3. **uint8 casts truncate, they don't round.** `np.clip(x, 0, 255).astype(np.uint8)`
   turns `250.7` into `250`, not `251`. Dart's `double.toInt()` also truncates
   toward zero, so the naive translation is correct — but only if nobody
   "helpfully" inserts a `.round()`.

4. **`random` dither and k-means init are not portable as written.** Both
   draw from NumPy's PCG64 generator, which Dart's `Random` will not
   reproduce. Options: (a) implement a small shared PRNG (xorshift128+) in
   both codebases, (b) make k-means initialization deterministic — picking
   evenly-spaced unique colors instead of random ones, which also makes it
   reproducible run-to-run in Python, or (c) accept that these two paths are
   verified statistically rather than by golden. **Recommended: (b) for
   k-means** (it removes a seed from the public surface and makes the palette
   stable across runs, which users will prefer anyway) and **(a) for `random`
   dither** if it's worth keeping at all — it's the least interesting of the
   11 methods.

5. **`subsample`'s rounding is a latent trap, not a current bug.** It uses
   `np.linspace(...).round()`, and NumPy rounds half-to-even (banker's
   rounding: `2.5 → 2`) while Dart rounds half away from zero (`2.5 → 3`).
   Across arbitrary (palette size, budget) pairs these disagree about 22% of
   the time. However, **zero reachable configurations hit it**: `n_colors` is
   always `2 ** bit_depth`, and across every power-of-two budget against every
   palette size from 2 to 599 the two rounding modes agree. So a direct Dart
   translation is safe today — but if `n_colors` ever becomes a free integer
   (a "limit to 5 colors" UI control, say), this silently becomes a real
   divergence. Replicate round-half-to-even if that happens.

6. **Float arithmetic order.** Dart `double` and NumPy `float64` are the same
   IEEE-754 type, so identical operation *order* gives identical results.
   Preserve the order in `_error_diffusion` — particularly the pre-scaled
   kernel weights (`weight * strength / divisor` computed once per call), which
   is both the performance fix and part of the numeric contract.

## 7. App architecture and milestones

**Shape of the app:**

- UI thread owns config state and renders; **all** filtering happens on an
  isolate (`Isolate.run` for one-shots, or a long-lived isolate with a port if
  setup cost matters). Typed lists transfer cheaply.
- **Two resolutions.** Interactive preview runs at a small canvas (and,
  crucially, the grid size the user selected — the grid is the look, so it
  can't be faked). Export runs at full resolution on demand. The measurements
  in §1 are the budget to design against.
- **Debounce slider input.** A drag across `dither_strength` should not
  enqueue 60 full pipeline runs; coalesce to the latest config and drop stale
  results.
- **Cancellation.** If a new config arrives while a run is in flight, abandon
  the old one. Error diffusion is the long pole and can't be interrupted
  mid-loop without a check, so poll a cancellation flag per row.

**Milestones:**

1. **Skeleton + goldens.** Fixture generator in Python; Dart project with the
   test harness reading fixtures. Nothing implemented yet — but the moment a
   stage lands, it's verified.
2. **Thin vertical slice.** Downsample → nearest-color (no dither) → upscale,
   with a fixed palette. End-to-end pixels on screen. **Profile this on a real
   device** — it validates the §2 recommendation before much is built on it.
3. **Dither + palettes.** Error-diffusion kernel table, ordered/Bayer, the
   full fixed-palette set, median cut.
4. **Full config surface.** Adjustments, effects, custom palettes, k-means,
   presets.
5. **App UX.** Image picking, preview/export split, sharing, saved presets.

Milestone 2 is deliberately early and deliberately measured — it's the cheapest
point at which the central architectural bet can be falsified.

## 8. Open decisions

Resolve these before or during the port rather than discovering them late:

1. **Drop the canvas resize?** (§6.1) Affects output fidelity and a ~19x cost
   on that stage. Decide once, apply to both codebases.
2. **Make k-means initialization deterministic?** (§6.4) Removes a seed from
   the API and makes palettes stable across runs; changes current output.
3. **Keep `random` dither?** It's the only method needing a cross-language
   PRNG, for the least distinctive result.
4. **Does the Python reference keep evolving after the port starts?** If yes,
   the goldens are the contract and both sides must be updated together. If no,
   freeze it and say so here — a reference implementation that drifts silently
   is worse than none.
5. **Later: move the full-canvas tail to the GPU?** Upscale, grid gap, and
   scanlines are per-pixel independent and run on the largest buffer. If
   exporting large canvases is slow on-device, this is the first thing to try
   — and it does not disturb the grid-level work at all.
