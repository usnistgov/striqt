# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Access the environment through `pixi`. The `test39` and `test314` environments carry the
test and dev tooling (pytest, ruff, ty) on Python 3.9 and 3.14; `doc` carries sphinx;
`cupy` exists only on Linux (the Jetson):

```sh
pixi run -e test39 pytest                                        # full suite, py39
pixi run -e test314 pytest                                       # full suite, py314
pixi run -e test39 pytest tests/waveform/test_fourier.py -q      # one file
pixi run -e test39 pytest -k test_stft_linearity                 # one test by name
pixi run -e test39 ruff format src tests                         # format (single quotes, 88 cols)
pixi run -e test39 ruff check src tests
pixi run -e test314 ty check src                                  # type check (must pass; py39 has a known baseline)
pixi run -e test314 pyright src                                   # advisory; accepted count in Conventions → Typing
pixi run -e test314 pyrefly check                                 # advisory; accepted count in Conventions → Typing
pixi run -e doc sphinx-build -E -a -b html doc doc/html          # docs (see chores/push-docs.sh to deploy)
pixi run -e cupy pytest                                          # GPU path, Jetson only
```

Every pixi environment installs the conda-forge `soapysdr` package (the `null` driver
only) and striqt as an editable PyPI dependency with its `dev` extra; `pixi run` refreshes
that install when `pyproject.toml` changes. `pytest` is 8.x on py39 and 9.x on py314, and
only pytest 9 reads the `[tool.pytest]` table in `pyproject.toml`, so test configuration
that must hold on both goes in `tests/conftest.py`. `uv run --extra test|dev|doc` still
works as a fallback but is not the project's environment.

Adding `--frozen` to `pixi run` skips the manifest/lock-file consistency check, saving
the reinstall (and any network fetch) that a `pyproject.toml`/`pixi.toml` edit would
otherwise trigger — but it also means a newly added dependency silently isn't installed,
surfacing as an `ImportError` instead of pixi's normal "lock file not up-to-date" message.
Default to running without it; reach for `--frozen` only in a tight loop of commands
where the manifest is already known not to have moved.

SoapySDR tests come in three tiers: pure units and tests against the fake `SoapySDR` module in `tests/sensor/fake_soapy.py` run everywhere; `TestNullDriver` needs the real bindings, which every pixi environment has; tests marked `hardware` need an attached Airstack and are skipped unless `STRIQT_TEST_HARDWARE=1 pixi run -e cupy pytest -m hardware` opts in on the Jetson.

CLI entry points (defined in `pyproject.toml`, implemented in `src/striqt/cli/`): `sensor-sweep`, `check-sweep`, `plot-capture`, `convert-spec`, `rechunk-zarr`, `zip-zarr`.

`tests/sensor/` runs one canonical synthetic sweep YAML (`tests/sensor/sweeps/synthetic.yaml`) and the site-style sweeps under `tests/sensor/sweeps/site/` end to end through the CLI, writing to `tmp_path`; everything else builds specs programmatically from `tests/sensor/synthetic_sources.py` (capture presets, `acquire_corrected`, `run_in_memory` with a `NoSink`) and checks the corrected IQ against the `striqt.analysis.testing` generators. Nothing writes to `tests/sensor/sweeps/outputs/` any more.

## Architecture

Four packages under `src/striqt/`, stacked bottom-up. Each depends only on those below it:

| Package | Role |
| --- | --- |
| `striqt.waveform` | Array-namespace-agnostic DSP kernels (FFT/STFT, resampling, OFDM/3GPP sync, power detectors) plus the numba/cupy JIT kernels, lazy-import helper and function caches everything above relies on. Takes and returns raw arrays; knows nothing of xarray or specs. |
| `striqt.analysis` | Wraps DSP into named *measurements* that return `xarray.DataArray`s with coordinates, units, and metadata. Owns zarr I/O, the logging adapters, and the GPU compute lock. |
| `striqt.sensor` | IQ acquisition, calibration, resampling/correction, sweep sequencing, sinks, threading utilities. Owns the YAML/JSON sweep schema. |
| `striqt.figures` | One plot function per `striqt.analysis` data variable, driven by the labeled coordinates. |

Public API lives in each package's top level (`striqt.analysis`, `striqt.sensor`, …). Everything under `.lib` is internal and changes without warning. `striqt.waveform.{arrays,fourier,ofdm}` are re-export shims over `waveform/lib/*`; a private helper (`_get_3gpp_index_cyclic_prefix`, `_truncated_buffer`, …) has to be imported from `striqt.waveform.lib.<module>`.

### Specs are the schema

Every configurable thing is a frozen, validated `msgspec.Struct` subclassing `specs.SpecBase` (`analysis/specs/structs.py`, `sensor/specs/structs.py`). These structs *are* the YAML schema, the JSON schema, the function keyword arguments, and the names of variables/coordinates/attrs in saved zarr files — all at once. Renaming a spec field renames the YAML key and the on-disk metadata field.

Field types are semantic aliases in `specs/types.py`, built as `Annotated[float, Meta('Standard name', 'Hz')]`. The `Meta` helper (`analysis/specs/helpers.py`) feeds `standard_name`/`units` into xarray attrs *and* the generated JSON schema, so annotate new fields with it rather than using a bare `float`.

### The measurement registry

`analysis/lib/register.py` holds `AnalysisRegistry`, a dict keyed by spec type, and the live instance `register.registry` that `measurements/shared.py` re-exports and every measurement module decorates with. A measurement is declared by decorating a function:

```python
@registry.coordinates(dtype='float32', attrs={'standard_name': 'Time elapsed', 'units': 's'})
def time_elapsed(capture, spec): ...

@registry.measurement(spec_type=specs.ChannelPowerTimeSeries, dtype='float32',
                      coord_factories=[power_detector, time_elapsed], ...)
def channel_power_time_series(iq, capture, **kwargs): ...
```

The decorator wraps the function so it accepts the spec's fields as keyword arguments, validates them, and returns a `DelayedDataArray` (see `analysis/lib/dataarrays.py`). The `@hint_keywords(spec_type)` stacked above it is an identity at runtime that gives the type checker the spec's fields as keyword parameters (see Typing under Conventions). Materialization to xarray is deferred so the GPU can queue several measurements before anything is pulled back to the CPU. `evaluate_by_spec` runs every measurement in the `analysis:` block under a `stopwatch`, then calls `sw.arrays.free_cupy_mempool()` when the input was cupy.

`registry.tospec()` then synthesizes the `Analysis` struct via `msgspec.defstruct`, one optional field per registered measurement. This is why the `analysis:` block of a sweep YAML validates against the set of registered measurements: adding a measurement adds a YAML key. Registration happens as an import side effect of `analysis/measurements/__init__.py`, so a new measurement module must be imported there to exist.

Three registry features matter when reading a measurement module:

- **`caches=`**: `KwArgCache` is single-element, keyed on capture + the listed kwargs, and lets several measurements share one expensive intermediate — every spectrogram-derived measurement reuses `spectrogram_cache`/`_cached_spectrogram` in `measurements/spectrum.py`, and the PSS/SSS measurements share `ssb_iq_cache`. Caches are scoped by `registry.cache_context(capture)`, which the sensor wraps around each capture.
- **`prefer_iq_source=`**: the IQ handed to a measurement is one of three stages of `dataarrays.AcquiredIQ` — `pre_align` (resampled and filtered), `pre_filter` (resampled only), or `aligned` (after trigger shifts). Sync detectors ask for `pre_align`; everything else defaults to `aligned`, falling back to `pre_align` when no trigger ran.
- **`registry.signal_trigger`** (`AlignmentSourceRegistry`): registers a measurement as a start-of-waveform detector. `Trigger` runs it with `as_xarray=False` and the sensor uses the returned lags to align captures when a source spec sets `signal_trigger:` (`cellular_5g_pss_sync`, `cellular_5g_sss_sync`).

### Waveform numerics

- **Backend dispatch.** `sw.array_namespace(x)` returns the array module; `sw.is_cupy_array(x)` takes an *array*, not a namespace (an `is_cupy_array(xp)` check is always False and silently disables the GPU path). Elementwise dB conversions in `lib/power_analysis.py` evaluate with numexpr on numpy and with the `cupy.fuse` kernels in `lib/jit/cuda.py` on cupy; both paths must stay in lockstep, including `out=` handling via `_arraylike_with_buffer`. `ofdm.corr_at_indices` dispatches the same way between `lib/jit/cpu.py` (numba `njit`) and `lib/jit/cuda.py` (`numba.cuda.jit`). `lib/jit/__init__.py` points `NUMBA_CACHE_DIR` at the platformdirs cache and stamps numba caches by content hash so reinstalls do not invalidate them; coverage.py excludes the jitted bodies (`[tool.coverage.report]` in `pyproject.toml`).
- **STFT layout.** `stft(x, axis=a)` returns windows stacked on axis `a` and frequency bins on `a + 1`, with the bins already in ascending `fftfreq` order because the shift is baked into the window (`get_window(..., fftshift=True)`); only `noverlap` of 0 or `nperseg // 2` is supported. `fftfreq` is computed with `decimal` so bin frequencies are exact; band selections through `_freq_band_edges` (`zero_stft_by_freq`, `downsample_stft`) are half-open `[lo, hi)` on that grid. `_truncated_buffer` flattens (copies) its input, so the `out=` arguments of `istft` and `downsample_stft` are not honoured yet.
- **Resampling.** `design_cola_resampler` returns a `ResamplerDesign` whose `nfft / nfft_out` equals `fs_sdr / fs_target` exactly, both divisible by the COLA window's divisor (hamming, `RESAMPLE_COLA_WINDOW` in `sensor/lib/compute/corrections.py`). `oaresample` and `resample` implement it through STFT overlap-add; `oafilter` is the same machinery used as a low-pass. `resample` fftshifts in the time domain (`time_fftshift`) so downsampling becomes a slice.
- **Caching.** `lib/util.py` provides `lru_cache` (tracked in `_caches`, cleared by `clear_caches()`) and `persistent_cache`, a `shelve` file under the platformdirs cache dir used for window design searches (`get_window`, `find_window_param_from_enbw`); despite sitting under `lru_cache` it is not itself LRU — its eviction order past `maxsize` is whatever the dbm backend iterates first. The test suite swaps the shelf for an in-memory dict (`tests/conftest.py:isolated_persistent_cache`) because the macOS dbm backend corrupts under eviction.
- **OFDM / cellular.** `lib/ofdm.py` models 3GPP frames: `get_3gpp_phy` → `Phy3GPP` (nfft, cyclic prefix sizes and start indices per TS 38.211 §5.3.1) feeds `cellular_cyclic_autocorrelation` through `corr_at_indices`; `pss_5g_nr`/`sss_5g_nr` build the m-sequences, `index_pss_symbols` encodes the TS 38.213 §4.1 SSB cases, `pss_params`/`sss_params` → `SyncParams`, `get_5g_ssb_iq` extracts the burst, `correlate_sync_sequence` produces a (coarse lag, fine lag) grid (`COARSE_LAG_DIM`, `FINE_LAG_DIM`), and `weighted_ssb_detect`/`choose_ssb_offset` pick the offset. The tests validate the sync path at 30 kHz subcarrier spacing only; the CP model at 15 kHz and 60 kHz is known wrong (strict xfails in `tests/waveform/test_waveform_ofdm.py`) and fixing it changes the shape of `cp_sizes`.

### Sensor bindings

A *sensor* is a binding of (source backend, sink, peripherals, spec types). `sensor/lib/bindings.py:bind_sensor` pairs a `Sensor` (implementation classes) with a `Schema` (spec types) and registers a `Controller` subclass under a string key. Each binding also gets a tagged `Sweep` subclass; all of them are unioned into the tagged type returned by `get_tagged_sweep_type()`. That tag is the `sensor_binding:` key at the top of every sweep YAML — it selects which capture/source/peripheral spec types the rest of the file is validated against.

Built-in bindings are in `sensor/bindings.py` (`air7101b`, `air7201b`, `air8201b` for hardware; `single_tone`, `noise`, `sawtooth`, `dirac_delta` synthetic sources; `mat_file`, `tdms_file`, `zarr_iq` file sources; `warmup`). Out-of-tree bindings are loaded through the `extensions:` block of a sweep spec, which inserts a path and imports a module before the rest of the spec is decoded (`sensor/lib/io.py:_import_extensions_from_spec`). `calibration.bind_manual_yfactor_calibration` derives a calibration binding from an existing one by adding a `noise_diode_enabled` capture field and swapping in `YFactorSink`.

The principal downstream user is the aggregate-directivity-acquisition project, which binds site-style sensors through `extensions:` and runs on a Jetson with cupy. Its environment does not install on a laptop, so its patterns are reproduced in-tree: `tests/sensor/sweeps/src/extensions.py` and the `site*` sweep directories under `tests/sensor/sweeps/`. The GPU path is the production path but cannot be executed here; a cupy-only change is unverified until someone runs `pixi run -e cupy pytest` on the Jetson, and cupy tests skip locally through the `_cupy is None` check in `tests/conftest.py`.

### Sweep execution

`open_resources(spec, path)` opens source, sink, peripherals, and calibration concurrently and returns a `Resources` TypedDict. `iterate_sweep(resources)` then runs a three-stage software pipeline in threads — acquire capture *i+1*, analyze capture *i*, write capture *i-1* — so a result is yielded once it has been through the sink. It is built on `util.zip_offsets(captures, (-2, -1, 0, 1))`; its docstring diagrams the stage offsets. GPU analysis deliberately runs in the foreground thread.

Captures are expanded from `captures:` plus `loops:` (`Range`, `List`, `Repeat`, `FrequencyBinRange`) by `specs.helpers.loop_captures`, then per-source overrides from `adjust_captures` are applied. Output is appended along the `capture` dimension of a zarr store (`ZarrCaptureSink`; `ZarrTimeAppendSink` instead appends along the spectrogram time axis, and `.zarr.zip` paths go through `Zipper`).

The `Controller` (`sensor/lib/controller.py`) owns the open source backend. `lookup` lets other threads wait for and fetch the controller (or its source ID) by source spec; `read_retries` wraps `read_iq` in `util.retry` for `ReceiveStreamError`/`OverflowError` when the source spec sets `receive_retries`. `controller.acquire()` returns an `AcquiredIQ` with only `pre_align` filled.

`sensor/lib/compute/corrections.py:correct_iq` is the per-capture signal path, in order: resample to `sample_rate` (`_scale_only` when no resampling is needed, otherwise `_resample`, or `_oaresample` when `STRIQT_USE_OARESAMPLE=1`), conjugate for high-side LO (`STRIQT_IGNORE_HIGHSIDE_LO` disables), FIR low-pass to `analysis_bandwidth` when finite, trim to `duration`, then run the `signal_trigger` and shift. `compute/analyze.py:analyze` applies that, evaluates the measurements, and packs a `DelayedDataset`. `compute/gpu.py:sweep_touches_gpu` decides whether `prepare_compute` runs a one-capture warmup sweep first (`build_warmup_sweep`) to pay the JIT and import costs before the real captures.

### Logging and concurrency

Loggers are `logging.LoggerAdapter`s (`analysis/lib/util.py:StriqtLogger`) registered by bare suffix in `_logger_adapters` and fetched with `get_logger(name)`: `'analysis'` is created when `striqt.analysis` imports; `'sweep'`, `'source'`, `'sink'`, `'periph'` are created when `striqt.sensor.lib.util` imports, which also calls `show_messages(logging.INFO)` and so installs stderr handlers as an import side effect. The underlying loggers are named `striqt.<suffix>`, so a handler on `logging.getLogger('striqt')` sees all of them; that is how `log_to_file` collects the sweep log. The adapter `extra` dict carries `capture_index`/`capture_count`/`capture_progress`, which the screen format prints; `sensor.lib.util.log_capture_context` swaps it per capture. `stopwatch` logs at the custom levels `PERFORMANCE_INFO` (15) and `PERFORMANCE_DETAIL` (12), which `log_verbosity(-1..3)` maps from the CLI `-v` count. `log_to_file` (wired from `Sink.log_path`/`log_level` by `open_resources`) writes the records as a JSON array, which is also valid YAML. All of this is process-global state: tests that touch it must snapshot and restore levels, handlers, and `extra` (the autouse `restore_logging_state` fixture in `tests/conftest.py` does this for every test).

`analysis.util.compute_lock(array)` is an `RLock` taken only for cupy arrays (or no array), serializing GPU work across threads. `sensor/lib/util.py` provides the cooperative cancellation used by the pipeline: `cancel_threads()` sets a global event, `propagate_thread_interrupts()` raises `ThreadInterruptRequest` in worker threads only (never the main thread), and `share_thread_interrupts()` clears it on exit. `ExceptionStack.defer()` collects exceptions from several threads and `handle()` re-raises one exception or an `ExceptionGroup`, with interrupts yielding to real errors; `await_and_ignore` applies it to a list of futures. `threadpool` is a lazily created module attribute (`__getattr__`). `DebugOnException` prints IPython-style tracebacks (`sensor/lib/tracebacks.py`) and is the CLI's top-level handler.

### Known gaps are recorded as strict xfails

Library defects that need a design decision rather than a local fix are written as `pytest.mark.xfail(strict=True, reason=...)` tests against the intended behaviour, so `grep -rn "xfail" tests` lists them with their causes. Read the reason before treating a red test as new, and expect a genuine fix to turn that xfail into an unexpected pass that must then be removed. The ledger `tests/xfail-audit.md` records each one (see Testing → The xfail ledger).

## Conventions

- **Python 3.9 is the floor** (`requires-python = ">=3.9"`, ruff `target-version = "py39"`; the `test39` and `test314` pixi environments cover both ends). Modules start with `from __future__ import annotations as __` so annotations can use modern syntax, but anything evaluated at runtime cannot.
- **msgspec resolves struct annotations at runtime.** In spec structs use `Union[...]`/`Optional[...]`, not `X | None`, and keep the names importable at runtime. `eval_type_backport` is a hard dependency for this reason. Several recent commits are reverts of "simplifications" here — collapsing a `Union`, or rewriting a `Literal` containing bools, has broken msgspec decoding before. Run the tests in `tests/sensor/` after touching a spec type.
- **Lazy imports are enforced.** Heavy modules are loaded via `util.lazy_import('scipy')` under an `if TYPE_CHECKING:`/`else:` split. `tests/test_imports.py` asserts that `scipy`, `xarray`, and `pandas` are still unreified and that `dask` was never imported after importing all four packages. Adding a module-level `import xarray` in the wrong place fails that test.
- **Test layout.** One test module per source module, named `test_<package>_<module>.py` (`tests/sensor/test_sensor_specs_helpers.py` covers `sensor/specs/helpers.py`), with `# %%` cells per function or class under test; add new tests to the existing module rather than creating a file per feature. `tests/` has no packages: module basenames must be unique across all subdirectories, and shared helpers are imported by bare name (`from conftest import ...`, `from sweep_strategies import ...`) because pytest's prepend import mode puts each test directory on `sys.path`. Do not add a `conftest.py` below `tests/` — pytest drops `sys.modules['conftest']` before loading each one, so a nested conftest would be what `tests/waveform/*` get from `from conftest import ...`. Put fixtures in the root conftest and strategies in one plainly named module per package (`tests/analysis/analysis_strategies.py`, `tests/sensor/sweep_strategies.py`), with the backend-agnostic primitives (`scalars`, `bounded_ints`, `finite_floats`, `short_text`) in the root conftest beside the array strategies. The root conftest also loads the hypothesis profile (function-scoped fixtures allowed, no deadline), so property tests need no `@settings` unless they set `max_examples`; cupy-dependent tests take the `cupy_available` fixture, and tests that should run on both backends take the parametrized `xp` fixture.
- **Array-namespace agnostic numerics.** Use `sw.array_namespace(x)` to get the namespace and dispatch; don't hard-code `numpy`. Everything must work on numpy and cupy, in 32-bit float.
- Formatting is ruff with single quotes and `line-length = 88`.
- For functions that need to return structured data based on named fields, prefer use of `dataclasses` or `msgspec.Struct` over `typing.TypedDict` or named tuples
- **Typing.** Every new or edited function, method and closure carries hints for all parameters and the return; only `self`/`cls` and lambdas are exempt. Wrapper catch-alls are `*args: P.args, **kwargs: P.kwargs`. Reuse the aliases in `striqt.waveform.lib.typing` (`Array`, `ArrayLike`, `XpType`, `ArrayBackend`, `WindowType`, `DTypeLike`, `LRUWrapped`), `striqt.analysis.lib.typing` (`Measurement`, `AnalysisFunc`, `WrappedAnalysis`, `CoordFunc`, `ZarrStore`) and `striqt.sensor.lib.typing` (`PS`/`PC`, `ResamplerKws`, `PassThroughWrapper`); `xp` is `ModuleType` when required and `XpType` when `None` means "infer"; `dtype` is `DTypeLike`. A decorator that returns a wrapper names its type variables in the return annotation (`-> CoordFuncWrapper[TC, TM, R]`, not the bare alias), otherwise ty erases the wrapped signature to `Unknown`.
  - `**kwargs` forwarded to a fixed target are `**kwargs: Unpack[<Name>Kws]` with the `TypedDict` defined in that package's `lib/typing.py` (`ResamplerKws` is the model); anything else is `**kwargs: Any`. `TypedDict` is for keyword sets only; structured return values stay `dataclasses`/`msgspec.Struct`. ty 0.0.81 does not flag unknown keywords through `Unpack` (strict xfail in `tests/test_typing.py`).
  - Measurement signatures are typed through `@hint_keywords(spec_type)`; a measurement body is `(iq: Array, capture: specs.Capture, **kwargs: Any) -> Measurement`. `tests/test_typing.py` runs ty over probes generated from `sa.registry` and asserts that every registered measurement exposes its spec fields statically, and over hand-written probes for the other keyword-unpacking sites.
  - `pixi run -e test314 ty check src` must report no diagnostics. `pixi run -e test39 ty check src` carries a baseline from its older dependency stubs (zarr 2, IPython 8, matplotlib 3.9, pandas 2) that must not grow; ty applies the 3.9 rules in both (`[tool.ty.environment]` in `pyproject.toml`). `tests/test_typing.py` enforces the test314 gate.
  - Suppress with `# ty: ignore[rule-name]` on the offending line, never a blanket `# ty: ignore` or `# type: ignore`. An ignore that is unused in one environment is a warning there, so on version-gated branches (zarr 2/3) suppress only what the test314 stubs reject.
  - pyright and pyrefly run in advisory mode, pinned to 3.9 like ty (`[tool.pyright]`, `[tool.pyrefly]` in `pyproject.toml`). Their remaining diagnostics are checker gaps, not hint problems: pyright 53 errors and pyrefly 40 errors on `src/` as of 2026-09-24 (numpy stubs without `normalize_axis_tuple`, the zarr 2/3 `hasattr` gate, `dict`-subclass `__hash__`, optional `cupy` imports, `TypeIs` to uninstalled torch/cupy, msgspec runtime-valued annotations in bound sweeps, deliberate protocol deviations, the `for k in list(locals())… del k` re-export loops, and a few flagged defects). Do not add `# pyright: ignore` or `# pyrefly: ignore` comments (ty never honoured them and they hide more than they explain): fix a new diagnostic with a better hint, or leave it in that count. Write `Union`/`Optional` rather than `X | Y` inside the `TYPE_CHECKING` alias blocks of the `lib/typing.py` modules, since pyright evaluates those under 3.9.
- **Comments:** only write one if a competent engineer (or a future Claude) reading the code cold would be surprised or misled without it. Worth writing: a non-obvious *why* (constraint, workaround, domain rule, performance trade-off); something that looks wrong but is intentional; external context that can't be inferred from the code (section from a technical standard, peer-reviewed publication, government report, public documentation, upstream bug/issue number); an invariant or precondition the types don't enforce. Do not write:
  - changelog or progress notes (`# added null check`, `# refactored from previous version`) — the diff is already in git. Progress notes belong in the chat response, the commit message, or the PR description, never in a source file comment;
  - narration of the obvious (`# loop over captures`, `# return the result`);
  - restatements of the signature or type annotations;
  - conversational leakage (`# as requested`, `# per the instructions`, `# TODO: user said to`);
  - section banners, unless the file is genuinely huge. Note that the existing `# %% Capture specs` markers are Jupyter-style cell delimiters, not decoration.

  If a comment is needed to explain *what* code does, prefer renaming or extracting a function instead — that removes the cause rather than the symptom. Comments are for *why*.

  After editing, re-read every comment you added and delete any that restate the code or describe the change rather than the code. This pass is cheap and catches most stragglers.

- **Docstrings** are the API reference (sphinx autodoc + napoleon render them into `doc/`), so write them as the caller's contract: inputs, outputs, side effects, exceptions. Google sections only — `Args:`, `Returns:`, `Yields:`, `Raises:`, `Examples:`, `References:` — never `Arguments:`, numpy underlines, or ad hoc headers.
  - **Summary line.** One line, imperative mood, lowercase first word, naming the effect or the returned quantity with its units: "return the duration (in s) of one 3GPP slot at `subcarrier_spacing`". Not "an iterator that ...", "helper for ...", "this function ...", or a third-person verb ("designs ..."). Include a trailing period unless the docstring
  is one-line docstring. `pixi run -e test39 ruff check --select D401 src` lists non-imperative summaries; it is not enforced, so run it on files you touch.
  - **Language** Write in American english and spelling.
  - **State what the signature cannot.** An `Args:` entry adds meaning, never type: units (Hz, s, S/s, dB vs linear power), array shape and which axis holds what, dtype, valid range or ratio constraint, and what `None`/`inf`/`'auto'` selects. `x: input array` and `fs: sampling rate` are omissions, not entries. `Returns:` gives shape, axis layout or dimension names, dict keys, units, and whether the array namespace follows the input. Generators use `Yields:` and say what ends the iteration (`loop=True` is infinite).
  - **Side effects are contract.** In-place modification (`overwrite_x`, `out=`, including an `out=` that is accepted but not honoured), process-global state (logger handlers, caches, the GPU lock), threads started, files written.
  - **`Raises:`** lists every exception the caller can act on with its condition. Omit exceptions that only a bug would raise.
  - **Avoid repeating exact code in the function when documenting the return value**. Rather, state which inputs will be used to compute the returned value, and the scope
  of possible return values such as bounds, etc.
  - **Behaviour, not mechanism.** Name the method only when it decides the result or the cost (overlap-add edge behaviour, the copy made by `_truncated_buffer`, a disk-cached window design). No step-by-step narration of the body; when the steps *are* the observable precedence (`loop_captures`), state them as rules.
  - **Present tense, present code.** Nothing about what the code used to do, was changed to do, or may do later — no "now", "no longer", "previously", "new", "legacy", "used to", "fix", "workaround", "TODO", "for now". No hedging — "should" where "must" is meant, "try to", "attempt", "safely", "properly", "simply", "just", "basically", "note that". No first person or reader address ("we", "our", "you"). No performance boasts ("with speed optimizations"): state the cost or nothing.
  - **Cite external context.** A standard section (3GPP TS 38.211 §5.3.1), a paper, or an upstream issue URL goes under `References:` or inline in parentheses.
  - **Do not duplicate what the tooling injects.** Spec-struct fields are documented through `Meta(description=..., units)`; the class docstring says what the block configures and how its fields interact, never a field list. A measurement docstring ends with `Args:\n    {args}`, which the registry fills; document the *output* (dimension names, units) in prose above it. `Controller` subclass and `arm` docstrings are generated.
  - **Markup.** Identifiers in single backticks (they render italic: no `default_role`), literal code in double backticks, examples in doctest form (ruff formats them with `docstring-code-format`).
  - **Length follows audience.** Public API (`striqt.analysis`, `striqt.sensor`, the `waveform` shims) gets full sections; a private `.lib` helper gets a summary line plus only the sections that carry information. No docstring beats one that restates the name.
  - **Scope of a change.** Bring the docstring of any function you edit to this standard, including stale parameter names and exception lists. Do not rewrite docstrings of untouched functions in the same change.

  After editing, re-read every docstring you touched: delete any sentence that restates the signature, describes the change rather than the code, or hedges.

## Testing

The judgement calls (what to assert, when a test is redundant, how to record a defect) are in
the `writing-tests` skill; this section is the map of the suite that the skill refers to.

### Where things are tested

- **Kernel numerics at the kernel.** `tests/waveform/test_fourier.py` and
  `test_power_analysis.py` prove the STFT, window design, resampling, binned power, the cyclic
  fold and its selectivity, the dB conversions and the roundoff models. Measurement tests
  assert what the measurement adds (spec fields → coordinates, dims, dtype, units, axis order,
  detector and statistic selection, documented edge behaviour), not the numerics underneath.
- **Every measurement's dims and dtype come from one registry-driven contract test**: the
  last cell of `tests/analysis/test_analysis_register.py` builds `CONTRACT_SPECS` over the
  registered measurements and asserts `da.dtype == info.dtype` and the full `dims` tuple.
  Per-measurement tests assert what it cannot know: coordinate order, sizes, the frequency
  grid, coordinate dtypes.
- **End-to-end sweeps are covered once**, in `tests/sensor/test_sensor_sweeps.py`: the CLI
  over `tests/sensor/sweeps/synthetic.yaml` and the site sweeps, `check-sweep`, and in-memory
  sweeps over the four synthetic bindings checked against the generators. Sensor signal-path
  properties (origin/delay, level, band placement, stage semantics) live in
  `test_sensor_compute_corrections.py` on `acquire_corrected`, not at the sweep level.
- **The type checker's view** of the keyword-unpacking signatures is pinned by
  `tests/test_typing.py` and the probes under `tests/ty_probes/` (see Typing); import-time
  side effects by `tests/test_imports.py`.
- The downstream project's patterns are reproduced by `tests/sensor/sweeps/src/extensions.py`
  and the `site*` sweep directories; the papers the cyclic-power and sync tests derive their
  expectations from are under `_training_material/papers/` (untracked; locate with a search).

### Shared setup has one owner

The suite is flat and every helper is imported by bare name, so a second definition of the
same thing is invisible until the two drift. Look here before writing setup:

- `tests/conftest.py`: fixtures (`xp`, `cupy_available`, `real_soapy`, `armed_tone_controller`,
  the autouse `restore_logging_state` and `isolated_persistent_cache`), the array and scalar
  hypothesis primitives (`float_arrays`, `iq_waveforms`, `scalars`, `bounded_ints`,
  `finite_floats`, `short_text`), the sweep-file path constants (`SWEEP_DIR`, `SITE_DIR`,
  `SITE_SPEC`, `FAKE_SOAPY_SPEC`), and the `construct`/`raises_on_both_paths` helpers for
  checking a spec on both the direct and the `from_dict` path.
- `tests/numeric_checks.py`: only assertions and oracles — `assert_close` (the single entry
  point), `assert_within`, `reference_power`, `cross_backend`, `numpy_and_cupy`,
  `elementwise_rtol`, `levels`, `populated`. No waveform builders and no tolerance formulas
  of its own.
- `tests/analysis/analysis_strategies.py`: analysis spec factories and constants
  (`POWER_BINS`, `CELL_FS`, `CELL_FRAME`, `capture_of`, `registered_tolerance`, the SSB
  spectrogram fixtures).
- `tests/sensor/sweep_strategies.py`: sensor spec factories and constants (`SOURCE`, the
  function-generator source spec nine modules share; `make_capture`/`make_sweep`/
  `make_sweep_kws` generic builders; `SPG`, `RESOLUTION_MSG`).
- `tests/sensor/synthetic_sources.py`: the sensor harness — the four capture presets,
  `preset_capture`, `preset_sweep`, `tone_capture`, `tone_captures`, `tone_sweep`, `fs_sdr`,
  `expected_raw`, `build_acquired_iq`, `acquire_corrected`, `expected_corrected`,
  `run_in_memory`.
- `tests/sensor/soapy_factories.py`: `MCR` and the soapy spec factories; `fake_soapy.py`: the
  fake `SoapySDR` module (`stub_controller`/`receive_buffers` for buffer tests live in
  `test_sensor_sources_buffers.py`).

### Inputs and tolerance models

- Build inputs from `striqt.analysis.testing` (`tone`, `single_tone`, `noise`, `sawtooth`,
  `dirac_delta`, `circular_awgn`): backend-agnostic, windowable, and the same definitions the
  synthetic sensor sources use, so a two-port generator call reproduces a two-port
  acquisition. They use `numpy.random.RandomState` because that is the API numpy and cupy
  share.
- Tolerances come from the library's own models, which are the contract the suite tests to:
  `striqt.waveform.power_analysis` (`log_conversion_tol`, `bin_power_rtol`, `bin_power_rms`,
  `stat_rtol`, `level_tolerance_dB`, `off_peak_dB_tolerance`, …), `striqt.waveform.fourier`
  (`fft_tolerance_rms`, `peak_factor`, `on_peak_roundoff`, `off_peak_floor_dBc`),
  `striqt.waveform.arrays` (`unit_roundoff`, `accum_rms`, `accum_rtol`, `mean_atol`),
  `striqt.waveform.ofdm` (`corr_atol`); a measurement's budget is its registered tolerance
  (`registered_tolerance` in `analysis_strategies`, `capture_tolerances`/`sweep_tolerances` in
  `striqt.sensor.lib.compute.tolerance`). Module-level constants inside `striqt.waveform.lib`
  (`ROUNDOFF_SAFETY` and friends) are implementation details: never import them.
- Exact expectations the arithmetic allows: a bin-centred unit tone through a boxcar window
  reads exactly 0.0 dB in one bin; a constant envelope puts exactly 1.0 of a normalized
  histogram in one bin; a unit-amplitude waveform reads exactly 0.0 dBm, since the power
  measurements treat `|iq|**2` as mW.

### Backends and process-global state in tests

The `xp` fixture is the only backend selector: `pytest_generate_tests` parametrizes it from
the `namespaces` marker (default `('numpy', 'cupy')`, widen with
`@pytest.mark.namespaces('numpy', 'cupy', 'dask')`) and `namespace_module` does the skip. Draw
arrays on numpy under `@given` and convert with `as_xp`; take `cupy_available` only when a test
needs both backends at once and compare through `numpy_and_cupy`. Cupy branches do not execute
off the Jetson. Logging adapters, handlers and levels, the persistent shelf cache and
`sys.path` are process-global; the autouse root-conftest fixtures snapshot and restore them,
and a module-scoped fixture that installs a fake module must not share a module with a
function-scoped one that installs the same fake. The hypothesis `striqt` profile suppresses
`function_scoped_fixture` and `differing_executors` (the latter because the hypothesis plugin
does not recognise `pytest_generate_tests` parametrization).

### pytest extensions

`[project.optional-dependencies].test` pins `pytest`, `hypothesis`, `pytest-subtests` (py39
only; the fixture is core in pytest 9), `pytest-lazy-fixtures`, `pytest-cov`,
`coverage[toml]`, `nptdms`. Do not add `pytest-regressions`, `pytest-cases`, `pytest-check` or
`pytest-mock`: golden files would hide the derivation that makes a tolerance reviewable, and
the other three duplicate `parametrize`, `subtests` and `monkeypatch`.

### The xfail ledger

`tests/xfail-audit.md` is the registry of strict xfails. An entry gives the node id, the
mechanism with file and line, why the test's expectation is the right one, the impact tier,
and the fix with its difficulty; its numbered items are stable identifiers, and a suite-state
paragraph records the `passed/skipped/xfailed` counts of each pass. `reason=` names the
mechanism and stays verbatim once written. A fix turns the xfail into an unexpected pass:
remove the marker and retire the item in the same change, and update the item when a rename
or fold changes a node id it names. A run with any xpass is a failed run.

### Verifying a test change

Run both interpreters (pytest and hypothesis versions differ) and pass `tests` explicitly,
because pytest 8 on py39 ignores `[tool.pytest]`:

```sh
pixi run -e test39 ruff format src tests && pixi run -e test39 ruff check tests
pixi run -e test39 pytest -q -p no:cacheprovider tests -rxX
pixi run -e test314 pytest -q -p no:cacheprovider tests -rxX
grep -rn "mark.xfail" tests --include='*.py' | wc -l   # reconcile with the ledger
pixi run -e test314 ty check src                        # must stay clean
```

Lint `tests`, not `src tests` (see the ruff note under Typing). Account for every count
delta: `passed` moves only by the arithmetic of folds, `xfailed` only by markers added or
retired, and an equal drop in `passed` and `skipped` is the fingerprint of an
`xp`-parametrized test disappearing. Keep new tests under about 0.3 s by shrinking the
compute; the whole suite runs in under a minute per interpreter. If `pixi run` fails on a
network fetch, `--frozen` skips the reinstall; as a last resort
`PYTHONPATH="$PWD/src" ./.pixi/envs/<env>/bin/python -m pytest tests` runs without the
editable install but does not exercise the console entry points.

## Status

Early beta. Base-package APIs are expected to be stable; `.lib` internals, YAML schemas, and CLI options may still shift. The sweep schema is documented in `doc/reference/schema.rst` (generated from the msgspec structs). `tests/sensor/sweeps/synthetic.yaml`, `tests/sensor/sweeps/site/*.yaml` and `tests/sensor/sweeps/fake_soapy-*.yaml` are the most current worked examples.
