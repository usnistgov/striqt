# Architecture

## Package layout

Four packages under `src/striqt/` are stacked bottom-up:

1. `striqt.waveform`: array numerics on numpy and cupy
2. `striqt.analysis`: measurement specs and the measurement registry
3. `striqt.sensor`: acquisition, sensor bindings, and sweep execution
4. `striqt.figures`: plotting

Each package depends only on the packages below that package. The command-line entry points in
`striqt.cli` sit over all four.

The public API of each package is the top-level namespace (`striqt.analysis`,
`striqt.sensor`, and so on). Everything under a `.lib` subpackage is internal and
changes without warning.

### Shims over `striqt.waveform.lib`

The top-level modules of `striqt.waveform` re-export names from `striqt.waveform.lib`:

- `striqt.waveform.arrays` re-exports `waveform/lib/arrays.py`.
- `striqt.waveform.fourier` and `striqt.waveform.power_analysis` re-export the
  packages `waveform/lib/fourier` and `waveform/lib/power`.
- `striqt.waveform.comms_5gnr`, `striqt.waveform.comms_lte`, and
  `striqt.waveform.comms_802_16` re-export per standard from `waveform/lib/comms`. A
  name shared by two standards, such as `Phy3GPP`, is exported by both of the shims.

A private helper (`_truncated_buffer`, `_get_3gpp_index_cyclic_prefix`, and so on)
is imported from the `striqt.waveform.lib` module that defines the helper
(`striqt.waveform.lib.fourier.stft`, `striqt.waveform.lib.comms.phy_3gpp`).
`waveform/lib/fourier/__init__.py` imports only the submodules of the package, so
inside `lib` a name is reached as `fourier.stft.stft` or
`fourier.windows.get_window`.

A shim is a plain `from .lib.<module> import (...)` block with no `__all__`. The
import list is the export list. Ruff's F401 is switched off for these files through
`[tool.ruff.lint.extend-per-file-ignores]` in `pyproject.toml`, so a new shim
(`sensor/io.py`, `analysis/testing.py`, and so on) also gets a path added there.

`__all__` is written only where the list does a second job:

- `waveform/arrays.py` curates what `from .arrays import *` in
  `waveform/__init__.py` pulls into the root namespace. The roundoff models stay
  reachable as `sw.arrays.accum_rms` but out of `sw`.
- `analysis/measurements/__init__.py` curates `from .measurements import *` in
  `analysis/__init__.py`, so that the measurement submodules (`cellular`, `power`,
  `spectrum`, and so on) do not leak into `striqt.analysis`.

## Specs are the schema

Every configurable thing is a frozen, validated `msgspec.Struct` that subclasses
`specs.SpecBase` (`analysis/specs/structs.py`, `sensor/specs/structs.py`). These
structs are at once the YAML schema, the JSON schema, the function keyword arguments,
and the names of the variables, coordinates, and attrs in saved zarr files. Renaming
a spec field renames the YAML key and the on-disk metadata field.

Field types are semantic aliases in `specs/types.py`, built as
`Annotated[float, Meta('Standard name', 'Hz')]`. The `Meta` helper
(`analysis/specs/helpers.py`) feeds `standard_name` and `units` into the xarray
attrs and into the generated JSON schema. A new field is annotated with `Meta`
rather than with a bare `float`.

## The measurement registry

`analysis/lib/register.py` defines `AnalysisRegistry`, a dict keyed by spec type, and
the live instance `register.registry`. `measurements/shared.py` re-exports the
instance, and every measurement module decorates with the instance.

A measurement is declared by decorating a function with
`@registry.measurement(spec_type=..., dtype=..., coord_factories=[...])`. A
coordinate factory is declared with `@registry.coordinates(...)`. The decorator
wraps the function so that the wrapper accepts the fields of the spec as keyword
arguments, validates the fields, and returns a `DelayedDataArray`
(`analysis/lib/dataarrays.py`). The `@hint_keywords(spec_type)` stacked above the
registration is an identity at runtime that gives the type checker the fields of the
spec as keyword parameters.

Materialization to xarray is deferred so that the GPU can queue several measurements
before anything is pulled back to the CPU. `evaluate_by_spec` runs every measurement
in the `analysis:` block under a `stopwatch`, then calls
`sw.arrays.free_cupy_mempool()` when the input was a cupy array.

### Analysis groups

`specs.BuiltinAnalyses` (`analysis/specs/structs.py`) is the hand-written struct with
one `Optional` field per registered measurement, named for the measurement.
`specs.BuiltinTriggers` is the same struct for the two triggers.
`striqt.sensor.specs.BundledAnalysis` and `BundledTriggers` are aliases of the two.
Both set `omit_defaults=True`, so `to_dict` and the saved attrs name only the enabled
measurements.

The `analysis:` block of a sweep YAML validates against the analysis group of the
binding, `Schema.analysis`, which defaults to `BuiltinAnalyses`. A binding can
restrict the block to a subset of the registered measurements, or name a measurement
that an extension registers with `sa.registry.measurement`. `bind_sensor` runs
`register.check_group_registered` on the group. The check requires:

- every field is an `Optional` registered spec type with default `None`;
- every field is named by the registered measurement name;
- the group sets `omit_defaults=True`.

Adding a built-in measurement means registering the measurement in a module that
`analysis/measurements/__init__.py` imports, and adding a field for the measurement
to `BuiltinAnalyses`. `register.check_group_fields` runs at the end of that
`__init__` and fails the import when the registry and the struct disagree.

`Source.signal_trigger` is a trigger name or a `BuiltinTriggers` with exactly one
field set. The spec in that field then wins over the sweep analysis.

### Registry features in a measurement module

Three registry features matter when reading a measurement module.

- **`caches=`.** `KwArgCache` is a single-element cache keyed on the capture plus
  the listed kwargs. A cache lets several measurements share one expensive
  intermediate. Every spectrogram-derived measurement reuses `spectrogram_cache` and
  `_cached_spectrogram` in `measurements/spectrum.py`, and the PSS and SSS
  measurements share `ssb_iq_cache`. Caches are scoped by
  `registry.cache_context(capture)`, which the sensor wraps around each capture.
- **`prefer_iq_source=`.** The IQ handed to a measurement is one of three stages of
  `dataarrays.AcquiredIQ`: `pre_align` (resampled and filtered), `pre_filter`
  (resampled only), or `aligned` (after trigger shifts). Sync detectors ask for
  `pre_align`. Everything else defaults to `aligned`, falling back to `pre_align`
  when no trigger ran.
- **`registry.signal_trigger`** (`AlignmentSourceRegistry`). The decorator registers
  a measurement as a start-of-waveform detector. `Trigger` runs the measurement
  with `as_xarray=False`. The sensor uses the returned lags to align captures when a
  source spec sets `signal_trigger:` (`cellular_5g_pss_sync`,
  `cellular_5g_sss_sync`).
