# Testing

This file is the map of the test suite. The judgement calls (what to assert, when a test is
redundant, how to record a defect) are in the `writing-tests` skill, which is not tracked in the
repository.

## Where things are tested

- **Kernel numerics at the kernel.** The `tests/waveform/test_waveform_fourier_*.py`
  modules and `test_waveform_power_convert.py`/`test_waveform_power_detect.py` prove the
  STFT, window design, resampling, binned power, the cyclic fold and its selectivity, the
  dB conversions and the roundoff models; the `test_waveform_comms_*.py` modules prove the
  frame models, the cyclic-prefix correlator and the 5G NR sync pipeline. Measurement tests
  assert what the measurement adds (spec fields → coordinates, dims, dtype, units, axis order,
  detector and statistic selection, documented edge behavior), not the numerics underneath.
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
- **The noise-line and histogram-limit helpers of `striqt.figures`** are tested in
  `tests/figures/` against the synthetic 3-capture datasets of `figures_datasets.py`
  (`make_dataset`, `predicted_levels_dBm`, `occupied_bins`), with the expected levels and
  bin supports from the dataset construction; the one plotting test runs headless through
  `MPLBACKEND=agg`, set in the root conftest before any third-party import.
- **The type checker's view** of the keyword-unpacking signatures is pinned by
  `tests/test_typing.py` and the probes under `tests/ty_probes/` (see Typing); import-time
  side effects by `tests/test_imports.py`; the docstring prose gate (no "it"/"its" for code,
  no `striqt.<pkg>.lib` path in a public docstring) by `tests/test_docstrings.py`, whose
  pre-existing hits are listed in `tests/docstring-baseline.txt` and only ever removed.
- The downstream project's patterns are reproduced by `tests/sensor/sweeps/src/extensions.py`
  and the `site*` sweep directories; the papers the cyclic-power and sync tests derive their
  expectations from are under `_training_material/papers/` (untracked; locate with a search).

## Shared setup has one owner

The suite is flat and every helper is imported by bare name, so a second definition of the
same thing is invisible until the two drift. Look here before writing setup:

- `tests/conftest.py`: fixtures (`xp`, `cupy_available`, `real_soapy`, `armed_tone_controller`,
  the autouse `restore_logging_state` and `isolated_persistent_cache`, and
  `isolated_bindings`/`isolated_analysis_registry` for a test that binds a sensor or
  registers a measurement with the live registries), the array and scalar
  hypothesis primitives (`float_arrays`, `iq_waveforms`, `scalars`, `bounded_ints`,
  `finite_floats`, `short_text`), the sweep-file path constants (`SWEEP_DIR`, `SITE_DIR`,
  `SITE_SPEC`, `FAKE_SOAPY_SPEC`), and the `construct`/`raises_on_both_paths` helpers for
  checking a spec on both the direct and the `from_dict` path.
- `tests/numeric_checks.py`: only assertions and oracles — `assert_close` (the single entry
  point), `assert_within`, `reference_power`, `cross_backend`, `numpy_and_cupy`,
  `elementwise_rtol`, `levels`, `populated`. No waveform builders and no tolerance formulas
  of its own.
- `tests/waveform/waveform_strategies.py`: the comms builders (`FS`, `phy_5g`, `sync_params`,
  `cp_case`, `embed_pss_ports`, `get_ssb_iq`, …) shared by the `test_waveform_comms_*`
  modules, and the fourier strategies (`noise_waveforms`, `sample_rates`,
  `stft_parameters`, `cross_backend_sigma`) shared by the `test_waveform_fourier_*` modules.
- `tests/analysis/analysis_strategies.py`: analysis spec factories and constants
  (`POWER_BINS`, `CELL_FS`, `CELL_FRAME`, `capture_of`, `registered_tolerance`, the SSB
  spectrogram fixtures).
- `tests/sensor/sweep_strategies.py`: sensor spec factories and constants (`SOURCE`, the
  function-generator source spec nine modules share; `make_capture`/`make_sweep`/
  `make_sweep_kws` generic builders; `SPG`, `RESOLUTION_MSG`; the `QuickAnalyses` and
  `CellularAnalyses` groups with the unregistered `quick_binding` and
  `restricted_cellular_binding`).
- `tests/sensor/synthetic_sources.py`: the sensor harness — the four capture presets,
  `preset_capture`, `preset_sweep`, `tone_capture`, `tone_captures`, `tone_sweep`, `fs_sdr`,
  `expected_raw`, `build_acquired_iq`, `acquire_corrected`, `expected_corrected`,
  `run_in_memory`.
- `tests/sensor/soapy_factories.py`: `MCR` and the soapy spec factories; `fake_soapy.py`: the
  fake `SoapySDR` module (`stub_controller`/`receive_buffers` for buffer tests live in
  `test_sensor_sources_buffers.py`).

## Inputs and tolerance models

- Build inputs from `striqt.analysis.testing` (`tone`, `single_tone`, `sawtooth`,
  `dirac_delta`, `circular_awgn`): backend-agnostic, windowable, and the same definitions the
  synthetic sensor sources use, so a two-port generator call reproduces a two-port
  acquisition. They use `numpy.random.RandomState` because that is the API numpy and cupy
  share.
- Tolerances come from the library's own models, which are the contract the suite tests to:
  `striqt.waveform.power_analysis` (`log_conversion_tol`, `bin_power_rtol`, `bin_power_rms`,
  `stat_rtol`, `level_tolerance_dB`, `off_peak_dB_tolerance`, …), `striqt.waveform.fourier`
  (`fft_tolerance_rms`, `peak_factor`, `on_peak_roundoff`, `off_peak_floor_dBc`),
  `striqt.waveform.arrays` (`unit_roundoff`, `accum_rms`, `accum_rtol`, `mean_atol`),
  `striqt.waveform.lib.comms` (`corr_atol`); a measurement's budget is its registered tolerance
  (`registered_tolerance` in `analysis_strategies`, `capture_tolerances`/`sweep_tolerances` in
  `striqt.sensor.lib.compute.tolerance`). Module-level constants inside `striqt.waveform.lib`
  (`ROUNDOFF_SAFETY` and friends) are implementation details: never import them.
- Exact expectations the arithmetic allows: a bin-centered unit tone through a boxcar window
  reads exactly 0.0 dB in one bin; a constant envelope puts exactly 1.0 of a normalized
  histogram in one bin; a unit-amplitude waveform reads exactly 0.0 dBm, since the power
  measurements treat `|iq|**2` as mW.

## Backends and process-global state in tests

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
does not recognize `pytest_generate_tests` parametrization).

## pytest extensions

`[project.optional-dependencies].test` pins `pytest`, `hypothesis`, `pytest-subtests` (py39
only; the fixture is core in pytest 9), `pytest-lazy-fixtures`, `pytest-cov`,
`coverage[toml]`, `nptdms`. Do not add `pytest-regressions`, `pytest-cases`, `pytest-check` or
`pytest-mock`: golden files would hide the derivation that makes a tolerance reviewable, and
the other three duplicate `parametrize`, `subtests` and `monkeypatch`.

## The xfail ledger

`tests/xfail-audit.md` is the registry of strict xfails. An entry gives the node id, the
mechanism with file and line, why the test's expectation is the right one, the impact tier,
and the fix with its difficulty; its numbered items are stable identifiers, and a suite-state
paragraph records the `passed/skipped/xfailed` counts of each pass. `reason=` names the
mechanism and stays verbatim once written. A fix turns the xfail into an unexpected pass:
remove the marker and retire the item in the same change, and update the item when a rename
or fold changes a node id it names. A run with any xpass is a failed run.

## Verifying a test change

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
