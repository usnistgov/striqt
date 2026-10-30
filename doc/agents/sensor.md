# Sensor architecture

`striqt.sensor` binds a source backend, peripherals, spec types, and an action into a
named *sensor*, and runs actions against that binding through a threaded acquire,
analyze, and write pipeline. This page describes the binding mechanism, the action
runners, the per-capture signal path, and the process-global logging and concurrency
state that the pipeline relies on. Paths are relative to `src/striqt/`.

## Sensor bindings

A *sensor* is a binding of (source backend, peripherals, spec types, action).
`sensor/lib/bindings.py:bind_sensor` pairs a `Sensor` (implementation classes) with a
`Schema` (spec types) and registers a `Controller` subclass under a string key.

Each binding gets one tagged subclass of the action spec type `Schema.action` (default
`specs.Sweep`). A single `msgspec.defstruct` builds that subclass and re-annotates
`source`, `peripherals`, `captures` (when the action declares that field), and
`analysis` (when the action is a `Sweep`) with the types of the schema. The tagged
classes of registered bindings are unioned into the type returned by
`get_tagged_sweep_type()`.

The tag is the `sensor_binding:` key at the top of every sweep YAML. The tag selects
the capture, source, peripheral, and analysis spec types that the rest of the file is
validated against. Each binding key maps to one action: `air7101b_calibration` is a
separate key from `air7101b`.

The bound class has 4 class attributes:

- `sensor`: the `SensorBinding`.
- `schema`: the `Schema`.
- `controller`: the registered `Controller` subclass.
- `sweep_cls`: the bound `Sweep` class with the same spec types. When the action is a
  `Sweep`, `sweep_cls` is the class itself. Otherwise `sweep_cls` is an untagged
  companion named `<key>.sweep`.

### Action specs

Action specs form a data-only chain in `specs/structs.py`:

- `ActionSpec`: `description`, `source`, `peripherals`, `extensions`, and `sink`. The
  `sink` is a `Sink` with `path`, `log_path`, and `log_level`.
- `CapturePlan`, a subclass of `ActionSpec`: adds `captures`, `loops`, and
  `adjust_captures`, with the pure loop checks.
- `Sweep`, a subclass of `CapturePlan`: adds `plot_hint`, `analysis`, and `options`,
  and narrows `sink` to `ZarrSink`, which adds the zarr store fields. `validate_sweep`
  checks a `Sweep`.
- `ManualYFactorCalibration`, a subclass of `CapturePlan`: adds `calibration`. The
  `analysis` field is deprecated and ignored.

`specs.sequencing._expansion_args` takes `loop_only_nyquist` explicitly, so a
`CapturePlan` expands without `options`. `specs/paths.py:PathFormatter` reads the action
spec. The class name of the action spec is the `{sensor_binding}` field.

## Actions and runners

An *action* is an action spec type paired with a runner. `sensor/lib/actions/registry.py`
maps action spec types to runner classes through `@registry.runner(spec_type=...)`. The
lookup walks the MRO of the bound class.

`actions/core.py:ActionRunner` is the `Protocol` that every runner satisfies. A runner
is constructed from the action spec, decoded from a file or built from the API, and
exposes:

- `spec`: the action spec. Construction stores the spec and does nothing else.
- `sweep`, a read-only property: the `Sweep` that `open_resources` opens the hardware
  for. Reading the property runs the derivation (`build_calibration_sweep`,
  `build_cellular_sweep`, both `lru_cache`d on the spec) and raises when the spec
  cannot be converted. A bound spec reads `sweep` at the end of `__post_init__`
  through `specs.action.validate_action`, so a bound spec that constructs is one that
  the runner can run. `SweepRunner.sweep` is the spec itself.
- `open_sink(format_path)`: returns the sink that `open_resources` enters. A
  `sink.path` of None writes nothing.
- `iterate(resources)`: runs the action and invokes the `on_analyzed` and `on_sink_cached`
  callbacks of `resources['callbacks']`.

Nothing is inherited between runners. Each runner calls `iterate_captures`, the pipeline
described under Sweep execution below, with the per-capture `extra_data=`
hook and a shared `indexer=`.

`ActionSpec.check_schema(schema)` is a classmethod hook that `bind_sensor` calls, so an
action can reject spec types at bind time. The calibration requires
`channel_power_time_series` in `schema.analysis`. The cellular sweep requires every
`Cellular5GNRAnalysis` field. The runners build a generated analysis as the group of the
binding through `specs.action.analysis_group_type(spec)`, which is `BuiltinAnalyses` for
an unbound spec.

### The sweep runner

`SweepRunner` (`actions/sweep.py`) runs the spec itself. The runner opens a
`ZarrCaptureSink`, or a `NoSink` when `sink.path` is None.

### The Y-factor calibration runner

`ManualYFactorRunner` (`actions/yfactor.py`) builds a sweep with the `lru_cache`d
`build_calibration_sweep`. The built sweep has a generated `channel_power_time_series`
analysis, `reuse_iq=True`, `loop_only_nyquist=True`, `skip_warmup=True`, and the `Sink`
of the spec widened to a `ZarrSink`. `iterate` then runs the passes:

1. Confirm the ENR with the operator.
2. For each port and diode state, prompt the operator, call `Controller.discard_iq`,
   and run the captures of that port through `iterate_captures` with
   `{'noise_diode_enabled': state}` as the `extra_data` mapping of the pipeline. The
   diode state is invisible to `buffers.is_reusable`, so the cached acquisition must not
   cross a switch.
3. Collect the captures of every pass in the `MemoryCacheSink` (`lib/sinks.py`) that
   `open_sink` returns.
4. After the last pass, `yfactor.save_yfactor_corrections` indexes the captures by the
   fields that `yfactor.calibration_fields(spec, sweep)` returns, computes the
   corrections with `compute_y_factor_corrections`, and writes the netCDF.

The Y-factor math is in `actions/yfactor.py`. The lookups that measurement sweeps use
are in `lib/compute/cal_lookups.py`. There is no calibration peripheral, derived capture
class, or toggle loop.

A calibration is bound with
`bind_sensor(name, ctrl_cls.sensor, dataclasses.replace(ctrl_cls.schema, action=ManualYFactorCalibration))`.
`bind_manual_yfactor_calibration` (in `lib/bindings.py`, re-exported by
`striqt.sensor.calibration`) does that and emits a `DeprecationWarning`.

## The cellular sweep

`Cellular5GNRSweep` (`specs/structs.py`) and `Cellular5GNRSweepRunner`
(`actions/cellular.py`) are the third action.

The `captures` of a `Cellular5GNRSweep` are the capture type of the binding and name a
channel by `center_frequency`. `check_schema` requires the capture type to have that
field. The root `cell_channels` table maps a label to a `CellChannel`:
`channel_bandwidth`, `subcarrier_spacing` (30 kHz only), `ssb_frequency_offset`, the
guard bands, the TDD pattern, and `min_sample_rate`. The `analysis` is a
`Cellular5GNRAnalysis` of optional blocks named for the measurements that the blocks
enable. Each block declares only the free parameters.

Each block is declared with `@subset_of(<measurement spec>, without=..., optional=...)`
(`analysis/specs/helpers.py`). The decorator verifies at class creation that the kept
fields match the measurement spec in type and default, and records the relation for
`spec_subset`. `actions/cellular.py` gives the derivations of the omitted parameters in 3
tables:

- `FIXED`: parameters that the subcarrier spacing fixes.
- `DEFAULTED`: parameters that stay exposed as `Optional` and take the derived value
  when None.
- `PER_CAPTURE`: parameters written into the `adjust_analysis` of each capture from the
  channel of that capture.

An omitted parameter in none of the tables stays at the measurement default.
`check_block_fields` verifies at import that the tables agree with the declaration of
each block.

`build_cellular_sweep` is cached and runs when the runner's `sweep` is read at spec
construction. The function resolves each capture against the table:

- `sample_rate` comes from `striqt.waveform.comms_5gnr.nr_sample_rate`: the smallest
  multiple of 896·SCS at or above the channel bandwidth plus the FIR transition band.
  The 3GPP TS 38.104 table in `phy_3gpp.py` validates the bandwidth.
- `analysis_bandwidth` equals the channel bandwidth.
- The cell parameters are written into `adjust_analysis` under the keys of the capture.

The function then expands the analysis block into an instance of the analysis group of
the binding from those tables: SCS/2 resolution with the 13/28 and 15/28 OFDM grid, 1
resource block and 1 subcarrier integration bandwidths, a 10 ms cyclic period, and a 1
symbol detector period.

The derived `Sweep` keeps the loops and `adjust_captures` of the spec. A loop over
`center_frequency` is rejected: captures are the innermost iteration, and the channels
are listed as captures instead.

Bind the action with `bind_sensor` and `action=Cellular5GNRSweep`. The built-in keys
are `air7101b_cellular`, `air7201b_cellular`, and `air8201b_cellular` in
`sensor/bindings.py`.

## Extensions and the downstream project

Built-in bindings are in `sensor/bindings.py`. Out-of-tree bindings are loaded through
the `extensions:` block of a sweep spec, which inserts a path and imports a module
before the rest of the spec is decoded (`sensor/lib/io.py:_import_extensions_from_spec`).
`bind_sensor` records the module of the caller on the controller class (`module=`
overrides that), so that the API docs group derived bindings with the extension that
defines the derived bindings.

A derived binding changes the action or the analysis group with
`dataclasses.replace(ctrl_cls.schema, action=..., analysis=...)`.
`tests/sensor/sweeps/src/extensions.py` binds `site_quick` that way over
`SiteQuickAnalyses`, and `tests/sensor/sweeps/site/site-quick.yaml` runs through the
CLI test.

The principal downstream user is the aggregate-directivity-acquisition project, which
binds site-style sensors through `extensions:` and runs on a Jetson with cupy. The
environment of that project does not install on a laptop, so the patterns of that
project are reproduced in-tree: `tests/sensor/sweeps/src/extensions.py` and the `site*`
sweep directories under `tests/sensor/sweeps/`. The GPU path is the production path but
cannot be executed on a laptop. A cupy-only change is unverified until someone runs
`pixi run -e cupy pytest` on the Jetson. Cupy tests skip locally through the
`_cupy is None` check in `tests/conftest.py`.

## Sweep execution

`open_resources(spec, callbacks=...)` builds the runner registered for the type of the
spec and opens the source, peripherals, and sink for `runner.sweep`. The returned
`Resources` dict has the keys `action_spec` (the spec passed in), `sweep_spec` (the
built sweep), `runner`, and `callbacks`.

A front end registers hooks for one run as keywords of `open_resources`, each one
callable or None: `on_source_opened(sweep, source_id)` runs from a worker thread after
the source opens; `on_analyzed(dd)` receives the `DelayedDataset` of each capture in the
analysis thread, after `compute.analyze` and before the dataset is submitted to
`sink.append`; `on_sink_cached(resources, cache, capture, result, **kwargs)` runs in the
analysis thread each time a measurement fills a `KwArgCache` (the pipeline forwards the
cache's own callback arguments and prepends `resources`, since the SNR line needs the
calibration, `format_path`, and `source_id`). A front end that wants two behaviors on
one event writes one function. A callback exception ends the run like an analysis
error. No hooks are registered by default. `open_resources` bundles the three into the
internal dataclass `lib/resources.py:SweepCallbacks` under `resources['callbacks']`,
which `iterate_captures` reads; `iterate_sweep(res, callbacks=...)` replaces that
bundle for one run.

`lib/compute/summary.py` returns structured data and never logs: `overloads(dd)` gives
one `PortOverload` per port (`adc`, `im3_victims`) and `spectrogram_snr(sweep, capture,
result)` one `SpectrogramSNR` per port with a finite SNR, None without a calibration;
the `describe_*` functions render each as one line. The logging callbacks are CLI code:
`cli/sensor_sweep.py:log_overloads` (`on_analyzed`) and `log_spectrogram_snr`
(`on_sink_cached`, gated on `'spectrogram' in cache.name`), which `sensor-sweep` registers
beside `confirm_labels` and `web-plotter` imports and calls from its own callbacks.
`from_delayed` emits none of these messages.

`iterate_sweep(resources)` calls `resources['runner'].iterate`. Every runner ends in
`actions/core.py:iterate_captures`, the three-stage software pipeline in threads
(acquire, analyze, write). The docstring of `iterate_captures` diagrams the stage
offsets. The `extra_data` of the runner is merged at the end of each acquisition, and
the fields of the action spec are the dataset attrs. GPU analysis deliberately runs in
the foreground thread.

Captures are expanded from `captures:` plus `loops:` (`Range`, `List`, `Repeat`,
`FrequencyBinRange`) by `specs.sequencing.loop_captures`. The per-source overrides from
`adjust_captures` are then applied.

Output is written by `ZarrCaptureSink` (`lib/sinks.py`) into a zarr store that the sink
allocates at `open` from the sweep spec: one row per port of every expanded capture
along `capture`, every data variable and coordinate at full length, and no data.
`compute/datasets.py:sweep_template` builds that zero-row dataset without IQ: the axis
lengths and fixed coordinates come from the coordinate factories of the registry, the
per-capture coordinates from the capture type and the `acquisition_info_cls` of the
source backend (`sources/base.py`), and captures with different grids are unioned per
dimension as `xr.concat(join='outer')` would. `analysis/lib/io.py:allocate_store` writes
the metadata and resizes the arrays. Each `append` writes the arrays of the
`DelayedDataset` into the chunks it completes (`ZarrSink.max_chunk_bytes` bounds a
chunk), queues the rest, and adds the attrs that measurement bodies return, dropping a
key that differs between captures. Store metadata is staged in memory by
`analysis/lib/stores.py` and flushed after every capture for a directory store and once
at `close` for a `.zarr.zip`, which is written in place, so no store key is written
twice. `close` writes the partial chunks, trims the arrays to the rows written, and
consolidates the metadata. `ZarrSink.batched_write_count` is deprecated and ignored.
`ZarrTimeAppendSink` instead appends along the spectrogram time axis and is reached only
by passing an instance through `iterate_sweep(res, sink=...)`; `Zipper` remains for the
`zip-zarr` CLI.

### Timestamps and the trigger strobe

`SourceBackend.read` returns the time (ns) of the sample at ``buffers[i][offset]``. The
first read of a capture begins one lead overlap before the record, so the stamp precedes
corrected sample 0 by `lead / fs_sdr`.

`Controller.read_iq` passes that stamp and `overlaps[0]` to
`sources/buffers.py:find_trigger_holdoff`, which returns the number of source samples
dropped before the record: at least the lead, plus `transient_holdoff_time` after a
re-arm (no carryover), rounded up to the next `trigger_strobe` edge when a strobe is
set. The record start time is the stamp plus the holdoff.

With `gapless`, `ReceiveBuffers.stash_carryover` keeps the trailing lead and tail of the
record and stamps the first carried sample one lead before the next record. `apply`
returns that stamp for the next capture in place of a source read.

`sources/base.py:VirtualSource` keeps a synthetic clock. The clock is 0 at the first
`arm`, so corrected sample 0 of the first capture is on every strobe grid. A re-arm
continues the stream: the first read after `arm` is stamped where the previous samples
ended, and the origin of the new capture is one new-rate lead overlap later.
`ZarrIQSource.read` replaces the stamp with the store's `start_time` less the lead, so
`start_time` stays the time of file index 0.

### The per-capture signal path

The conditioning runs in `sensor/lib/compute/corrections.py:Preprocessor`, reached
through `AcquiredIQ.preprocess` on the sensor `AcquiredIQ`
(`sensor/specs/dataclasses.py`). Each stage runs on the first call that needs the
stage and is memoized on the `Preprocessor`, one per acquisition. The preprocessor
holds the acquisition through a weak reference. A `dataclasses.replace` copy,
including the controller's `reuse_iq` copy, starts with a fresh preprocessor. The
stages, in order:

1. Resample to `sample_rate`: `_scale_only` when no resampling is needed, otherwise
   `_oaresample`, or `_resample` when `STRIQT_USE_OARESAMPLE=0`. This is the only
   stage that reads `x`, and the one that may consume `x` in place under
   `overwrite_x`.
2. Conjugate for a high-side LO. `STRIQT_IGNORE_HIGHSIDE_LO` disables the conjugation.
3. Low-pass to `analysis_bandwidth`: the response on the STFT bins inside
   `_oaresample`, or the 4001-tap FIR when `STRIQT_USE_OARESAMPLE=0` or the capture
   is not resampled, with the same band edges (`FIR_TRANSITION_BW` centered on half
   the bandwidth). The FIR runs only for a measurement registered
   `prefer_filtered=True` on a finite bandwidth; `_oaresample` filters the resampled
   waveform for every measurement.
4. Trim to `duration`.
5. Run the `signal_trigger` and shift. This stage runs for a measurement registered
   `prefer_aligned=True` when a trigger is set. The trigger measurement receives the
   `AcquiredIQ`, and the registry selects the input of the trigger with the
   preferences registered on the trigger. A trigger registered `prefer_aligned=True`
   is rejected at import.

`compute/analyze.py:analyze` installs the trigger and `overwrite_x` with
`attach_corrections`, evaluates the measurements, and packs a `DelayedDataset`.

### Warmup

`prepare_compute(sweep)` runs `actions/warmup.py:warmup_sweep_for`. The result is `None`
under `options.skip_warmup` or when `compute/gpu.py:sweep_touches_gpu` is false.
Otherwise the result is a one-capture `NoSource` `Sweep` derived by
`actions/warmup.py:derive_warmup_sweep`. The warmup sweep is bound per origin through
`warmup_binding`, so that the captures of the warmup sweep keep the capture type of the
origin. `open_warmup_resources` opens the `NoSource`, `NoPeripherals`, and `NoSink`
resources, and a `SweepRunner` runs the warmup sweep, paying the JIT and import costs
before the real captures.

## Logging and concurrency

Loggers are `logging.LoggerAdapter`s (`analysis/lib/util.py:StriqtLogger`) registered by
bare suffix in `_logger_adapters` and fetched with `get_logger(name)`. The `'analysis'`
logger is created when `striqt.analysis` imports. The `'sweep'`, `'source'`, `'sink'`,
and `'periph'` loggers are created when `striqt.sensor.lib.util` imports. That import
also calls `show_messages(logging.INFO)` and so installs stderr handlers as an import
side effect.

The underlying loggers are named `striqt.<suffix>`, so a handler on
`logging.getLogger('striqt')` sees every one of those loggers. `log_to_file` collects the sweep log
through such a handler.

The adapter `extra` dict has the keys `capture_index`, `capture_count`, and
`capture_progress`, which the screen format prints. `sensor.lib.util.log_capture_context`
swaps the `extra` dict per capture.

`stopwatch` logs at the custom levels `PERFORMANCE_INFO` (15) and `PERFORMANCE_DETAIL`
(12). `log_verbosity(-1..3)` maps the CLI `-v` count to those levels.

`log_to_file` is wired from `Sink.log_path` and `Sink.log_level` by `open_resources`.
The records are written as a JSON array, which is also valid YAML.

All of this is process-global state. Tests that touch the logging state must snapshot
and restore levels, handlers, and `extra`. The autouse `restore_logging_state` fixture
in `tests/conftest.py` does this for every test.

### Concurrency

`analysis.util.compute_lock(array)` is an `RLock` taken only for cupy arrays, or when
no array is passed. The lock serializes GPU work across threads.

`sensor/lib/util.py` provides the cooperative cancellation that the pipeline uses:

- `cancel_threads()` sets a global event.
- `propagate_thread_interrupts()` raises `ThreadInterruptRequest` in worker threads
  only, never in the main thread.
- `share_thread_interrupts()` clears the event on exit.

`ExceptionStack.defer()` collects exceptions from several threads. `handle()` re-raises
one exception or an `ExceptionGroup`, and interrupts yield to real errors.
`await_and_ignore` applies an `ExceptionStack` to a list of futures.

`threadpool` is a lazily created module attribute of `sensor/lib/util.py`, created on
first access through the module `__getattr__`.

`DebugOnException` prints IPython-style tracebacks (`sensor/lib/tracebacks.py`) and is
the top-level handler of the CLI.
