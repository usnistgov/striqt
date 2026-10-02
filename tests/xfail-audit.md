# Audit of strict xfails

Every xfail in this repo is `strict=True` and is written against the behavior
the code was *meant* to have, so the list doubles as the known-defect registry
(see CLAUDE.md, "Known gaps are recorded as strict xfails"). This document
records, for each open one, why the failing test is believed to describe a real
defect rather than a wrong expectation, and how much it matters to the way the
library is actually used.

Items are numbered in the order they were found: 1 to 41 on 2026-09-14 (branch
`expand-waveform-tests`), 42 to 61 on 2026-09-15 and 16 from the SoapySDR and
Y-factor calibration test build-out (branch `test-soapy-and-cals`, confirmed
against the closed-form receiver model in `sweep_strategies` and the fake
`SoapySDR` module, not on hardware), and 62 to 73 on 2026-09-16 from the value
audit of the pre-existing tests, which inverted passing tests that had pinned
defective behavior into xfails against the intended behavior (that audit's own
write-up was retired on 2026-09-18 once its actions had all landed; its three
remaining observations are in the list at the end of this file). Items 74 to 76
were added on 2026-09-17 while writing the first direct tests of the
`striqt.analysis` measurements, and all three were fixed on 2026-09-17 and 18.
Items 77 to 89 were added on 2026-09-18 by the refocus of `tests/sensor/` on the
sensor signal path (uniform time origin, `correct_iq` stage values, buffers,
controller, pipeline, sinks, file sources), which also fixed thirteen defects in
the same change without giving them a number (listed under "Fixed on 2026-09-18
without a ledger number" below). Item 90 was added on 2026-09-23 by the
sweep-validation pass, and item 91 on 2026-09-24 as the first entry under "Tool
limitations": a gap in the type checker rather than in striqt, kept as a strict
xfail so that the `ty` upgrade that closes it surfaces as an xpass. The
numbers are stable identifiers, so the gaps mark items that have since been
fixed or withdrawn (1 to 5, 7, 11, 12, 15, 21, 25, 26, 29, 35 to 37, 40, 42, 47,
51, 60, 61, 72 to 76, and 89); their write-ups are in the git history of this file. Items
are listed in priority order, not numeric order.

Suite state on 2026-09-22, after the pass that consolidated the duplication this
branch introduced: `1685 passed, 395 skipped, 94 xfailed, 305 subtests` on
`test39` and `1686 passed, 394 skipped, 94 xfailed, 305 subtests` on `test314`
(one xarray-version-gated test skips on py39, and one passes there only since the
canceled-lookup fix of 2026-09-21), from 76 xfail sites (`grep -rn "mark.xfail"
tests --include='*.py' | wc -l`; a site that marks a parametrized test counts
once). No xpass.

Suite state on 2026-09-23, after the sweep-validation pass (`waveform-validators`):
`1925 passed, 395 skipped, 90 xfailed, 311 subtests` on `test39` and
`1926 passed, 394 skipped, 90 xfailed, 311 subtests` on `test314`. Four strict
xfails turned into passes and their markers and entries were removed: item 30
(`design_oafilter` rect/None, the `if`/`elif` chain), item 32 (`iq_to_cyclic_power`
truncation along the wrong axis) and item 33 (`iq_to_cyclic_power` 1-D input) — the
latter two by one fix that tests and slices along `axis`. Item 34 (negative axis)
still xfails. The pass added the corrections checks of `validate_sweep` (sensor) and the
validator-passes-implies-measurement-runs property test (analysis).

Suite state on 2026-09-24, after the typing pass (ty-checked signatures in
`tests/test_typing.py`, hints on the functions added since 2026-09-03, rule-named
suppressions): `1938 passed, 396 skipped, 91 xfailed, 311 subtests` on `test39` and
`1940 passed, 394 skipped, 91 xfailed, 311 subtests` on `test314`, from 73 xfail sites.
Relative to 2026-09-23 the new module adds 6 (py39) or 7 (py314) passing node ids (one
per probe file under `tests/ty_probes/`, the registry-generated probe, and on py314 the
`ty check src` clean run, which is skipped on py39: `skipped` 395 to 396), plus the
strict xfail of item 91 (`xfailed` 90 to 91); the remaining +7 passes on each
interpreter come from the measurement-test regrouping commits (`901fe2d9` to
`14612673`) that postdate the 2026-09-23 line. The 76 sites quoted on 2026-09-22
predate that regrouping's marker folds; `grep -rn "mark.xfail" tests --include='*.py'
| wc -l` gives 72 before the new module and 73 with it. No xpass.

Suite state on 2026-09-24 (evening), after the pyright/pyrefly hint pass and three
defect fixes: `1944 passed, 396 skipped, 89 xfailed, 311 subtests` on `test39` and `1946
passed, 394 skipped, 89 xfailed, 311 subtests` on `test314`. `xfailed` fell 91 to 89:
item 53 lost its `test_port_change_rebuilds_the_stream` node when `RxStream.setup` was
fixed to close and re-open the stream on a new port set (its `test_capture_changes_port`
node still xfails, the `==`/`!=` inversion is untouched), and item 72 was withdrawn: the
tuple form of `adjust_captures` was never a feature (the union alternative and the
`_get_capture_adjust_map` branch were unfinished strays from unrelated commits), so the
type is narrowed back to the mapping form and the former xfail is now a rejection row of
`test_invalid_adjustments_rejected`. The other passes come from the new
`retry(tries < 1)` tests and the stream re-setup test. No xpass.

Suite state on 2026-09-28, after the `check-sweep` pass: `1955 passed, 396 skipped, 88
xfailed, 311 subtests` on `test39` and `1957 passed, 394 skipped, 88 xfailed, 311
subtests` on `test314`, from 70 xfail sites (`xfailed` 89 to 88 for item 11; the
eleven new passes are the tolerance, default-path and coordinate tests). `sweep_tolerances` now budgets `loop_captures` unchanged
instead of re-applying `adjust_captures` over the looped fields (three new tests in
`test_sensor_compute_tolerance.py` and
`test_check_sweep_budgets_the_looped_field_over_its_adjustment`). Item 11 is retired:
the `Sink.path` default is `{spec_name}-{start_time}`, matching `get_path_fields` (the
`doc/reference-sweep.yaml` the old entry cited no longer exists), so a spec without a `sink:` block opens
(`test_check_sweep_formats_the_default_sink_path`). `list_capture_adjustments` follows
the same loop-wins rule as `loop_captures` (D28): a field that is both looped and
adjusted lists the loop's values in loop order, and an adjusted field no loop touches
lists its adjusted value (`test_check_sweep_lists_the_looped_field_over_its_adjustment`).
No xpass.

Suite state on 2026-09-28 (later), after the Tier D test consolidation (`85ec5315`,
`23d8fc78`): `1936 passed, 400 skipped, 88 xfailed, 311 subtests` on `test39` and `1938
passed, 398 skipped, 88 xfailed, 311 subtests` on `test314`. Collected ids fell 2447 to
2424: the 24-row `test_validator_rejects_at_the_analysis_path_naming_the_field` table in
`test_analysis_register.py` kept the two rows that exercise distinct message axes (every
removed row has a message-matching twin in its measurement module), `test_flush_saves_a_file`
went as redundant with the calibration lookup tests, and three pairs or triplets folded into
parametrized tests (`test_timestamps_advance_by_the_samples_read[sawtooth|no_source]`,
`test_get_array_namespace[numpy|cupy]`, `TestSweepAnalysisValidation::
test_an_invalid_combination_names_its_origin[looped_capture|adjust_analysis|analysis_loop]`).
No xfail node id changed; `xfailed` and the 70 sites are unchanged. No xpass.

Suite state on 2026-09-28 (evening), after removing `striqt.analysis.testing.noise`, a
pure wrapper of `circular_awgn` (`power=noise_psd * sample_rate`): `1909 passed, 376
skipped, 88 xfailed, 311 subtests` on `test39` and `1911 passed, 374 skipped, 88 xfailed,
311 subtests` on `test314`. The 51 node ids that went were the `[noise]` rows of the
`GENERATORS`-table tests in `test_analysis_testing.py` (half of them the skipped cupy
variants) and `test_noise_is_circular_awgn_at_the_integrated_power`, which asserted the
relation the wrapper was; every other caller now calls `circular_awgn` with the scaled
power. No xfail node id changed. No xpass.

Relative to the msgspec-idiom message pass (`1667/395/94/305` and
`1668/394/94/305`), both interpreters gained 18 passes and nothing else moved:
+6 in `test_sensor_specs_structs.py::TestSweepLoops` for the new
`_validate_loop_capture_collisions` and the sharpened duplicate-loop message
(5 functions, one parametrized on a leading `repeat`), +4 in
`test_sensor_specs_sequencing.py` for the `only_fields` loop-index regression
(parametrized), the located missing-required-field error, and a bare `Repeat`
with no captures, and +8 for the new `TestSlotPeriod` in
`test_waveform_ofdm.py` covering the extracted `ofdm.slot_period`. `skipped`,
`xfailed` and `subtests` did not move, so no cupy case was lost and no marker was
added or removed. The three `_validate_range` tests in
`TestCellularCyclicAutocorrelator` were rewritten against
`SpecValidationError` rather than added, which is why they contribute 0.

Relative to the capture-provenance pass that added `loop_capture_origins`
(`1666/395/95` and `1667/394/95`), that earlier pass gained one pass and lost one
xfail: fixing the `"global"` wording in the `adjust_captures` source-key error
retired item 40, whose marker came off `test_source_key_error_names_defaults`.

Earlier in the branch, the pass count rose by 44 from commit 7f56f985 to
`1667`/`1668`, which is 47 new node ids less 3 retired: +14 for the new
`tests/analysis/test_analysis_register.py`, +6 for the new
`tests/waveform/test_waveform_util.py` covering `persistent_cache`'s
best-effort shelf, +10 net in
`test_analysis_specs_helpers.py` for `SpecValidationError.locations`/`.at()`
(two of its existing tests gained parametrize rows and one whose case folded
into a row went away), +2 parametrize rows on
`TestSpectrogram::test_non_integer_binning_raises`, +12 for
`loop_capture_origins` and `describe_capture_origin` in
`test_sensor_specs_sequencing.py`, and a net 0 in `test_sensor_specs_structs.py`
where `test_an_invalid_looped_capture_names_its_expanded_index` was renamed to
`..._names_its_captures_entry_and_loop_point`. `skipped` and `xfailed` did not
move, so no cupy case was lost and no marker was added or removed.
Every strict xfail becomes an unexpected pass when its defect is fixed and must
have its marker removed in the same change.

## Method

For each xfail I:

1. Re-read the test and the library code it targets, and confirmed the failure
   mechanism named in the xfail `reason` against the source (file:line below).
2. Asked what evidence says the *test's* expectation, rather than the current
   behavior, is the intended one: a docstring, a type annotation, a 3GPP
   clause, the YAML reference in `doc/`, a sibling code path that already does
   it right, or how the sensor pipeline and the site-style sweeps consume it.
3. Traced callers (`grep` over `src/` and `tests/sensor/sweeps/`) to decide
   whether the production path reaches the defect. "Production path" means
   `sensor-sweep` → `read_yaml_spec` → `open_resources` → `iterate_sweep` →
   `correct_iq` → registered measurements. The `site*.yaml` sweeps and
   `tests/sensor/sweeps/src/extensions.py` stand in for the downstream
   aggregate-directivity-acquisition (ADA) project, whose environment does not
   install here.
4. Where cheap, reproduced the defect end to end in the `test39` environment
   (noted as "reproduced").

### Confidence

- **High**: mechanism confirmed in source, and the intended behavior is
  documented or forced by a sibling code path.
- **Medium**: mechanism confirmed, but the intended behavior is inferred from
  convention (numpy semantics, an unused parameter, an unreachable branch)
  rather than stated anywhere.

Every xfail reason was found to be accurate, with two understatements noted
inline (cyclic-power negative axis, `oafilter` gain).

### Impact

- **A**: produces wrong output or a crash today for the in-tree site sweeps or
  the downstream project's documented usage.
- **B**: reachable from a plausible sweep YAML or from a spec field's default
  value; not exercised by any in-tree sweep.
- **C**: reachable only through the Python API, dead code, or the GPU path with
  an argument nothing passes.
- **D**: cosmetic (message text, class name, log formatting).

Priority is the joint ranking: A before B before C before D, and within a tier
High confidence before Medium, then by severity of the consequence. No Tier A
item is open.

## Summary table

| # | Test | Confidence | Impact | Fix |
| --- | --- | --- | --- | --- |
| 43 | `test_finite_bandwidth_with_a_small_resampler_fft[*]` (3 parameter sets) | High | B | easy |
| 77 | `test_overlaps_are_even` (hypothesis); `test_impulse_acquisition[odd_overlap]` | High | B | small |
| 78 | `test_lo_shift_is_removed_from_the_output[left]`, `[right]` | High | B | design |
| 79 | `test_reused_iq_can_be_corrected_for_a_wider_analysis_filter` | High | B | design |
| 90 | `test_upsampled_overlap_covers_the_output_rate_filter_pad` | High | B | small |
| 66 | `test_zero_stft_by_freq_zeroes_outside_passband[off_grid]`; `test_downsample_stft_passband_zeroing[off_grid]` | High | B | easy |
| 6 | `test_subframe_cp_layout[15kHz]`, `[60kHz]`; `test_slots_tile_the_frame[15kHz]`, `[60kHz]`; `test_lte_and_5g_agree_at_15khz` | High | B (spec default) | moderate |
| 8 | `test_adjust_captures_missing_required_default_lookup_raises` | High | B | trivial |
| 9 | `test_remap_keyed_on_an_unknown_field_is_rejected` | High | B | trivial |
| 13 | `test_fft_bins_odd_count_even_length` | High | B | easy |
| 44 | `TestLookupPowerCorrection::test_recovers_the_receiver_gain[single_frequency_calibration]` | High | B | trivial |
| 52 | `TestSoapySourceClose::test_close_releases_the_stream_and_device` | High | B | trivial |
| 14 | `test_leading_repeat_is_accepted` | High | B | easy |
| 16 | `test_remap_keyed_on_a_fixed_numeric_alias_accepts_numeric_keys` | High | B | trivial |
| 17 | `test_nested_include_resolves_relative_to_the_including_file` | High | B | moderate |
| 18 | `test_absolute_include_outside_root` | High | B | easy |
| 62 | `test_three_deep_adjust_analysis_is_hashable` | High | B | small |
| 65 | `test_no_overlap_roundtrip_is_identity` | High | B | design |
| 67 | `test_cuda_unsorted_indices` | High | B (GPU, unverified) | trivial |
| 59 | `test_probe_soapy_info_with_a_channel_sensor` | High | B (latent) | trivial |
| 19 | `test_infinite_sample_rate_reports_sample_period_message` | High | B (unlikely) | trivial |
| 10 | `test_chained_remap_declared_before_its_key_resolves` | Medium | B | moderate |
| 45 | `TestLookupPowerCorrection::test_recovers_the_receiver_gain[calibration_without_a_lo_shift_loop]` | Medium | B | easy |
| 80 | `test_oaresample_impulse_lands_on_its_output_sample[resample_filter]`, `[resample_only]` | High | C | moderate |
| 81 | `test_oaresample_acquisition[resample_filter]`, `[resample_only]` | High | C | small |
| 82 | `test_zarr_resampler_pins_file_rate` | High | C | design |
| 83 | `test_mat_two_port` | High | C | easy |
| 84 | `test_mat_request_past_end` | High | C | easy |
| 85 | `test_test_only_resources_run_a_sweep` | High | C | easy |
| 86 | `test_get_trigger_from_an_analysis_group_trigger` | High | C | easy |
| 87 | `test_from_delayed_accepts_the_acquisition_info_defaults` | High | C | trivial |
| 88 | `test_build_capture_coords_adds_a_window_loop_coordinate` | High | C | moderate |
| 53 | `test_capture_changes_port` | High | C | easy |
| 54 | `TestSoapySourceSetup::test_initial_ports_on_a_non_stream_all_source` | High | C | trivial |
| 68 | `test_split_rejects_a_tuple_field_shorter_than_port`; `test_pairwise_rejects_different_port_counts` | High | C | easy |
| 70 | `TestSoapyCapture::test_center_frequency_tuple_length_must_match_ports` | High | C | easy |
| 20 | `test_list_port_direct_is_frozen`; `test_var_tuple_list_is_frozen_on_direct_construction`; `test_freeze_depths_count_dict_list_and_tuple_nesting`; `test_freeze_depths_of_real_specs`; `test_direct_struct_with_list_validates_and_replaces` | High | C | trivial |
| 63 | `test_negative_sample_rate_is_rejected_on_convert` | High | C | trivial |
| 64 | `test_descending_range_from_zero_rejected[frame_range]`, `[symbol_range]` | High | C | trivial |
| 22 | `test_analysis_group_accepted` | High | C | moderate |
| 24 | `test_recurses_into_frozendict` | High | C | trivial |
| 49 | `test_port_info_round_trips_as_probed`; `test_arg_info_validate_round_trips` | High | C | trivial |
| 55 | `TestHardwareTimeSync::test_call_returns_the_sync_time` | High | C | trivial |
| 57 | `TestRxStreamRead::test_read_without_an_enable_delay` | High | C | trivial |
| 27 | `TestIstft::test_out_buffer_is_used`; `test_downsample_stft_writes_into_out` | High | C | easy |
| 28 | `test_oafilter_downsample_preserves_level` | High | C | easy |
| 23 | `test_second_directory_with_the_same_import_name_is_imported` | High | C | design |
| 31 | `TestIqToBinPower.test_negative_axis` | Medium | C | trivial |
| 34 | `TestIqToCyclicPower.test_negative_axis` | Medium | C | easy |
| 38 | `test_capture_type_attrs` | High | D | trivial |
| 46 | `TestLookupPowerCorrection::test_out_of_range_message_shows_the_limit_in_mhz` | High | D | trivial |
| 71 | `TestCaptureRemap::test_multi_key_undecodable_key_is_a_validation_error[text]`, `[int]` | High | D | trivial |
| 48 | `TestYFactorSinkFlush::test_saved_attrs_record_the_calibration_fields` | High | D | trivial |
| 41 | `test_message_names_the_field` | High | D | trivial |
| 56 | `TestHardwareTimeSync::test_unsupported_source_names_itself` | High | D | trivial |
| 58 | `TestRxStreamClose::test_close_is_silent` | High | D | trivial |
| 39 | `test_calibration_capture_class_has_a_descriptive_name` | High | D | trivial |
| 50 | `test_air7201b_init_like_matches_its_source_spec` | High | D | trivial |
| 69 | `test_capture_type_of_an_unbound_sweep` | High | D | easy |
| 91 | `test_unknown_keyword_through_unpack_is_flagged` | High | tool (none on striqt) | `ty` upgrade |

## Tier B: reachable from plausible YAML or a spec default

### 43. `_get_resample_overlap` swaps the `ceildiv` arguments

`tests/sensor/test_sensor_compute_corrections.py::test_finite_bandwidth_with_a_small_resampler_fft[*]`

- **Mechanism.** `corrections.py:441` computes
  `ceildiv(design['nfft'], filter_pad)`; the block size is meant to be the
  smallest multiple of `nfft` that covers `filter_pad`
  (`ceildiv(filter_pad, nfft)`). When the resampler FFT is small (a sample
  rate that is an integer divisor of the MCR designs `nfft = 258`; 10 MS/s
  gives 550) the block is a single FFT and either the pad assertion at
  `corrections.py:162` or the parity assertion at `corrections.py:454` fails.
  Large FFTs (15.36 MS/s from 125 MHz designs 6250) happen to satisfy both.
- **Why the test is right.** `get_correction_overlaps` documents that it
  returns the extra samples the resampler and filter consume; an assertion
  is not a valid answer for a valid capture.
- **Impact.** Any capture with a finite `analysis_bandwidth`,
  `host_resample: true` (the default) and a sample rate that divides the
  MCR. The in-tree sweeps use 107.52 MS/s and 15.36 MS/s and are unaffected;
  `air7101b.yaml` too. Reproduced on numpy; the cupy path shares the code.
- **Fix.** Swap the arguments. Whether the pad assertions then hold for
  every capture still needs checking (one extra block of 2064 samples is
  not always more than `filter_pad = 2001` after halving). Easy.

### 77. `_get_resample_overlap` produces odd overlaps for most host-resampled captures

`tests/sensor/test_sensor_compute_corrections.py::test_overlaps_are_even`,
`::test_impulse_acquisition[odd_overlap]`

- **Mechanism.** `corrections.py:449-456` builds `pad_end = pad_blocks*block_size
  - analysis_size` with `analysis_size = round(duration*fs_sdr)`, asserts
  `pad_end % 2 == 0` (which fails whenever `analysis_size` is odd) and returns
  `pad_end // 2` for each side, which is odd whenever `pad_end` is 2 mod 4.
  `controller.py:455` `read_iq` then rejects odd overlaps as "not even". Over
  sample rates of 1 to 20 MS/s, 100 to 40000 output samples and finite or
  infinite bandwidth, roughly 49% of pairs abort in the assertion and a further
  24% acquire an odd overlap; about 12% of the domain acquires (the harness
  presets in `tests/sensor/synthetic_sources.py` were chosen from that 12%).
  `6.144e6 / 2 ms / bw = inf` is the pinned example (3125 per side).
- **Why the test is right.** `correct_iq` handles an odd lead pad correctly on a
  hand-built `AcquiredIQ` (`test_correct_iq_trims_an_odd_lead_pad` passes), so
  the restriction is only the controller's even-overlap contract disagreeing
  with the pad splitter. Any `sample_rate`/`duration` a YAML may set with the
  default `host_resample: true` should acquire.
- **Impact.** B. In-tree sweeps use sizes that happen to work; the downstream
  project's 107.52 and 53.76 MS/s captures at 10 ms multiples do too.
- **Fix.** Round each side up to even and grow the block accordingly, or drop
  the even requirement in `read_iq` (whose reason is the int16 stride). Small;
  `_get_resample_overlap`, `read_iq` and `get_read_count` must agree afterwards.

### 78. `lo_shift` is never undone on the default resample path

`tests/sensor/test_sensor_compute_corrections.py::test_lo_shift_is_removed_from_the_output[left]`,
`[right]`

- **Mechanism.** `design_cola_resampler` moves the tuned LO by
  `resampler['lo_offset'] = ±(bw/2 + bw_lo/2)` so LO leakage falls outside the
  analysis band, and `SingleToneSource` synthesizes the signal at
  `frequency_offset + lo_offset` exactly as an SDR tuned to `fc - lo_offset`
  would deliver it. `corrections.py:337` `_resample` calls
  `sw.resample(x, ny, scale=...)` without `shift=`, and `_scale_only` never
  shifts, so only the experimental `_oaresample` path (which passes
  `frequency_shift=lo_offset`) recentres the band. Measured: a -1 MHz tone comes
  out at +0.661 MHz for `lo_shift: right` at 7.68 MS/s / 3.072 MHz bandwidth.
- **Why the test is right.** `test_lo_shift_design_moves_the_lo_out_of_band`
  (passing) shows the design does displace the LO by `lo_offset`, so the
  correction must undo it or the whole capture is off-center by
  `bw/2 + 125 kHz`.
- **Impact.** B. `lo_shift` is a `SensorCapture` field and a loop field in the
  in-tree and downstream calibration YAMLs, but every production value is
  `none`.
- **Fix.** Pass `shift=round(lo_offset * N_in / fs_sdr)` to `sw.resample`. The
  open question is bin alignment: `lo_offset * N_in / fs_sdr` is not an integer
  for the designs tried (4318.6 bins here), so either `design_cola_resampler`
  must place `lo_offset` on the padded record's bin grid or the shift must be a
  time-domain mixer. Design decision.

### 79. `reuse_iq` under-acquires the filter overlap of a wider analysis bandwidth

`tests/sensor/test_sensor_controller.py::test_reused_iq_can_be_corrected_for_a_wider_analysis_filter`

- **Mechanism.** `buffers.is_reusable` (`buffers.py:326-345`) deliberately
  ignores `analysis_bandwidth`, `sample_rate` and `host_resample`, so a
  `reuse_iq: true` controller hands the first capture's `pre_align` to a second
  capture with a finite `analysis_bandwidth`. `get_correction_overlaps` sized
  that first acquisition for the first capture's filter (512 per side for the
  unfiltered `SCALE_ONLY` preset, 12800 once a 5 MHz filter is requested), so
  `correct_iq` runs out of samples and fails `assert x_pre_align.shape[axis] ==
  size_out` at `corrections.py:116`.
- **Why the test is right.** Sharing one acquisition across analysis settings is
  the documented purpose of `SweepOptions.reuse_iq`, and the warmup sweep sets
  it. The corrected output must still have `duration*sample_rate` samples.
- **Impact.** B. Any `reuse_iq: true` sweep whose reusable captures differ in
  `analysis_bandwidth` crashes at the second capture.
- **Fix.** Either acquire with the maximum overlaps over the group of reusable
  captures (needs the sweep's capture list at acquire time) or make
  `is_reusable` compare `analysis_bandwidth` (one line, but narrows reuse to
  less than the docstring promises). Design decision; medium.

### 90. The resampler overlap measures the FIR pad at the wrong sample rate

`tests/sensor/test_sensor_compute_corrections.py::test_upsampled_overlap_covers_the_output_rate_filter_pad`

- **Mechanism.** `sensor/lib/compute/corrections.py:168`
  `_get_resampler_overlaps` requires its FFT pad to exceed
  `_get_filter_overlap(capture)` = `FILTER_SIZE//2 + 1`, a count of **fs_sdr**
  samples, but `correct_iq` runs the 4001-tap FIR *after* resampling
  (`corrections.py:93-100`), so the transient spans `FILTER_SIZE//2` samples of
  `capture.sample_rate`. Whenever the design upsamples (`fs_sdr > sample_rate`)
  the lead and tail overlaps are short by `sample_rate / fs_sdr`. Worked example,
  reproduced directly: `sample_rate` 12.065 MS/s over 24977 samples with
  `analysis_bandwidth` 6.0325 MHz gives `fs_sdr` 12.5 MS/s and overlaps
  (2061, 2061), which buy only 1989.3 of the 2000 output samples needed, so the
  first and last ~11 samples of the trimmed output keep FIR edge transient.
- **Why the test is right.** The overlaps exist to absorb the filter transient
  before the trim; `get_correction_overlaps`' own docstring describes the pad as
  covering the analysis filter.
- **How it was found.** By the hypothesis test
  `test_overlaps_cover_the_filter_and_trigger_pads`, which asserts the same
  inequality over the drawn capture domain. That test now `assume`s
  `fs_sdr <= sample_rate` so it does not fail probabilistically; removing the
  `assume` is how to confirm a fix over the whole domain. Hypothesis keys its
  example database on the test's source, so editing that test discards the stored
  counterexample and a green run does not clear this item.
- **Impact.** B. Reachable from any YAML whose `sample_rate` is not an exact
  submultiple of the master clock rate, together with a finite
  `analysis_bandwidth`. None of the four synthetic presets upsamples, so the
  suite's own captures are unaffected.
- **Fix.** Scale the required pad to the output rate (compare against
  `_get_filter_overlap(capture) * fs_sdr / capture.sample_rate`, or count the pad
  in output samples throughout). Small, but it changes the acquired sample count
  of every upsampling capture, so `DOCUMENTED_OVERLAPS` and any downstream
  expectation of a read size move with it.

### 66. `_freq_band_edges` drops the last in-band bin for an off-grid cutoff

`tests/waveform/test_fourier.py::TestStftFrequencyEditing::test_zero_stft_by_freq_zeroes_outside_passband[off_grid]`,
`test_downsample_stft_passband_zeroing[off_grid]`

- **Mechanism.** `fourier.py:446` takes the index of the last bin
  `<= cutoff_hi` as the exclusive stop of the passband, so a cutoff between
  bins zeroes a bin that lies inside the passband.
- **Why the test is right.** The on-grid cases pass, and the half-open
  `[lo, hi)` contract in CLAUDE.md is only true on-grid. The caller
  `analysis/lib/source.py:83` passes spec-derived passbands that need not
  fall on the grid.
- **Fix.** Use `searchsorted(freqs, cutoff_hi)` (or `+ 1`) for the stop.
  Easy.

### 6. `Phy3GPP` cyclic-prefix layout is wrong at 15 kHz and 60 kHz

`tests/waveform/test_waveform_ofdm.py::TestPhy3GPP::test_subframe_cp_layout[15kHz]`,
`[60kHz]` (added 2026-09-16: tiles `cp_sizes` over a 1 ms subframe and
asserts the TS 38.211 5.3.1 rule directly);
`tests/waveform/test_waveform_ofdm.py::TestPhy3GPP::test_slots_tile_the_frame[15kHz]`,
`[60kHz]`, and `test_lte_and_5g_agree_at_15khz[*]` (5 sample rates); both
tables draw their marks from the module's `scs_cp_layout` builder

- **Mechanism.** `ofdm.py:1058` adds the extra `16κ` CP samples once per 14
  symbols. TS 38.211 §5.3.1 adds them at `l = 0` and `l = 7·2^μ` of each 1 ms
  subframe: twice per slot at 15 kHz, once per slot at 30 kHz, once every
  other slot at 60 kHz. `contiguous_size` (`ofdm.py:880`) and the slot grid
  `contiguous_size * slot` (`ofdm.py:954`) inherit the error, while
  `frame_size` (`ofdm.py:943`) is computed correctly from the sample rate. The
  library's own `LTE_MIN_CP_SIZES` table (`ofdm.py:1016`) has the two long CPs
  per 14 symbols that the 5G path lacks.
- **Why the test is right.** The class docstring says the 5G model equals LTE
  at 15 kHz and cites TS 38.211; the LTE table in the same file disagrees with
  the 5G computation at 15 kHz.
- **Impact.** At 15.36 MS/s the slot grid is short by 8 samples per slot at
  15 kHz (normal CP is 72 samples, so slot 9 of 10 is fully misaligned) and
  long by 4 per slot at 60 kHz (CP is 18, so slots 5 to 39 are misaligned). The
  error resets each frame, so the cyclic autocorrelation is degraded rather
  than destroyed. `Phy3GPP` is built only by `cellular_cyclic_autocorrelation`,
  whose `subcarrier_spacings` default is `(15e3, 30e3, 60e3)`
  (`analysis/specs/structs.py:280`), and `doc/reference-sweep.yaml` lists all
  three. Every in-tree sweep that enables the measurement overrides to 30 kHz
  only, and the site sweeps do not enable it, so the exposure is the default.
- **Fix.** Make `cp_sizes` subframe-aware; the shape of `cp_sizes` and
  `contiguous_size` change, and `_get_3gpp_index_cyclic_prefix` must follow.
  Fixing it will also make `correlate_sync_sequence` raise "same excess CP"
  for Case A at 15 kHz, because symbols 2 and 8 then straddle `l = 7`;
  `pss_params` returns `cp_offsets=[4]*14` where the standard gives
  `[4]*7 + [8]*7`, so that path needs a design decision too. Moderate. A
  related inconsistency not covered by an xfail: `generation='4G'` at 30 or
  60 kHz scales the LTE table and gives 40/36 instead of 44/36.

### 8. Required `defaults` remaps are silently omitted on a miss

`tests/sensor/test_sensor_specs_sequencing.py::test_adjust_captures_missing_required_default_lookup_raises`

- **Mechanism.** `helpers.py:286`: the `elif required: return msgspec.UNSET`
  and `else: raise KeyError` arms are inverted. A `defaults` remap is its own
  `field_default`, so `required=True` takes the silent arm. The per-port
  lookup path routes through the same two arms, so it has the same gap.
- **Why the test is right.** `CaptureRemap.required` defaults to `True`, the
  `KeyError` text already exists, and the sibling
  `test_adjust_captures_missing_required_source_lookup_raises` shows the
  source-block case raising as intended.
- **Impact.** `sites/global.yaml` `antenna_index` (required, no default) is
  this shape: a port outside `{0, 1}` produces a capture with
  `antenna_index=None` instead of an error.
- **Fix.** Swap the two arms. Trivial.

### 9. A remap keyed on an unknown field is pruned, not rejected

`tests/sensor/test_sensor_specs_sequencing.py::test_remap_keyed_on_an_unknown_field_is_rejected`

- **Mechanism.** `helpers.py:562` and `:575`: `continue` immediately precedes
  the `raise msgspec.ValidationError(...)` in both the scalar and tuple-key
  branches, so the raise is unreachable.
- **Why the test is right.** The dead error messages show validation was
  intended; the sweep structs are `forbid_unknown_fields=True` everywhere else.
- **Impact.** A typo such as `centre_frequency` deletes the whole remap and the
  coordinate goes missing from the output with no diagnostic.
- **Fix.** Delete the two `continue` lines. Trivial.

### 13. `binned_mean(fft=True)` raises for some odd bin counts

`tests/waveform/test_waveform_arrays.py::TestBinnedMean::test_fft_bins_odd_count_even_length[8-3]`, `[14-3]`, `[1024-5]`

- **Mechanism.** `arrays.py:139-147`: when `(n//2 - count//2) % count == 0`
  with odd `count` and even `n`, `count*block_count` is `n+1`, `start` is 0,
  `stop` is `n+1`, the slice is skipped, and `axis_to_blocks` rejects the
  unaligned length.
- **Why the test is right.** The docstring promises fft-aligned bins with
  truncation of incomplete ones; raising is not one of the documented
  outcomes.
- **Impact.** Callers are the spectrogram frequency-bin averaging in
  `measurements/spectrum.py` (`_cached_spectrogram` and `spectrogram_freqs`; count is
  `integration_bandwidth / frequency_resolution`, axis 2, `fft=True`) and the
  SSB spectrogram (count 2). All shipped YAMLs use a ratio of 24. A user who
  picks an odd ratio such as 45 kHz over 15 kHz gets a `ValueError` for about
  a third of realistic `nfft` values.
- **Fix.** Clamp `block_count` so the block span never exceeds `n`. Easy.

### 44. A single-frequency calibration cannot be looked up

`tests/sensor/test_sensor_calibration.py::TestLookupPowerCorrection::test_recovers_the_receiver_gain[single_frequency_calibration]`

- **Mechanism.** `calibration.py:572` calls `squeeze(drop=True)` before
  `dropna('center_frequency')`, so a length-1 `center_frequency` dimension
  is gone by the time it is named, and xarray raises `ValueError`.
- **Why the test is right.** The exact-match branch below it
  (`sel(center_frequency=...)`) would otherwise succeed for that frequency.
- **Impact.** A calibration sweep with one `center_frequency` value.
- **Fix.** `dropna` first, or squeeze every dimension except
  `center_frequency`. Trivial.

### 52. `SoapySource.close` never closes the device

`tests/sensor/test_sensor_sources_soapy.py::TestSoapySourceClose::test_close_releases_the_stream_and_device`

- **Mechanism.** `soapy.py:648` and `:649` read `self._device` and
  `self._rx_stream` with `getattr(..., None)`; the attributes are `device` and
  `rx_stream`, so both are `None` and the `try` body does nothing.
- **Impact.** The stream is never deactivated or closed and `Device.close`
  is never called. On the Jetson the process exit releases the radio, so a
  single sweep works; opening a second controller in one process, or the
  `Controller.close`/reopen path, relies on the driver tolerating it.
- **Fix.** Rename the two lookups. Trivial.

### 14. A leading `Repeat` breaks calibration loop insertion

`tests/sensor/test_sensor_calibration.py::TestNoiseDiodeToggle::test_leading_repeat_is_accepted[*]`

- **Mechanism.** `calibration.py:369-393` `_ensure_loop_at_position` only
  allows for a leading `port` loop; `_validate_loops`
  (`sensor/specs/sweep.py`) requires a `Repeat` to be first. The toggle
  is inserted ahead of the `Repeat`, or the helper raises when the `Repeat` is
  already first.
- **Why the test is right.** The two constraints are stated in code and are
  mutually exclusive as written.
- **Impact.** No in-tree calibration YAML uses `kind: repeat`; any that adds
  one fails to decode.
- **Fix.** Offset the insert index by one when the first loop has no field.
  Easy.

### 16. Remaps keyed on a fixed numeric alias reject numeric keys

`tests/sensor/test_sensor_specs_sequencing.py::test_remap_keyed_on_a_fixed_numeric_alias_accepts_numeric_keys`

- **Mechanism.** `helpers.py:832` and `:872` set `lookup_types[field] = str`
  after a fixed value or a remap is processed, overriding the real field type
  that `lookup_types` was initialized from. Later keys are converted with
  `msgspec.convert(k, str, strict=False)`, which rejects an int.
- **Why the test is right.** `SwitchInput` is `Annotated[int, ...]` in the
  extensions module; forcing `str` contradicts the schema.
- **Impact.** A site that pins `switch_input: 1` and keys `antenna_index` on
  it fails at decode with an "Expected lookup keys matching the type of capture
  field" error. The in-tree files
  avoid it because `switch_input` is looped, not fixed.
- **Fix.** Delete both assignments. Trivial.

### 17. Nested `!include` is globbed against the wrong directory

`tests/analysis/test_analysis_io.py::test_nested_include_resolves_relative_to_the_including_file`

- **Mechanism.** `analysis/lib/io.py:852` globs relative to the top-level
  spec's directory while `io.py:840` opens relative to the innermost fragment;
  the glob results are also relativised to the top-level directory and then
  re-joined onto the fragment directory.
- **Why the test is right.** The `nested_paths` stack exists only to resolve
  nested includes relative to the including file.
- **Impact.** No in-tree fragment includes another file. A downstream fragment
  in a subdirectory that includes a sibling gets `FileNotFoundError`.
- **Fix.** Pass the fragment directory as `root_dir` and stop relativising.
  Moderate, shared with item 18.

### 18. Absolute `!include` outside the spec tree fails

`tests/analysis/test_analysis_io.py::test_absolute_include_outside_root`

- **Mechanism.** `io.py:805` calls `relative_to(root_dir)` on the absolute
  match, which raises `ValueError` when the file is outside the top-level
  directory.
- **Why the test is right.** `get_include_path` (`io.py:835`) has an explicit
  absolute-path branch, and `test_absolute_include_inside_root` passes.
- **Impact.** `source: !include /etc/striqt/radio.yaml` from a spec elsewhere
  fails with "is not in the subpath of". Not used in-tree.
- **Fix.** Return absolute matches as-is. Easy.

### 62. `SpecBase` freezes `dict[str, Any]` fields to depth 2 only

`tests/analysis/test_analysis_specs_helpers.py::test_three_deep_adjust_analysis_is_hashable`

- **Mechanism.** `SpecBase.__post_init__` (`analysis/specs/structs.py:82-86`)
  freezes `Any`-typed leaves two levels down, so
  `SingleToneCapture(port=0, adjust_analysis={'spectrogram': {'window': ['kaiser', 8]}})`
  keeps the inner list and `validate()` raises "unhashable frozendict entry".
- **Why the test is right.** The YAML path is safe only because
  `_YAMLFrozenLoader` pre-converts sequences to tuples; a spec built in
  Python (the extensions module, notebooks) should behave the same.
- **Fix.** Freeze `Any` leaves to full depth (`max_depth=None`). Small.

### 65. `stft(noverlap=0)` is scaled by `1/nfft` relative to the overlapped path

`tests/waveform/test_fourier.py::TestIstft::test_no_overlap_roundtrip_is_identity`

- **Mechanism.** With `norm=None` the no-overlap path applies `w / nfft`
  uncompensated (`fourier.py:558`), while `_stack_stft_windows`
  (`fourier.py:750`) normalizes the same factor away, so the same segment
  differs by `nfft` between the two supported overlaps and
  `istft(stft(x))` returns `x / nfft` for `noverlap=0`.
- **Why the test is right.** The previous test fitted an arbitrary gain by
  least squares and so could not see this. Either overlap should round-trip.
- **Impact.** Measurements use `norm='power'`, which is consistent across
  overlaps; the raw path is exposed through the public `stft`.
- **Fix.** Decide which scale is the contract and apply it on both paths.

### 67. The CUDA `_corr_at_indices` stops at the first out-of-range index

`tests/waveform/test_waveform_jit.py::TestCorrAtIndicesKernels::test_cuda_unsorted_indices`

- **Mechanism.** `jit/cuda.py:25-26` `break`s at the first out-of-range
  index pair, while `jit/cpu.py:23-28` zero-fills it and continues.
  `ofdm.corr_at_indices` flattens a (symbol, slot, frame, cp) meshgrid
  (`ofdm.py:133`), which is not monotonic, so the kernels disagree whenever a
  valid index follows an invalid one (IQ shorter than the requested frames).
- **Why the test is right.** The two backends must agree (CLAUDE.md,
  lockstep requirement). The test could not be run here (cupy skips); the
  mechanism was checked by replaying both loops in plain Python. Needs a
  Jetson run to confirm the strict marker.
- **Fix.** Replace `break` with `continue` after zero-filling. Trivial.

### 59. `_probe_channel` indexes a sensor `ArgInfo`

`tests/sensor/test_sensor_sources_soapy.py::test_probe_soapy_info_with_a_channel_sensor`

- **Mechanism.** `soapy.py:176` calls `getSensorInfo(*args, key)[0]`;
  `SoapySDR.Device.getSensorInfo` returns one `ArgInfo`, which is not
  indexable, so `TypeError`. The line above (`.name`) and the global-sensor
  path in `probe_soapy_info` (`soapy.py:239`) use it without indexing.
- **Impact.** A device that lists any per-channel sensor cannot be opened
  (`probe_soapy_info` runs from `SoapySource.setup`). The Airstack evidently
  lists none, since sweeps run on it; other SoapySDR drivers (bladeRF, USRP
  via UHD) do expose per-channel sensors. Latent.
- **Fix.** Drop the `[0]`. Trivial.

### 19. `Capture(sample_rate=inf)` raises "math domain error"

`tests/analysis/test_analysis_specs_structs.py::TestCapture::test_infinite_sample_rate_reports_sample_period_message`

- **Mechanism.** `analysis/lib/util.py:50` `math.remainder(inf, 1)` raises
  `ValueError`, which `isroundmod` does not catch; the duplicate in
  `waveform/lib/arrays.py:38` has the same gap. `nan` happens to produce the
  intended message.
- **Why the test is right.** `Capture.__post_init__` intends its own
  sample-period message for any non-integer sample count, and the type aliases
  carry no finiteness constraint.
- **Impact.** `sample_rate: .inf` in YAML surfaces as
  `ValidationError: math domain error`. Not a realistic configuration.
- **Fix.** Return `False` for non-finite ratios in both copies. Trivial.

### 10. Chained remaps resolve only in declaration order

`tests/sensor/test_sensor_specs_sequencing.py::test_chained_remap_declared_before_its_key_resolves`

- **Mechanism.** `helpers.py:298` iterates the merged fields once with
  `ChainMap(ret, capture)`; a remap keyed on an alias declared later sees the
  capture's raw value and misses silently.
- **Why the test is right.** Nothing documents order dependence for
  `adjust_captures`. The legacy `coord_aliases` prose in
  `doc/reference-sweep.yaml:49` did say "previously defined aliases", so
  Medium confidence: this may have been an accepted constraint that was never
  carried into the new format's documentation.
- **Impact.** `sites/global.yaml` and `sites/radio02.yaml` chain
  `antenna_model` on `antenna_name` on `port`, and work because the files
  happen to be ordered. Reordering a YAML block silently drops coordinates.
- **Fix.** Iterate to a fixpoint or topologically sort by `key`, with cycle
  detection. Moderate.

### 45. Lookups require every exact-match field to be a dimension

`tests/sensor/test_sensor_calibration.py::TestLookupPowerCorrection::test_recovers_the_receiver_gain[calibration_without_a_lo_shift_loop]`

- **Mechanism.** `calibration.py:554` selects the exact-match fields with
  `sel`, which needs an index. `lo_shift` is a dimension only when it was
  looped; otherwise it is a scalar (or multi-dimensional) coordinate,
  `sel` raises `KeyError`, and `_describe_missing_data` then iterates the
  0-d coordinate at `calibration.py:521` and raises `TypeError`. xarray
  2026.7 (the `test314` environment) raises `ValueError` from `sel` itself
  instead; the xfail accepts either.
- **Why the test is right.** The scalar coordinate carries exactly the value
  being matched; the lookup has the information it needs. Medium: it is
  possible the design intends every calibration sweep to loop `lo_shift`,
  which `doc/calibration-sweep.yaml:48` does.
- **Impact.** A calibration sweep without a `lo_shift` loop produces a file
  that no measurement can use, with an unrelated error message.
- **Fix.** Compare scalar coordinates with `==` and only `sel` on
  dimensions; make `_describe_missing_data` use `np.atleast_1d`. Easy.

## Tier C: Python API only, dead code, or GPU-latent

### 80. `_oaresample` trims the wrong lead and cannot complete

`tests/sensor/test_sensor_compute_corrections.py::test_oaresample_impulse_lands_on_its_output_sample[resample_filter]`,
`[resample_only]`

- **Mechanism.** `corrections.py:373-378` slices `offset = nfft_out` samples from
  the front of the `sw.oaresample` output instead of the resampled lead overlap
  that `_get_oaresample_overlaps` requested, then asserts the remainder equals
  the capture plus its tail pad. With the requested lead it never does
  (114744 - 32784 = 81960 versus 28392 for `RESAMPLE_ONLY`). The kernel is
  right: the impulse sits at exactly `(lead + n0) * nfft_out / nfft` in the raw
  `oaresample` output.
- **Why the test is right.** Trimming must remove the lead pad the overlap design
  asked for; the module comment already admits a "residual time offset".
- **Impact.** C. Only under `STRIQT_USE_OARESAMPLE=1`, documented experimental.
- **Fix.** Trim `round(lead * nfft_out / nfft)` and size the tail consistently;
  `_get_oaresample_overlaps` and `_oaresample` must be re-derived together.
  Moderate.

### 81. `_get_oaresample_overlaps` returns odd overlaps

`tests/sensor/test_sensor_compute_corrections.py::test_oaresample_acquisition[resample_filter]`,
`[resample_only]`

- **Mechanism.** `corrections.py:401-425` returns `(43750, 9375)` for
  `RESAMPLE_FILTER` and `(89950, 17075)` for `RESAMPLE_ONLY`; `read_iq` rejects
  the odd tail. Same contract as item 77 on the experimental path.
- **Impact.** C (`STRIQT_USE_OARESAMPLE=1` only).
- **Fix.** Round to even once item 80 settles the trimming. Small.

### 82. `ZarrIQSource.get_resampler` lets the design pick a rate the file cannot supply

`tests/sensor/test_sensor_sources_file.py::test_zarr_resampler_pins_file_rate`

- **Mechanism.** `file.py` `ZarrIQSource.get_resampler` calls
  `design_resampler(capture, file_rate)` without `backend_sample_rate=`, so for
  `sample_rate = FS/2` the design chooses `fs_sdr = FS/2`, `nfft = nfft_out =
  258` as if the SDR decimated, while the file only holds `FS`. TDMS and MAT pin
  `backend_sample_rate`.
- **Why the test is right.** A file has one rate; with it pinned the design is
  `fs_sdr = FS`, `nfft = 516`, `nfft_out = 258`.
- **Impact.** C. Any host-resampled `zarr_iq` capture yields wrong timing and
  frequency. The bindings were not instantiable at all before 2026-09-18 (see
  the fixes list), so nothing has used them yet.
- **Fix.** The one-line `design_resampler(capture, fs, backend_sample_rate=fs)`
  fixes the design, but `_get_resample_overlap` (`corrections.py:430`) and
  `buffers.get_read_count` (`buffers.py:136`) derive `nfft`/`fs_sdr` from
  `design_resampler(capture, setup.master_clock_rate)` and would then disagree
  with the backend's design (and hit item 77). `FileCapture` forbids
  `backend_sample_rate`, yet the overlap machinery has no access to the backend.
  Design decision; the xfail is kept whole so that it records that file
  resampling is broken end to end, not only the design.

### 83. `MATSource.get_waveform` ignores `port` for multi-row files

`tests/sensor/test_sensor_sources_file.py::test_mat_two_port`

- **Mechanism.** `MATSource.get_waveform` returns `ret[:, :file_count]`, every
  row regardless of `port`; `VirtualSource.read` then cannot assign the
  `(rows, count)` result into one port buffer. Zarr uses row = port and the MAT
  stream itself reports `port = list(range(rows))`.
- **Impact.** C. Multi-row `.mat` files work single-port only.
- **Fix.** `ret[[port], :file_count]`, one line, once it is decided that `port`
  indexes rows (a `port: 3` label on a one-row file, which works today, would
  then fail). Easy.

### 84. `MATLegacyFileStream.read` wraps past the end of the file regardless of `loop`

`tests/sensor/test_sensor_sources_file.py::test_mat_request_past_end`

- **Mechanism.** `src/striqt/analysis/lib/io.py` `MATLegacyFileStream.read`
  (about lines 401-447) rebuilds `all_refs = list(self._refs)` on every call,
  and `_refs` is the whole matrix, so a request past the end (up to one extra
  file length) silently appends the file from index 0 with `loop=False`; it only
  raises "too few samples" when one request exceeds the leftover plus a full
  file. `test_mat_loop_repeats_file` passes for the same mechanism because it
  asserts the intended `loop=True` behavior.
- **Why the test is right.** Zarr and TDMS raise `ValueError` past the end.
- **Impact.** C. A capture longer than the file returns looped data with no
  error.
- **Fix.** Track the consumed position in the stream instead of re-listing the
  refs; touches `seek`/`_leftover` in `MATLegacyFileStream` and
  `MATNewFileStream`. Easy to medium.

### 85. `open_resources(test_only=True)` returns resources `iterate_sweep` cannot run

`tests/sensor/test_sensor_resources.py::test_test_only_resources_run_a_sweep`

- **Mechanism.** `resources.py:146-151` skips the peripherals when `test_only`,
  and `execute.py:235` `_acquire_both` indexes `res['peripherals']`
  unconditionally, raising `KeyError`.
- **Why the test is right.** `Resources` declares `peripherals` as
  `NotRequired` and `open_resources` documents its result as ready to run the
  sweep; the test harness (`synthetic_sources.run_in_memory`) has to pass
  `peripherals=` itself.
- **Impact.** C (`test_only` is a Python-API flag).
- **Fix.** Open `NoPeripherals` when skipping, or guard with
  `res.get('peripherals')` in `_acquire_both`. Easy; needs a choice.

### 86. `Source.signal_trigger` cannot decode the analysis-group form

`tests/sensor/test_sensor_compute_analyze.py::test_get_trigger_from_an_analysis_group_trigger`

- **Mechanism.** `sensor/specs/structs.py:105` annotates `signal_trigger:
  Union[str, AnalysisGroup, None]` with the fieldless `AnalysisGroup` base, so
  `validate()` and YAML decoding reject any measurement key. The
  `AnalysisGroup` branches of `_get_signal_trigger_name` and
  `get_trigger_from_spec` (`analyze.py:50-75`) are unreachable through a
  validated spec; `BundledTriggers` (`structs.py:307`) exists and is unused.
- **Impact.** C. Only the string form works, which is what every YAML uses.
- **Fix.** Annotate with `BundledTriggers`. Easy, but a schema change.

### 87. `from_delayed` rejects the `AcquisitionInfo` default `sweep_index=None`

`tests/sensor/test_sensor_compute_datasets.py::test_from_delayed_accepts_the_acquisition_info_defaults`

- **Mechanism.** `datasets.py:194` writes `AcquisitionInfo.sweep_index` (default
  `None`, typed `Union[int, None]`) into the `int` template that
  `_coords_template` derives, raising `TypeError`. Only `iterate_sweep`'s
  `_AcquisitionIndexer` ever sets it, so `compute.analyze` on a bare
  `Controller.acquire()` result cannot be packaged.
- **Impact.** C (Python API).
- **Fix.** Default `sweep_index: int = 0`, or coerce `None` in
  `build_capture_coords`. One line once decided.

### 88. An analysis loop over an un-inferable field breaks `from_delayed`

`tests/sensor/test_sensor_compute_datasets.py::test_build_capture_coords_adds_a_window_loop_coordinate`

- **Mechanism.** `analysis/lib/register.py:401` stores
  `parameter_fields[name] = None` when `infer_coord_info` rejects the field
  type (`window`, `time_statistic`, `power_detectors`, ...); `datasets.py:327`
  `_coords_template` then calls `make_var(None)` for an analysis loop over that
  field and raises `AttributeError`.
- **Why the test is right.** An analysis loop over `window` is a valid spec
  (`sweep_strategies.loop_sets` builds them), yet the first `from_delayed` of
  such a sweep dies.
- **Impact.** C, B if the downstream project loops over one of these fields.
- **Fix.** Give the registry an object-dtype fallback for un-inferable fields.
  Moderate; analysis package.

### 53. Port changes do not rebuild the stream

`tests/sensor/test_sensor_sources_soapy.py::test_capture_changes_port`

- **Mechanism.** `capture_changes_port` (`soapy.py:497`) returns `==` rather
  than `!=`, and `SoapySource.arm` never assigns `self._capture`, so
  `soapy.py:666` calls `RxStream.setup` on every arm and relies on it to
  decide whether the port set changed.
- **Partially fixed 2026-09-24.** `RxStream.setup` had an early return that
  fired whenever a stream existed, which hid a `self.close()` call missing its
  `device` argument on the re-open path; both are fixed and
  `TestRxStreamSetup::test_setup_with_new_ports_closes_and_reopens_the_stream`
  pins the close-and-reopen sequence. That made
  `TestSoapySourceArm::test_port_change_rebuilds_the_stream` pass, so its
  marker was removed (`xfailed` 91 to 90): a port change now rebuilds the
  stream because `arm` always reaches `setup`, not because
  `capture_changes_port` is right.
- **Impact.** Every in-tree Soapy spec sets `stream_all_rx_ports = True`, so
  the stream always carries every port and the port set never changes. The
  inverted predicate is now harmless in `arm` but wrong for any other caller.
- **Fix.** `!=` and record `_capture`. Easy.

### 54. `SoapySource.setup(rx_ports=...)` fails without `stream_all_rx_ports`

`tests/sensor/test_sensor_sources_soapy.py::TestSoapySourceSetup::test_initial_ports_on_a_non_stream_all_source`

- **Mechanism.** `soapy.py:634` calls `self.rx_stream.setup(self.device)`
  without the ports it just passed to `RxStream(ports=rx_ports)`, and
  `RxStream.setup` only consults its own `self.ports` in the early-return
  check, so `soapy.py:338` raises.
- **Impact.** As item 53: masked by `stream_all_rx_ports` on every in-tree
  spec. `Controller.from_sweep_spec` always passes `init_rx_ports`.
- **Fix.** `self.rx_stream.setup(self.device, rx_ports)`. Trivial.

### 68. Ragged per-port tuples are truncated or copied instead of rejected

`tests/sensor/test_sensor_specs_captures.py::test_split_rejects_a_tuple_field_shorter_than_port`,
`::test_pairwise_rejects_different_port_counts`

- **Mechanism.** `split_capture_ports` zips each tuple field against `port`,
  so a tuple shorter than `port` is copied whole onto the extra ports;
  `pairwise_by_port` zips two split lists and truncates to the shorter.
- **Why the test is right.** Nothing in the specs or docs sanctions ragged
  tuples; both outcomes silently mis-assign per-port values.
- **Impact.** The capture validators reject mismatched `gain` tuples, so only
  fields without a length check (item 70) and hand-built captures reach it.
- **Fix.** Raise `ValueError` on a length mismatch in both helpers. Easy.

### 70. `SoapyCapture` does not check the `center_frequency` tuple length

`tests/sensor/test_sensor_specs_structs.py::TestSoapyCapture::test_center_frequency_tuple_length_must_match_ports`

- **Mechanism.** Only `gain` is validated against the port count; a
  `center_frequency` tuple of the wrong length is accepted and
  `split_capture_ports` then drops the extras (item 68).
- **Fix.** Extend `_validate_multichannel` to every per-port tuple field.
  Easy.

### 20. Variable-length tuple fields are not frozen on direct construction

`tests/sensor/test_sensor_specs_structs.py::TestSoapyCapture::test_list_port_direct_is_frozen`;
`tests/analysis/test_analysis_specs_structs.py::TestSpecBaseFreezing::test_var_tuple_list_is_frozen_on_direct_construction`;
added 2026-09-16, formerly passing tests that pinned the depth-0 result:
`tests/analysis/test_analysis_specs_helpers.py::test_freeze_depths_count_dict_list_and_tuple_nesting`,
`::test_freeze_depths_of_real_specs`, and
`tests/analysis/test_analysis_specs_structs.py::TestSpecBaseFreezing::test_direct_struct_with_list_validates_and_replaces`

- **Mechanism.** `_inspect_container_depth` (`analysis/specs/helpers.py:341`)
  handles `TupleType` but not `VarTupleType`, so `tuple[T, ...]` fields get
  depth 0 and `SpecBase.__post_init__` never freezes them. Affected fields
  include `port`, `center_frequency`, `gain`, `subcarrier_spacings`,
  `power_detectors`, `captures`, `loops`, `CaptureRemap.key`, and the
  extensions module's `antenna_index`.
- **Why the test is right.** The sibling fixed-tuple test passes, and
  `_validate_multichannel` is `lru_cache`d, so an unfrozen list makes
  `SoapyCapture(port=[0, 1])` raise "unhashable type".
- **Impact.** YAML and `from_dict` go through `msgspec.convert`, which yields
  tuples, so sweeps are unaffected. Only `Controller.arm(port=[...])` and
  hand-built specs hit it.
- **Fix.** Add the `VarTupleType` branch. Trivial, but it also changes which
  fields `to_dict(unfreeze=True)` emits as lists in zarr attrs.

### 63. `types.SampleRate` accepts a negative value

`tests/analysis/test_analysis_specs_structs.py::TestCapture::test_negative_sample_rate_is_rejected_on_convert`

- **Mechanism.** `analysis/specs/types.py:69` carries no `gt=0` bound, unlike
  the sensor's `BackendSampleRate`, so `-1e6` converts.
- **Why the test is right.** Every consumer does `round(duration * sample_rate)`
  and a negative rate yields a negative sample count. Asserted on the convert
  path only, because `Meta` bounds are never checked on direct construction.
- **Impact.** Sensor sweeps decode through `BackendSampleRate`, which has the
  bound; only analysis-level specs built directly are open.
- **Fix.** Add `gt=0`. Trivial.

### 64. `_validate_range` skips the order check when the start is 0

`tests/analysis/test_analysis_specs_structs.py::TestCellularCyclicAutocorrelator::test_descending_range_from_zero_rejected[frame_range]`, `[symbol_range]`

- **Mechanism.** `analysis/specs/structs.py:282` returns before the
  `end >= start` check when `start` is 0, so `(0, -5)` becomes `range(0, -5)`,
  an empty selection, and the measurement silently produces nothing.
- **Why the test is right.** The existing message "Expected an end value at or
  after `start`" is the stated rule for every other start.
- **Fix.** Drop the early return. Trivial.

### 22. `SoapySource.signal_trigger` rejects the `AnalysisGroup` form

`tests/sensor/test_sensor_specs_structs.py::TestSignalTrigger::test_analysis_group_accepted`

- **Mechanism.** `sensor/specs/structs.py:148` checks membership in the
  trigger registry, which fails for a struct. `FunctionSource` accepts it, and
  `compute/analyze.py:66` implements the group form.
- **Why the test is right.** The annotation and `analyze.py` both intend it.
  However the annotation is the bare `AnalysisGroup`, which has no fields, so
  YAML cannot express it anyway; `BundledTriggers` (`structs.py:307`), the
  type that should be there, cannot be introspected on py3.9 and is
  unreferenced.
- **Impact.** Dead path in production; every YAML uses the string form.
- **Fix.** Relaxing the check is trivial; making it usable from YAML needs
  `AlignmentSourceRegistry.to_spec()` fixed and used. Moderate.

### 24. `freeze` does not recurse into an existing `frozendict`

`tests/analysis/test_analysis_specs_helpers.py::TestFreeze::test_recurses_into_frozendict`

- **Mechanism.** `helpers.py:265` branches on `dict`, and `frozendict` is a
  `Mapping`, so it is returned as-is with any lists inside it; `unfreeze`
  handles both.
- **Why the test is right.** Asymmetry with `unfreeze` and the hashability
  guarantee of `SpecBase`.
- **Impact.** YAML never produces the pattern; only hand-built frozendicts
  with lists inside reach it.
- **Fix.** `isinstance(obj, (dict, frozendict))` in `freeze` and in
  `SpecBase.__post_init__`. Trivial.



### 49. Capability struct annotations disagree with the probed values

`tests/sensor/test_sensor_sources_soapy.py::test_port_info_round_trips_as_probed`,
`::test_arg_info_validate_round_trips`

- **Mechanism.** `soapy.py:104` and `:106` annotate `full_freq_range` and
  `backend_sample_rate_range` as `_SoapyRange`, but `_probe_channel` fills
  them with `from_soapy_tuple` (the `# type: ignore` comments mark the
  spot), so `validate()` raises. `soapy.py:56` annotates
  `_SoapyArgInfo.range` as a bare `tuple`, so `validate()` decodes the
  `_SoapyRange` entries as dicts and the struct is not equal to itself
  after a round trip.
- **Impact.** `SoapyInfo` is built once per source and read as attributes;
  nothing validates or serializes it today.
- **Fix.** `tuple[_SoapyRange, ...]` in both places. Trivial.

### 55. `HardwareTimeSync.__call__` returns `None`

`tests/sensor/test_sensor_sources_soapy.py::TestHardwareTimeSync::test_call_returns_the_sync_time`

- **Mechanism.** `soapy.py:509` is annotated `-> int | None` and both
  branches assign `self.last_sync_time` without returning it.
- **Impact.** The two call sites (`SoapySource.setup`, `trigger`) ignore the
  return value.
- **Fix.** `return self.last_sync_time`. Trivial.

### 57. `RxStream.read` cannot handle `rx_enable_delay = None`

`tests/sensor/test_sensor_sources_soapy.py::TestRxStreamRead::test_read_without_an_enable_delay`

- **Mechanism.** `RxStream.enable` (`soapy.py:411`) treats `None` as "no
  delayed activation", but `soapy.py:437` adds the delay to the timeout
  unguarded and raises `TypeError`.
- **Impact.** Every in-tree spec sets a float. A subclass that opts out of
  the delayed start cannot read.
- **Fix.** `(self.source_spec.rx_enable_delay or 0)`. Trivial.

### 27. `istft` and `downsample_stft` never write into `out`

`tests/waveform/test_fourier.py::TestIstft::test_out_buffer_is_used`;
`TestStftFrequencyEditing::test_downsample_stft_writes_into_out`

- **Mechanism.** `_truncated_buffer` (`fourier.py:185`) uses
  `ndarray.flatten()`, which always copies.
- **Why the test is right.** `_unstack_stft_windows` documents `out` as the
  array that receives the result, and `downsample_stft` has a fast path that
  returns a view of `y` when no zeroing is needed.
- **Impact.** Nothing on the production path passes `out=` to `istft`;
  `oafilter` passes `out=y` to `downsample_stft` only when `nfft_out != nfft`,
  which no production caller does. The `out=` API is inert, not harmful.
- **Fix.** `reshape(-1)` on a contiguous buffer, with a fallback. Easy.

### 28. `oafilter` with `nfft_out != nfft` scales the level by `nfft / nfft_out`

`tests/waveform/test_fourier.py::TestOverlapAddFilters::test_oafilter_downsample_preserves_level`

- **Mechanism.** The STFT is normalized for `nfft`-point frames and the
  inverse reconstructs `nfft_out`-point frames (`fourier.py:882-923`).
  Measured gain is 1, 2, 4 for `nfft_out` of 256, 128, 64, so the xfail
  reason's "doubles" is the special case of a 2:1 ratio.
- **Why the test is right.** `oaresample` (`fourier.py:1540`) rescales by the
  size ratio explicitly, showing the intended convention.
- **Impact.** No caller passes `nfft_out != nfft`; `oaresample` is a separate
  implementation and `correct_iq` uses `resample`. Unreachable.
- **Fix.** Multiply by `nfft_out / nfft`. Easy.

### 23. A second extensions directory with the same module name is never imported

`tests/sensor/test_sensor_io.py::test_second_directory_with_the_same_import_name_is_imported`

- **Mechanism.** `sensor/lib/io.py:248` `importlib.import_module` returns the
  cached module; the "did not bind a sensor" warning then fires.
- **Why the test is right.** The behavior is undocumented either way; the
  warning encodes the current heuristic. A real reload would also collide with
  `bind_sensor`'s "already registered" check.
- **Impact.** The CLIs read one spec per process. Notebook and test sessions
  that load two sites are affected.
- **Fix.** Needs a registry replace-or-namespace policy first. Design decision.


### 31. `iq_to_bin_power` with a negative axis reduces the wrong axis

`tests/waveform/test_power_analysis.py::TestIqToBinPower::test_negative_axis`

- **Mechanism.** `power_analysis.py:414` reduces `axis + 1` without
  normalizing, so `axis=-1` reduces axis 0.
- **Why the test is right.** Numpy convention; not stated in the docstring.
  Medium.
- **Impact.** Callers pass `axis=1`. Unreachable.
- **Fix.** Normalise the axis at entry. Trivial.

### 34. `iq_to_cyclic_power` normalizes a negative axis too late

`tests/waveform/test_power_analysis.py::TestIqToCyclicPower::test_negative_axis`

- **Mechanism.** `power_analysis.py:484` normalizes after the
  `iq_to_bin_power` calls and the `power_shape[1]` check. The observed failure
  is item 31 propagating (the xfail reason now says so); the late
  normalization is real but secondary, and this test cannot fail or pass
  independently of item 31's.
- **Why the test is right.** Numpy convention only; the paper works with a
  single time series and does not define a channel axis. Medium.
- **Impact.** Unreachable (axis=1 in production).
- **Fix.** Normalise at entry, together with items 31 and 33. Easy.

## Tier D: cosmetic

### 38. `get_capture_type_attrs` returns empty attrs

`tests/analysis/test_analysis_specs_helpers.py::test_capture_type_attrs`

- **Mechanism.** `analysis/specs/helpers.py:171` reads `.extra` off the raw
  `Annotated` alias from `msgspec.structs.fields`, which has none; the
  metadata is on `msgspec.inspect.type_info(cls).fields`. Every field returns
  `{}`.
- **Why the test is right.** The `Meta` docstring and CLAUDE.md say it feeds
  `standard_name`/`units`.
- **Impact.** The only caller is `describe_field` behind the per-capture INFO
  log line, which prints `duration=0.001` instead of `duration=1 ms`. The
  zarr capture coordinates are built by `_coords_template` through
  `infer_coord_info`, which reads the metadata correctly, so saved output is
  unaffected (verified).
- **Fix.** Iterate `type_info(cls).fields` or reuse `infer_coord_info`. Trivial.

### 46. Range error text embeds a DataArray repr

`tests/sensor/test_sensor_calibration.py::TestLookupPowerCorrection::test_out_of_range_message_shows_the_limit_in_mhz`

- **Mechanism.** `calibration.py:580` and `:584` format
  `sel.center_frequency.max() / 1e6`, a 0-d `DataArray`, so the message
  reads `exceeds calibration max <xarray.DataArray ...> array(2000.) MHz`.
- **Fix.** Wrap in `float()`. Trivial.

### 71. `CaptureRemap.__post_init__` leaks msgspec decode errors

`tests/sensor/test_sensor_specs_structs.py::TestCaptureRemap::test_multi_key_undecodable_key_is_a_validation_error[text]`, `[int]`

- **Mechanism.** The multi-key form decodes each key with
  `msgspec.json.decode`, and a non-JSON or non-string key surfaces as
  `DecodeError` or `TypeError` ("bytes-like object") rather than a
  `ValidationError` naming the lookup key.
- **Fix.** Catch and re-raise as `ValidationError` with the key. Trivial.

### 48. The saved calibration loses `calibration_fields`

`tests/sensor/test_sensor_calibration.py::TestYFactorSinkFlush::test_saved_attrs_record_the_calibration_fields`

- **Mechanism.** `YFactorSink.flush` assigns `calibration_fields` and
  `sweep_start_time` attrs at `calibration.py:184`, but
  `_y_factor_power_corrections` returns a new `xr.Dataset`
  (`calibration.py:458`) that carries none of the input attrs, and that is
  what `io.save_calibration` writes.
- **Impact.** Nothing in `src/` reads the attribute. Metadata only.
- **Fix.** `.assign_attrs(by_field.attrs)` on the result. Trivial.

### 41. `BoundSweep` error text says `mock_sensor`

`tests/sensor/test_sensor_bindings.py::TestMockSource::test_message_names_the_field`

- **Mechanism.** `bindings.py:120` names a stale field. `_convert_dict_spec`
  raises a `KeyError` first, so the CLI never shows this text anyway.
- **Fix.** String change. Trivial. On 2026-09-16 the stale `mock_sensor`
  field on `BoundSweep` was renamed and the `SensorBinding` assertion fixed,
  but the message still says `mock_sensor`, so the xfail stands.

### 56. Unsupported time source message is not an f-string

`tests/sensor/test_sensor_sources_soapy.py::TestHardwareTimeSync::test_unsupported_source_names_itself`

- **Mechanism.** `soapy.py:515` reads `'unsupported time source
  {self.time_source!r}'` literally.
- **Impact.** Unreachable through specs: `types.TimeSource` is a `Literal`.
- **Fix.** Add the `f`. Trivial.

### 58. `RxStream.close` prints to stdout

`tests/sensor/test_sensor_sources_soapy.py::TestRxStreamClose::test_close_is_silent`

- **Mechanism.** `soapy.py:379`, a leftover `print('close stream')`.
- **Fix.** Delete it, or log at DEBUG on the `source` logger. Trivial.

### 39. The calibration capture class is named `capture_spec_cls`

`tests/sensor/test_sensor_calibration.py::test_calibration_capture_class_has_a_descriptive_name`

- **Mechanism.** `calibration.py:321` defines the class inline and renames
  only the peripherals class (`calibration.py:339`).
- **Impact.** Surfaces in the JSON schema `$defs`, the `adjust_captures`
  error text, and the `Controller.arm` signature. Not in zarr attrs.
- **Fix.** Set `__name__` and `__qualname__`. Trivial.

### 50. `air7201b` binds `init_like=Air7101BSourceSpec`

`tests/sensor/test_sensor_sources_deepwave.py::test_air7201b_init_like_matches_its_source_spec`

- **Mechanism.** `bindings.py:149`. `Schema.init_like` is documented as
  type-hinting only, so the wrong class affects editor hints for the
  `air7201b` controller and nothing at run time.
- **Fix.** Name change. Trivial.

### 69. `get_capture_type` returns the TypeVar for an unbound `Sweep` subclass

`tests/sensor/test_sensor_specs_captures.py::test_capture_type_of_an_unbound_sweep`

- **Mechanism.** The unbound branch uses `get_type_hints`, which does not
  substitute the generic parameters of
  `class X(Sweep[FunctionSource, NoPeripherals, SingleToneCapture])`, so the
  bare `SC` TypeVar is returned.
- **Impact.** Every in-tree caller passes a bound sweep.
- **Fix.** Read `__orig_bases__` for the parametrized base. Easy.

## Tool limitations

Gaps in the checkers and test tooling the suite relies on, not in striqt. They are
strict xfails for the same reason as the defects: the run that gains the fix (here,
a `ty` release) must fail with an xpass so that the marker is removed and the guard
it was standing in for is enabled.

### 91. ty 0.0.81 accepts unknown keywords through `Unpack[TypedDict]`

`tests/test_typing.py::test_unknown_keyword_through_unpack_is_flagged` (the probe
`tests/ty_probes/xfail_unpack_unknown_keyword.py`)

- **Mechanism.** `ty` 0.0.81 (the `dev` pin) lowers a `**kwargs:
  Unpack[SomeTypedDict]` parameter to `**kwargs: object`: `reveal_type` of
  `sensor/lib/compute/corrections.py:185` `design_resampler` ends in
  `..., window: str = ..., **kwargs: object`, and `sensor/lib/execute.py:21`
  `iterate_sweep` in `..., **replace: object`. The known keys keep their
  declared types, so a wrong value (`bw_lo='x'`, `sink=3`) and a missing required
  key are reported, but a keyword that is not in the TypedDict
  (`design_resampler(cap, 1e6, bogus=1)`) matches the `object` catch-all and
  produces no diagnostic.
- **Why the test is right.** PEP 692 specifies that a call passing a keyword the
  unpacked TypedDict does not declare is a type error, exactly as for an
  explicit keyword-only parameter list. `ty` already applies that rule to a
  TypedDict literal (`invalid-key`) and to a plain signature (`unknown-argument`),
  which is the rule the test expects.
- **Impact.** None on striqt's behavior. The two call sites that forward
  `**kwargs: Unpack[...]` (`design_resampler` to `sw.design_cola_resampler`,
  `iterate_sweep` to the `Resources` replacement) are the places where a
  mistyped keyword is not caught statically; at runtime the misspelt key still
  reaches the callee and raises there.
- **Fix.** Nothing in striqt; the test flips to an unexpected pass when the `ty`
  pin in `pyproject.toml` moves to a release that checks unpacked TypedDict
  keys, and the marker comes off then.

## Observations without an xfail

- **Ambient temperature is not used.** `_y_factor_power_corrections` takes
  every temperature from `Tref` (290 K, by convention) and never reads
  `calibration.ambient_temperature`; the tests use a receiver model at 290 K
  and so cannot tell. The user is opening an issue for this. The only code that
  did take an ambient temperature, `_y_factor_temperature` (and its dead
  caller `_y_factor_frequency_response_correction`), was removed on
  2026-09-16; it had defined `Ton = Tref * 10**(ENR/10)` without the `+1`
  that the IEEE ENR definition implies and that the live formula
  `NF = ENR_dB - 10 log10(Y - 1)` relies on.

- **`open_resources` never restores the working directory.**
  `resources.py:200` `chdir`s to the spec directory and nothing undoes it, so
  the CLI and every test that runs a sweep inherit that cwd. Not fixed on
  2026-09-16 because `Zipper` keeps a relative `zip_path` and `temp_dir` and
  archives at sink close, after the sweep, so restoring the cwd would move
  `.zarr.zip` output unless the sink first resolves its path against the spec
  directory (zarr 2.18's `DirectoryStore` absolutises its own path, so
  directory stores are unaffected). `test_sensor_sweeps` restores the cwd itself.

- **`summarize_calibration` labels a linear column as dB.** `calibration.py:39`
  names the column `Power Corr (dB)` but `_summarize_calibration_field`
  returns `power_correction` unchanged (units `mW/fs`). Whether the label or
  the value is the mistake is a judgment call, so the xfail that recorded it
  (item 47) was withdrawn; the mismatch itself remains.

- **`cast_iq` widens int16 in place over overlapping memory.** `buffers.py:286-295`
  copies the int16 view into a float32 view of the same buffer with
  `xp.copyto`. numpy detects the overlap and buffers; cupy does not, so on the
  Jetson with `transport_dtype='int16'` this looks like a data race. Unverified
  here; needs a hardware check.

- **`read_iq`'s silent `break` is unreachable today.** `controller.py:501-504`
  drops the tail of the buffer if `received + request > buffer_count`. With the
  virtual sources `buffer_count == output_count` and the loop invariant keeps
  the sum in range, so the branch only fires if `find_trigger_holdoff` returns
  more than the `2*trigger_strobe` padding. Nothing pinned.

- **`find_trigger_holdoff` is not minimal.** When the first strobe edge falls
  short of the minimum holdoff it adds `ceil(min/strobe)*strobe` and can skip
  one extra period (up to `min_holdoff + 2*strobe`). Pinned as current
  behavior in `test_sensor_sources_buffers.py`; alignment within one sample and
  the lower bound are the asserted contract.

- **`cast_iq` slices `2*acquired_count` complex samples.** `buffers.py` on the
  float32/complex64 path slices `buffer[:, :2*acquired_count]`, a leftover from
  a float32 view, while the int16 path yields exactly `acquired_count`. Harmless
  while `read_iq` passes the full buffer width; tested only at full width.

- **`mock_binding(register=False)` still extends the tagged union.**
  `lib/bindings.py:129-133` appends the mock `BoundSweep` to `tagged_sweeps`,
  so after any warmup a YAML with `sensor_binding: mock_warmup_single_tone`
  decodes.

- **`cancel_threads()` from a peripheral does not stop a main-thread sweep.**
  `propagate_thread_interrupts` is a no-op on the main thread by design and the
  function sources never check the flag; `test_sensor_execute.py` consumes
  `iterate_sweep` in a worker thread to observe the interrupt.

- **`ZarrTimeAppendSink` concatenates along time only within a batch.**
  Separate batches are still appended along `capture` by `sa.dump`'s default
  `append_dim`; tested with `batched_write_count == len(captures)`.

- **`analysis/lib/io.py` `TDMSFileStream.read` references an undefined
  `float_dtype`.** The same latent `np.finfo(...).dtype` misuse that was fixed
  in `sensor/lib/sources/file.py` on 2026-09-18; the analysis-side reader is
  unused by the sensor.

- **`isolated_extension_import` does not restore `ss.lib.bindings.registry`.**
  `tests/conftest.py` snapshots `sys.path` and `sys.modules` around an extension
  import but not the binding registry, so a binding registered by one test stays
  registered for the rest of the session. `tests/sensor/site_strategies.py` is
  coupled to that leak: it guards its own registration with
  `if 'site_single_tone' not in ss.lib.bindings.registry`.

- **`tests/sensor/sweeps/outputs/` holds 2.0 GB of stale stores.** Gitignored, 662
  entries, written by sweep tests that no longer exist; nothing writes there since
  the 2026-09-18 refocus. `sweeps/air7101b.yaml:7` still names
  `calibration: outputs/calibration.nc`, which is not among them.

- **`fragments/captures/tone.yaml` sets an `adjust_analysis` key no measurement
  carries.** `frame_slots: null` at `:14` is not a field of any measurement in
  `fragments/analysis/quick.yaml`, so every `site-cpu` run logs an unused-key
  warning.

## Fixed on 2026-09-21 without a ledger number

- **A canceled source lookup reported a timeout instead of the cancellation.**
  `controller.lookup.instance` raised
  `TimeoutError('no controller instance initializing given spec')` when its 0.5 s
  wait expired, without checking whether the open had been canceled. In
  `open_resources` the sink, the devices and `_prepare_sweep` run concurrently
  under `ExceptionStack(cancel_on_except=True)`, so a sink that raised in
  `__init__` could cancel the open before the source thread registered a
  controller; `_prepare_sweep`'s `lookup.id` then contributed that `TimeoutError`
  to the group and `ExceptionStack.handle()` raised an `ExceptionGroup` instead of
  the sink's own error. `instance` now calls `util.propagate_thread_interrupts()`
  before raising, which turns a canceled wait into the `ThreadInterruptRequest`
  that `ExceptionStack` already makes yield to the real error; an uncancelled wait
  still times out, as the class docstring promises. Confirming test:
  `tests/sensor/test_sensor_resources.py::test_sink_failure_surfaces_and_closes_the_source[init]`,
  which failed in all 3 full-suite py314 runs before the fix and passes in 4 of 4
  after, with 10 of 10 clean `pytest tests/sensor` runs on py314.

## Fixed on 2026-09-18 without a ledger number

Found and fixed in the same change by the sensor test refocus (each has a
confirming test named in the commit):

- `VirtualSource.read` origin: buffer position `p` now holds generator or file
  index `p - overlaps[0]`, so corrected sample 0 is generator index 0 for every
  synthetic and file source (previously `2*overlaps[0]` for tone, noise and
  sawtooth, compensated only in `DiracDeltaSource`, and `overlaps[0]` for
  files). `striqt.analysis.testing` generators accept negative `start_index`;
  noise pre-roll comes from `RandomState(seed + PREROLL_SEED_OFFSET)`.
- `correct_iq` conjugated `iq.pre_align` in place on the scale-only path even
  with `overwrite_x=False` (`corrections.py:82`).
- `buffers._alloc_empty_iq` compared `prior.shape < (ports, count)`
  lexicographically, reusing too-small or wrong-row-count buffers
  (`buffers.py:198`); a one-port capture after a two-port capture returned a
  stale second row.
- `Controller.acquire` with `reuse_iq=True` called `info.replace(start_time=None)`
  on `AcquisitionInfo`, which has no such field (`controller.py:611`).
- `Controller.acquire(overlaps=<tuple>)` raised `UnboundLocalError` for
  `signal_trigger` (`controller.py:567`).
- `buffers.get_read_count` sized a resampled read with `ceil` while the pad was
  built from `round(duration*fs_sdr)`, so a non-integral source-sample count
  made `_resample` fail `isroundmod` (`buffers.py:144`).
- `NoSource.read` added `samples_elapsed*sample_period_ns` as a float to a
  wall-clock `time_ns()` origin, quantizing timestamps to 256 ns
  (`sources/base.py`).
- `gpu.sweep_touches_gpu` tested `analysis_bandwidth is not None` (always true)
  and the loop branch never fired for finite values; `build_warmup_sweep`
  sized `num_rx_ports = max(port)` (one short) and passed the unhashable
  `SensorBinding` to `mock_binding`, so no warmup sweep could be built.
- `EvaluationOptions.extra_attrs` defaulted to a `dataclasses.field` object on
  a msgspec Struct (`datasets.py:35`).
- `ZarrTimeAppendSink` tested `'spectrogram' not in analysis` on a non-iterable
  Struct and could never be constructed (`sinks.py:322`).
- `TDMSSource` and `ZarrIQSource` were abstract (missing `close`, and for TDMS
  `get_id`/`get_info`), `TDMSSource.get_waveform` called a dtype object,
  `TDMSSource` inherited `transport_dtype='float32'`, `MATSource` passed
  `key=None` over the stream's default, TDMS output indexing overflowed for a
  non-zero offset, and Zarr's port bound check was `>` instead of `>=`
  (`sources/file.py`, `specs/structs.py`). None of the three file bindings had
  ever been instantiable.

