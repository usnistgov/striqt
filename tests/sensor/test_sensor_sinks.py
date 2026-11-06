"""striqt.sensor.lib.sinks: the sinks written through a real sweep, batch sizing, and the
zip archive path
"""

from __future__ import annotations

import dataclasses
import logging
import zipfile
from concurrent.futures import Future
from pathlib import Path

import numpy as np
import pytest
from conftest import FAKE_SOAPY_SPEC
from sweep_strategies import make_cellular_sweep
from synthetic_sources import (
    IQ_ONLY,
    ONE_PORT,
    SPECTROGRAM,
    mixed_rate_sweep,
    spectrogram_frames,
    tone_capture,
    tone_captures,
    tone_sweep,
)

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.sensor.lib import compute, sinks
from striqt.sensor.lib.sources import buffers

OFFSETS = (1e6, 2e6)
TONE = ss.bindings.single_tone


def zarr_sweep(path, offsets, *, analysis=IQ_ONLY, **sink_kws):
    """return a tone sweep with a `ZarrSink` at `path`"""
    sweep = tone_sweep(offsets, analysis=analysis)
    return sweep.replace(sink=ss.specs.ZarrSink(path=str(path), **sink_kws))


def run_to_store(sweep):
    """return the datasets that the sink returned from each append"""
    with ss.open_resources(sweep, None) as res:
        return [ds for ds in ss.iterate_sweep(res) if ds is not None]


# %% _BatchTracker


def test_batch_tracker_counts_each_burst_segment():
    """the segments of a burst are appended one at a time, so 2 captures of `burst=2`
    expanded by `expand_bursts` batch as 4 acquisitions in 2 groups of 2"""
    captures = tone_captures((1e6,) * 2, burst=2)
    expanded = ss.specs.sequencing.expand_bursts(captures)
    tracker = sinks._BatchTracker(expanded, min_size=2)
    assert tracker.total_size == 4
    assert tracker.size == 2
    assert tracker.next() == 2


# %% sweep_row_count


def test_sweep_row_count_sums_ports_times_burst():
    """one row per port per burst segment: a 1-port capture, a 1-port `burst=3`
    capture, and a 2-port `burst=2` capture give 1 + 3 + 4 rows"""
    captures = (
        *tone_captures((1e6,)),
        *tone_captures((2e6,), burst=3),
        tone_capture({**ONE_PORT, 'port': (0, 1)}, burst=2),
    )
    sweep = tone_sweep((1e6,), analysis=IQ_ONLY).replace(captures=captures)
    assert compute.datasets.sweep_row_count(sweep, 'x') == 1 + 3 + 4


# %% ZarrCaptureSink


def materialize(delayed):
    """return the `xr.Dataset` of each `DelayedDataset` that the sink returned"""
    return [compute.from_delayed(dd) for dd in delayed]


def outer_join(datasets):
    import xarray as xr

    return xr.concat(datasets, 'capture', join='outer', combine_attrs='drop_conflicts')


def json_attrs(attrs):
    """return `attrs` as a zarr store reads them back: tuples become lists"""
    import json

    return json.loads(json.dumps(dict(attrs)))


def test_zarr_capture_sink_store_equals_the_concatenated_captures(tmp_path):
    """the store holds what `xr.concat(join='outer')` of the per-capture datasets
    holds, with a chunk budget that leaves one partial chunk at close"""
    import xarray as xr

    offsets = (1e6, 2e6, 3e6)
    sweep = zarr_sweep(tmp_path / 'out.zarr', offsets, max_chunk_bytes=200_000)
    results = materialize(run_to_store(sweep))

    stored = sa.load(sweep.sink.path)
    expected = outer_join(results)
    assert stored.capture_index.values.tolist() == list(range(len(offsets)))
    assert stored.iq_waveform.dtype == np.complex64
    xr.testing.assert_equal(stored.iq_waveform, expected.iq_waveform)
    for name in expected.coords:
        if 'capture' in expected[name].dims:
            a, b = stored[name].values, expected[name].values
            if a.dtype.kind == 'f':
                assert np.array_equal(a, b, equal_nan=True), name
            else:
                assert np.array_equal(a, b), name
    assert dict(stored.iq_waveform.attrs) == json_attrs(expected.iq_waveform.attrs)


def test_zarr_capture_sink_allocates_the_store_at_open(tmp_path):
    """before any append the store has every variable and coordinate at the full
    row count of the sweep, and no data chunk"""
    sweep = zarr_sweep(tmp_path / 'out.zarr', (1e6, 2e6))
    unsunk = sweep.replace(sink=sweep.sink.replace(path=None))
    with ss.open_resources(unsunk, None) as res:
        source_id = res['source'].source_id
        sink = ss.sinks.ZarrCaptureSink(sweep)
        with sink:
            stored = sa.load(sweep.sink.path, consolidated=False)
            rows = compute.datasets.sweep_row_count(sweep, source_id)
            assert stored.sizes['capture'] == rows
            assert set(stored.data_vars) == set(sweep.analysis.to_dict())
            assert stored.iq_waveform.dtype == np.complex64
            assert {'capture_index', 'port', 'source_id'} <= set(stored.coords)
            # the fixed coordinates are written at open, and no row of a variable is
            chunk_files = [
                p
                for name in stored.data_vars
                for p in (Path(sweep.sink.path) / str(name)).rglob('*')
                if p.is_file() and not p.name.startswith(('.z', 'zarr.json'))
            ]
            assert chunk_files == []
            # an unwritten store is trimmed to zero rows at close
    assert sa.load(sweep.sink.path).sizes['capture'] == 0


def test_zarr_capture_sink_waits_for_the_backend_of_a_pending_source(
    tmp_path, isolated_lookup, monkeypatch
):
    """the sink waits for the backend of a source that is registered but not yet
    constructed, and reads the power units once the backend exists. A `lookup.id`
    stub that does not wait does not change that."""
    from threading import Event, Thread

    sweep = tone_sweep((1e6,), sink=ss.specs.ZarrSink(path=str(tmp_path / 'p.zarr')))
    monkeypatch.setattr(isolated_lookup, 'id', lambda spec, timeout=0.5: 'beef')
    sink_waiting = Event()
    sink_done = Event()
    outcome = []

    class WaitingIdSlot(Event):
        def wait(self, timeout=None):
            sink_waiting.set()
            return super().wait(timeout)

    isolated_lookup._id[sweep.source] = WaitingIdSlot()
    real_buffers = buffers.ReceiveBuffers

    def buffers_after_the_sink_asks(controller):
        # the controller is registered but has no backend until this returns
        for _ in range(600):
            if sink_waiting.wait(timeout=0.05) or sink_done.is_set():
                break
        return real_buffers(controller)

    monkeypatch.setattr(buffers, 'ReceiveBuffers', buffers_after_the_sink_asks)

    def open_sink():
        try:
            outcome.append(sinks.ZarrCaptureSink(sweep, None)._power_units)
        except BaseException as ex:
            outcome.append(ex)
        finally:
            sink_done.set()

    opener = Thread(target=open_sink)
    opener.start()
    with TONE.from_sweep_spec(sweep):
        opener.join(timeout=30)

    assert not opener.is_alive(), 'the sink is still blocked'
    assert outcome == ['mW']


def test_zarr_capture_sink_template_units_follow_the_source(fake_soapy_ext, tmp_path):
    """an uncalibrated SoapySDR source is in full scale: the `units` attrs of the power
    variable and the power bin coordinate are dBFS in the template allocated at open,
    in the appended rows, and in the written store"""
    spec = ss.read_yaml_spec(FAKE_SOAPY_SPEC)
    histogram = {
        'detector_period': 1e-4,
        'power_low': -120,
        'power_high': 0,
        'power_resolution': 1,
    }
    analysis = type(spec.analysis).from_dict({
        'channel_power_time_series': {'detector_period': 1e-4},
        'channel_power_histogram': histogram,
    })
    sweep = spec.replace(
        captures=(spec.captures[1].replace(duration=1e-3),),
        analysis=analysis,
        sink=spec.sink.replace(path=str(tmp_path / 'fake_soapy.zarr')),
    )
    names = ('channel_power_time_series', 'channel_power_bin')

    with ss.open_resources(sweep, FAKE_SOAPY_SPEC, test_only=True) as res:
        template = sa.load(sweep.sink.path, consolidated=False)
        template_units = {name: template[name].attrs['units'] for name in names}
        results = materialize(dd for dd in ss.iterate_sweep(res) if dd is not None)

    stored = sa.load(sweep.sink.path)
    assert template_units == dict.fromkeys(names, 'dBFS')
    assert {name: stored[name].attrs['units'] for name in names} == template_units
    assert {name: results[0][name].attrs['units'] for name in names} == template_units


@pytest.mark.parametrize('store', ['directory', 'zip'], ids=['directory', 'zip'])
def test_zarr_capture_sink_writes_each_key_once(tmp_path, caplog, store):
    """every chunk and metadata key of the store is written one time"""
    import logging

    name = 'out.zarr.zip' if store == 'zip' else 'out.zarr'
    sweep = zarr_sweep(tmp_path / name, (1e6, 2e6, 3e6), store=store)
    with caplog.at_level(logging.WARNING, logger='striqt.analysis'):
        run_to_store(sweep)
    assert not [r for r in caplog.records if 'already written' in r.getMessage()]

    path = Path(sweep.sink.path)
    if store == 'zip':
        assert zipfile.is_zipfile(path)
        names = zipfile.ZipFile(path).namelist()
        assert len(names) == len(set(names))
    stored = sa.load(path)
    assert stored.sizes['capture'] == 3
    assert ss.read_zarr_spec(path) == sweep


def test_zarr_capture_sink_unions_mixed_grids_like_outer_join(tmp_path):
    """captures with different frequency grids share the union grid with NaN fill,
    and the attrs that differ between the captures are dropped"""
    import xarray as xr

    sweep = mixed_rate_sweep(tmp_path / 'mixed.zarr')
    results = materialize(run_to_store(sweep))
    expected = outer_join(results)
    stored = sa.load(sweep.sink.path)

    xr.testing.assert_equal(
        stored.power_spectral_density, expected.power_spectral_density
    )
    assert np.array_equal(
        stored.baseband_frequency.values, expected.baseband_frequency.values
    )
    assert dict(stored.power_spectral_density.attrs) == json_attrs(
        expected.power_spectral_density.attrs
    )
    assert dict(stored.attrs) == json_attrs(expected.attrs)


def test_zarr_capture_sink_writes_extra_data_variables(tmp_path):
    """peripheral data appears as variables with the dims `from_delayed` gives them"""
    import xarray as xr

    sweep = zarr_sweep(tmp_path / 'extra.zarr', (1e6, 2e6))
    extra = {
        'state': 1.5,
        'temperature': xr.DataArray(
            np.arange(3.0),
            dims=['temperature_source'],
            coords={'temperature_source': ['cpu', 'gpu', 'board']},
        ),
    }
    unsunk = sweep.replace(sink=sweep.sink.replace(path=None))
    with (
        ss.open_resources(unsunk, None) as res,
        ss.sinks.ZarrCaptureSink(sweep) as sink,
    ):
        results = [
            compute.from_delayed(dd)
            for dd in ss.lib.actions.core.iterate_captures(
                res, extra_data=extra, sink=sink
            )
            if dd is not None
        ]
    stored = sa.load(sweep.sink.path)
    expected = outer_join(results)
    assert stored.state.dims == ('capture',)
    assert stored.temperature.dims == ('capture', 'temperature_source')
    assert np.array_equal(stored.state.values, expected.state.values)
    assert np.array_equal(stored.temperature.values, expected.temperature.values)
    assert stored.temperature_source.values.tolist() == ['cpu', 'gpu', 'board']


def test_zarr_capture_sink_rejects_a_dataset_missing_an_allocated_variable(tmp_path):
    """the store is allocated from the whole analysis group of the sweep. A
    `select_analysis` hook that drops a measurement fails at the first write."""
    analysis = ss.specs.BundledAnalysis.from_dict({
        **IQ_ONLY.to_dict(),
        **SPECTROGRAM.to_dict(),
    })
    sweep = zarr_sweep(tmp_path / 'narrowed.zarr', OFFSETS, analysis=analysis)
    unsunk = sweep.replace(sink=sweep.sink.replace(path=None))
    callbacks = ss.lib.resources.SweepCallbacks(
        select_analysis=lambda capture, analysis: (
            sa.specs.helpers.narrow_analysis_group(analysis, ['iq_waveform'])
        )
    )
    with (
        pytest.raises(ValueError, match=r"\['spectrogram'\]"),
        ss.open_resources(unsunk, None) as res,
        ss.sinks.ZarrCaptureSink(sweep) as sink,
    ):
        list(ss.lib.actions.core.iterate_captures(res, sink=sink, callbacks=callbacks))


def burst_segments(dd, burst):
    """return `dd` as the `burst` segments of one capture of `burst` acquisitions, each
    with its own `burst_index`"""
    capture = dd.capture.replace(burst=burst)
    return [
        dataclasses.replace(
            dd, capture=capture, extra_coords=dd.extra_coords.replace(burst_index=k)
        )
        for k in range(burst)
    ]


def pending(dd, **override):
    """return `dd` with an unresolved peripheral read, and the future to resolve"""
    future: Future = Future()
    pending = ss.specs.PendingExtraData(future, override)
    return dataclasses.replace(dd, pending_extra_data=pending), future


def test_zarr_capture_sink_defers_rows_behind_a_pending_peripheral_read(tmp_path):
    """rows wait in acquisition order until the peripheral read of the head
    resolves. The peripheral dict then joins the rows under the override."""
    burst = 3
    unsunk = tone_sweep((1e6,))
    (dd,) = [dd for dd in run_to_store(unsunk) if dd is not None]
    first, second, third = burst_segments(dd, burst)
    first, future = pending(first, shared='runner')

    sweep = unsunk.replace(
        captures=(first.capture,),
        sink=ss.specs.ZarrSink(path=str(tmp_path / 'deferred.zarr')),
    )
    with (
        ss.open_resources(unsunk, None),
        ss.sinks.ZarrCaptureSink(sweep) as sink,
    ):
        sink.append(first)
        sink.append(second)
        assert sink.rows_written == 0

        future.set_result({'temperature': 21.5, 'shared': 'peripheral'})
        sink.append(third)
        assert sink.rows_written == burst

    stored = sa.load(sweep.sink.path)
    assert stored.burst_index.values.tolist() == list(range(burst))
    assert stored.temperature.values.tolist()[0] == 21.5
    assert stored.shared.values.tolist()[0] == 'runner'


def test_zarr_capture_sink_writes_the_rows_of_a_failed_peripheral_read(
    tmp_path, caplog
):
    """a failed read is done, so the rows are written at once without the peripheral
    variables and with one WARNING"""
    unsunk = tone_sweep((1e6,))
    (dd,) = [dd for dd in run_to_store(unsunk) if dd is not None]
    dd, future = pending(dd)
    future.set_exception(RuntimeError('simulated peripheral failure'))

    sweep = unsunk.replace(
        sink=ss.specs.ZarrSink(path=str(tmp_path / 'failed.zarr')),
    )
    with (
        caplog.at_level(logging.WARNING, logger='striqt.sink'),
        ss.open_resources(unsunk, None),
        ss.sinks.ZarrCaptureSink(sweep) as sink,
    ):
        sink.append(dd)
        assert sink.rows_written == 1
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1

    stored = sa.load(sweep.sink.path)
    assert stored.sizes['capture'] == 1
    assert 'temperature' not in stored


def test_zarr_capture_sink_writes_one_row_group_per_burst_segment(tmp_path):
    """the `burst` appends of one 1-port capture fill the `burst` planned rows, with
    `burst_index` counting the segments and `capture_index` shared"""
    burst = 3
    unsunk = tone_sweep((1e6,))
    (dd,) = [dd for dd in run_to_store(unsunk) if dd is not None]
    segments = burst_segments(dd, burst)

    sweep = unsunk.replace(
        captures=(segments[0].capture,),
        sink=ss.specs.ZarrSink(path=str(tmp_path / 'burst.zarr')),
    )
    with (
        ss.open_resources(unsunk, None),
        ss.sinks.ZarrCaptureSink(sweep) as sink,
    ):
        for segment in segments:
            sink.append(segment)
        assert sink.rows_written == burst
        assert sink.acquisitions_elapsed == burst

    stored = sa.load(sweep.sink.path)
    assert stored.sizes['capture'] == burst
    assert stored.burst_index.values.tolist() == list(range(burst))
    assert stored.burst.values.tolist() == [burst] * burst
    assert stored.capture_index.values.tolist() == [0] * burst
    assert stored.port.values.tolist() == [0] * burst


@pytest.mark.parametrize(
    'offsets, loop, taken, capture_index',
    [
        ((1e6, 2e6), True, 5, [0, 1, 0, 1, 0]),
        ((1e6, 2e6, 3e6), False, 1, [0]),
    ],
    ids=['grows_past_the_plan', 'trims_to_the_rows_written'],
)
def test_zarr_capture_sink_store_has_the_rows_written(
    tmp_path, offsets, loop, taken, capture_index
):
    """the store has one row per appended capture: a looped sweep stopped after 5
    writes keeps more rows than the 2 planned, and a sweep stopped after 1 of 3
    captures leaves 1 row"""
    sweep = zarr_sweep(tmp_path / 'rows.zarr', offsets)
    unsunk = sweep.replace(sink=sweep.sink.replace(path=None))
    with (
        ss.open_resources(unsunk, None) as res,
        ss.sinks.ZarrCaptureSink(sweep) as sink,
    ):
        written = 0
        for dd in ss.iterate_sweep(res, loop=loop, sink=sink):
            written += dd is not None
            if written == taken:
                break
    stored = sa.load(sweep.sink.path)
    assert stored.sizes['capture'] == taken
    assert stored.capture_index.values.tolist() == capture_index


def test_zarr_sink_replaces_an_existing_store_only_with_force(tmp_path):
    sweep = zarr_sweep(tmp_path / 'twice.zarr', (1e6,))
    run_to_store(sweep)
    with pytest.raises(FileExistsError):
        run_to_store(sweep)

    unsunk = sweep.replace(sink=sweep.sink.replace(path=None))
    with (
        ss.open_resources(unsunk, None) as res,
        ss.sinks.ZarrCaptureSink(sweep, force=True) as sink,
    ):
        list(ss.iterate_sweep(res, sink=sink))
    assert sa.load(sweep.sink.path).sizes['capture'] == 1


def test_zarr_zip_path_is_written_directly_and_readable(tmp_path):
    path = tmp_path / 'out.zarr.zip'
    sweep = zarr_sweep(path, (1e6, 2e6))
    results = materialize(run_to_store(sweep))

    assert zipfile.is_zipfile(path)
    assert not path.with_suffix('').exists()
    assert ss.read_zarr_spec(path) == sweep
    stored = sa.load(path)
    assert np.array_equal(
        stored.iq_waveform.values, outer_join(results).iq_waveform.values
    )


# %% ZarrTimeAppendSink


def test_time_append_sink_requires_a_spectrogram():
    sweep = zarr_sweep('unused.zarr', (1e6,))
    with pytest.raises(ValueError, match='spectrogram'):
        ss.sinks.ZarrTimeAppendSink(sweep)


def test_time_append_sink_concatenates_the_frames(tmp_path):
    """2 captures written as 1 batch form 1 row with the frames of both captures back
    to back. The time grid of the join is tested at `concat_time_dim`."""
    sweep = zarr_sweep(tmp_path / 'spg.zarr', OFFSETS, analysis=SPECTROGRAM)
    frames, _ = spectrogram_frames(sweep.captures[0], SPECTROGRAM.spectrogram)

    # the ZarrCaptureSink of the runner would open the same store if the opened sweep
    # kept the path. A zarr sink looks up the source id in its constructor, which
    # requires the source to be open first
    unsunk = sweep.replace(sink=sweep.sink.replace(path=None))
    with (
        ss.open_resources(unsunk, None) as res,
        ss.sinks.ZarrTimeAppendSink(sweep, batch_size=2) as sink,
    ):
        results = [ds for ds in ss.iterate_sweep(res, sink=sink) if ds is not None]

    stored = sa.load(sweep.sink.path)
    assert stored.sizes['capture'] == 1
    assert stored.sizes['spectrogram_time'] == 2 * frames
    for i, ds in enumerate(results):
        half = slice(i * frames, (i + 1) * frames)
        assert np.array_equal(
            stored.spectrogram.values[0, half], ds.spectrogram.values[0]
        )


# %% runner sink selection


def test_zarr_sink_rejects_a_null_path():
    with pytest.raises(ValueError, match='path'):
        ss.sinks.ZarrCaptureSink(tone_sweep(OFFSETS))


@pytest.mark.parametrize(
    'make_spec, name, sink_type',
    [
        (lambda: tone_sweep(OFFSETS), None, ss.sinks.NoSink),
        (lambda: tone_sweep(OFFSETS), 'out.zarr', ss.sinks.ZarrCaptureSink),
        (make_cellular_sweep, None, ss.sinks.NoSink),
        (make_cellular_sweep, 'out.zarr', ss.sinks.ZarrCaptureSink),
    ],
    ids=['sweep_null_path', 'sweep_path', 'cellular_null_path', 'cellular_path'],
)
def test_runner_opens_the_sink_that_the_path_selects(
    tmp_path, make_spec, name, sink_type
):
    path = None if name is None else str(tmp_path / name)
    spec = make_spec().replace(sink=ss.specs.ZarrSink(path=path))
    with ss.open_resources(spec, None) as res:
        assert type(res['sink']) is sink_type


# %% Zipper


def make_tree(root: Path) -> set[str]:
    """write a small directory tree under `root`.

    Returns:
        the file paths relative to `root`
    """
    files = {'zarr.json', 'a/c/0.0', 'a/zarr.json', 'b/0'}
    for rel in files:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(rel.encode())
    return files


@pytest.mark.parametrize('remove', [True, False], ids=['remove', 'keep'])
def test_zipper_from_zarr_archives_every_file(tmp_path, remove):
    store = tmp_path / 'data.zarr'
    files = make_tree(store)
    out = sinks.Zipper.from_zarr(store).archive(remove=remove)

    assert Path(out) == tmp_path / 'data.zarr.zip'
    with zipfile.ZipFile(out) as zf:
        assert set(zf.namelist()) == files
        assert zf.read('a/c/0.0') == b'a/c/0.0'
    assert store.is_dir() == (not remove)


def test_zipper_refuses_an_existing_zip_without_force(tmp_path):
    store = tmp_path / 'data.zarr'
    make_tree(store)
    (tmp_path / 'data.zarr.zip').write_bytes(b'')
    with pytest.raises(IOError, match='already exists'):
        sinks.Zipper.from_zarr(store)
    sinks.Zipper.from_zarr(store, force=True)


def test_zipper_from_zarr_requires_the_directory(tmp_path):
    with pytest.raises(IOError, match='no such directory'):
        sinks.Zipper.from_zarr(tmp_path / 'missing.zarr')
