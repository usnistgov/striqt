"""striqt.sensor.lib.sinks: the sinks written through a real sweep, batch sizing, and the
zip archive path
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
from numeric_checks import assert_close, elementwise_rtol
from sweep_strategies import make_cellular_sweep
from synthetic_sources import (
    IQ_ONLY,
    ONE_PORT,
    SPECTROGRAM,
    preset_capture,
    preset_sweep,
    spectrogram_frames,
    tone_captures,
    tone_sweep,
)

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.sensor.lib import compute, sinks

OFFSETS = (1e6, 2e6)


def zarr_sweep(path, offsets, *, analysis=IQ_ONLY, **sink_kws):
    """return a tone sweep with a `ZarrSink` at `path`"""
    sweep = tone_sweep(offsets, analysis=analysis)
    return sweep.replace(sink=ss.specs.ZarrSink(path=str(path), **sink_kws))


def run_to_store(sweep):
    """return the datasets that the sink returned from each append"""
    with ss.open_resources(sweep, None) as res:
        return [ds for ds in ss.iterate_sweep(res) if ds is not None]


# %% NoSink


def test_no_sink_append_returns_its_argument():
    sink = ss.sinks.NoSink(tone_sweep((1e6,)))
    token = object()
    assert sink.append(token) is token
    assert sink.captures_elapsed == 1


# %% _BatchTracker


def test_batch_tracker_cycles_the_homogeneous_group_sizes():
    """4 identical captures with batches of 2: 2 groups of 2, cycling"""
    tracker = sinks._BatchTracker(tone_captures((1e6,) * 4), min_size=2)
    assert tracker.total_size == 4
    assert tracker.size == 2
    assert tracker.next() == 2
    assert tracker.next() == 2


def test_batch_tracker_merges_captures_of_different_shape():
    """a group closes only once every distinct capture shape is both pending and
    still to come. 2 captures of 1 ms followed by 1 of 2 ms form a single group."""
    captures = tone_captures((1e6, 2e6)) + (
        preset_capture('single_tone', **{**ONE_PORT, 'duration': 2e-3}, snr=None),
    )
    tracker = sinks._BatchTracker(captures, min_size=1)
    assert tracker.total_size == 3
    assert tracker.size == 3


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


def mixed_rate_sweep(path, **sink_kws):
    """2 single-tone captures at 2 sample rates, so the PSD frequency grids differ"""
    captures = (
        preset_capture('single_tone', sample_rate=6.144e6),
        preset_capture('single_tone', sample_rate=3.072e6, analysis_bandwidth=2.5e6),
    )
    sweep = preset_sweep('single_tone', captures)
    return sweep.replace(sink=ss.specs.ZarrSink(path=str(path), **sink_kws))


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
            # the fixed coordinates are written at open; no row of a variable is
            chunk_files = [
                p
                for name in stored.data_vars
                for p in (Path(sweep.sink.path) / str(name)).rglob('*')
                if p.is_file() and not p.name.startswith(('.z', 'zarr.json'))
            ]
            assert chunk_files == []
            # an unwritten store is trimmed to zero rows at close
    assert sa.load(sweep.sink.path).sizes['capture'] == 0


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


def test_zarr_capture_sink_grows_past_the_planned_rows(tmp_path):
    """a looped sweep writes more rows than the plan and the store keeps them all"""
    sweep = zarr_sweep(tmp_path / 'loop.zarr', (1e6, 2e6))
    unsunk = sweep.replace(sink=sweep.sink.replace(path=None))
    taken = []
    with (
        ss.open_resources(unsunk, None) as res,
        ss.sinks.ZarrCaptureSink(sweep) as sink,
    ):
        for dd in ss.iterate_sweep(res, loop=True, sink=sink):
            if dd is None:
                continue
            taken.append(dd)
            if len(taken) == 5:
                break
    stored = sa.load(sweep.sink.path)
    assert stored.sizes['capture'] == 5
    assert stored.capture_index.values.tolist() == [0, 1, 0, 1, 0]


def test_zarr_capture_sink_trims_to_the_rows_written(tmp_path):
    """a sweep stopped after 1 of 3 captures leaves a store of 1 row"""
    sweep = zarr_sweep(tmp_path / 'short.zarr', (1e6, 2e6, 3e6))
    unsunk = sweep.replace(sink=sweep.sink.replace(path=None))
    with (
        ss.open_resources(unsunk, None) as res,
        ss.sinks.ZarrCaptureSink(sweep) as sink,
    ):
        for dd in ss.iterate_sweep(res, sink=sink):
            if dd is not None:
                break
    stored = sa.load(sweep.sink.path)
    assert stored.sizes['capture'] == 1
    assert stored.capture_index.values.tolist() == [0]


def test_zarr_sink_refuses_an_existing_store_without_force(tmp_path):
    sweep = zarr_sweep(tmp_path / 'twice.zarr', (1e6,))
    run_to_store(sweep)
    with pytest.raises(FileExistsError):
        run_to_store(sweep)


def test_zarr_sink_force_replaces_an_existing_store(tmp_path):
    sweep = zarr_sweep(tmp_path / 'twice.zarr', (1e6,))
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
    """2 captures written as 1 batch form 1 row. The spectrogram time axis of that row
    has the frames of both captures back to back on a uniform grid."""
    sweep = zarr_sweep(tmp_path / 'spg.zarr', OFFSETS, analysis=SPECTROGRAM)
    frames, hop_period = spectrogram_frames(sweep.captures[0], SPECTROGRAM.spectrogram)

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
    # concat_time_dim rebuilds the grid from the step of the float32 coordinate
    time = stored.spectrogram_time.values.astype('float64')
    assert_close(np.diff(time), hop_period, rtol=elementwise_rtol(np.float32))
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
