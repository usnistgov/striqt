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
    spectrogram_frames,
    tone_captures,
    tone_sweep,
)

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.sensor.lib import sinks

OFFSETS = (1e6, 2e6)


def zarr_sweep(path, offsets, *, analysis=IQ_ONLY, **sink_kws):
    """a tone sweep whose `ZarrSink` writes to `path`"""
    sweep = tone_sweep(offsets, analysis=analysis)
    return sweep.replace(sink=ss.specs.ZarrSink(path=str(path), **sink_kws))


def run_to_store(sweep):
    """the datasets that the sink returned from each append"""
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
    still to come, so 2 captures of 1 ms followed by 1 of 2 ms form a single group"""
    captures = tone_captures((1e6, 2e6)) + (
        preset_capture('single_tone', **{**ONE_PORT, 'duration': 2e-3}, snr=None),
    )
    tracker = sinks._BatchTracker(captures, min_size=1)
    assert tracker.total_size == 3
    assert tracker.size == 3


# %% ZarrCaptureSink


def test_zarr_capture_sink_batches_rows(tmp_path, monkeypatch):
    dumps = []
    dump = sa.dump

    def spy(store, data, **kws):
        dumps.append(data.sizes['capture'])
        return dump(store, data, **kws)

    monkeypatch.setattr(sa, 'dump', spy)
    offsets = (1e6, 2e6, 3e6, 4e6)
    sweep = zarr_sweep(tmp_path / 'out.zarr', offsets, batched_write_count=2)
    results = run_to_store(sweep)

    import xarray as xr

    assert dumps == [2, 2]
    stored = sa.load(sweep.sink.path)
    assert stored.capture_index.values.tolist() == list(range(len(offsets)))
    assert stored.iq_waveform.dtype == np.complex64
    expected = xr.concat(results, 'capture').iq_waveform.values
    assert np.array_equal(stored.iq_waveform.values, expected)


def test_zarr_zip_path_is_archived_and_readable(tmp_path):
    path = tmp_path / 'out.zarr.zip'
    sweep = zarr_sweep(path, (1e6, 2e6))
    run_to_store(sweep)

    assert zipfile.is_zipfile(path)
    assert not path.with_suffix('').exists()
    assert ss.read_zarr_spec(path) == sweep


# %% ZarrTimeAppendSink


def test_time_append_sink_requires_a_spectrogram():
    sweep = zarr_sweep('unused.zarr', (1e6,))
    with pytest.raises(ValueError, match='spectrogram'):
        ss.sinks.ZarrTimeAppendSink(sweep)


def test_time_append_sink_concatenates_the_frames(tmp_path):
    """2 captures written as 1 batch form 1 row whose spectrogram time axis
    contains both captures' frames back to back on a uniform grid"""
    sweep = zarr_sweep(
        tmp_path / 'spg.zarr', OFFSETS, analysis=SPECTROGRAM, batched_write_count=2
    )
    frames, hop_period = spectrogram_frames(sweep.captures[0], SPECTROGRAM.spectrogram)

    # the runner's ZarrCaptureSink would open the same store if the sweep it opens
    # kept the path. A zarr sink looks up the source id in its constructor, which
    # requires the source to be open first
    unsunk = sweep.replace(sink=sweep.sink.replace(path=None))
    with (
        ss.open_resources(unsunk, None) as res,
        ss.sinks.ZarrTimeAppendSink(sweep) as sink,
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
    """a small directory tree under `root`; returns its file paths relative to root"""
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
