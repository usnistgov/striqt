"""striqt.sensor.lib.sinks: NoSink passthrough, batch sizing, the zarr sinks written
through a real sweep (rows equal the in-memory datasets exactly), the .zarr.zip
archive path, and the Zipper"""

from __future__ import annotations

import zipfile
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from numeric_checks import assert_close, elementwise_rtol
from synthetic_sources import PSD_RESOLUTION, SCALE_ONLY, make_capture, make_sweep

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.sensor.lib import sinks

ONE_PORT = {**SCALE_ONLY, 'port': 0}
IQ_ONLY = ss.specs.BundledAnalysis.from_dict({'iq_waveform': {}})
SPECTROGRAM = ss.specs.BundledAnalysis.from_dict({
    'spectrogram': {'window': 'hann', 'frequency_resolution': PSD_RESOLUTION}
})
ZARR_SINK = ss.specs.Extension()
TIME_APPEND_SINK = ss.specs.Extension(sink='striqt.sensor.sinks.ZarrTimeAppendSink')


def tone_captures(offsets):
    return tuple(
        make_capture('single_tone', **ONE_PORT, frequency_offset=f, snr=None)
        for f in offsets
    )


def zarr_sweep(path, offsets, *, analysis=IQ_ONLY, extensions=ZARR_SINK, **sink_kws):
    """a sweep writing to `path` through the binding's default sink (or `extensions`)"""
    sweep = make_sweep('single_tone', tone_captures(offsets), analysis=analysis)
    sink = ss.specs.Sink(path=str(path), **sink_kws)
    return sweep.replace(sink=sink, extensions=extensions)


def run_to_store(sweep):
    """the datasets that the sink returned from each append"""
    with ss.open_resources(sweep, None) as res:
        return [ds for ds in ss.iterate_sweep(res) if ds is not None]


# %% NoSink


def test_no_sink_append_returns_its_argument(fake_source_id):
    sink = ss.sinks.NoSink(make_sweep('single_tone', tone_captures((1e6,))))
    token = object()
    assert sink.append(token) is token
    assert sink.captures_elapsed == 1


# %% _BatchTracker


def test_batch_tracker_cycles_the_homogeneous_group_sizes():
    """four identical captures with batches of two: two groups of two, cycling"""
    tracker = sinks._BatchTracker(tone_captures((1e6,) * 4), min_size=2)
    assert tracker.total_size == 4
    assert tracker.size == 2
    assert tracker.next() == 2
    assert tracker.next() == 2


def test_batch_tracker_merges_captures_of_different_shape():
    """a group closes only once every distinct capture shape is both pending and
    still to come, so two 1 ms captures followed by a 2 ms one form one group"""
    captures = tone_captures((1e6, 2e6)) + (
        make_capture('single_tone', **{**ONE_PORT, 'duration': 2e-3}, snr=None),
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


def test_time_append_sink_requires_a_spectrogram(fake_source_id):
    sweep = zarr_sweep('unused.zarr', (1e6,), extensions=TIME_APPEND_SINK)
    with pytest.raises(ValueError, match='spectrogram'):
        ss.sinks.ZarrTimeAppendSink(sweep)


def spectrogram_frames(capture, spec) -> tuple[int, float]:
    """(frame count, hop period) of the spectrogram of `capture` from the spec"""
    nfft = round(capture.sample_rate / spec.frequency_resolution)
    hop = nfft - round(Fraction(spec.fractional_overlap) * nfft)
    count = round(capture.duration * capture.sample_rate)
    return (count - nfft) // hop + 1, hop / capture.sample_rate


def test_time_append_sink_concatenates_the_frames(tmp_path):
    """two captures written as one batch land in one row whose spectrogram time
    axis holds both captures' frames back to back on a uniform grid"""
    offsets = (1e6, 2e6)
    sweep = zarr_sweep(
        tmp_path / 'spg.zarr',
        offsets,
        analysis=SPECTROGRAM,
        extensions=TIME_APPEND_SINK,
        batched_write_count=2,
    )
    frames, hop_period = spectrogram_frames(sweep.captures[0], SPECTROGRAM.spectrogram)
    results = run_to_store(sweep)

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


# %% Zipper


def make_tree(root: Path) -> set[str]:
    """a small directory tree under `root`; returns its file paths relative to root"""
    files = {'zarr.json', 'a/c/0.0', 'a/zarr.json', 'b/0'}
    for rel in files:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(rel.encode())
    return files


def test_zipper_from_zarr_archives_every_file(tmp_path):
    store = tmp_path / 'data.zarr'
    files = make_tree(store)
    zipper = sinks.Zipper.from_zarr(store)
    out = zipper.archive()

    assert Path(out) == tmp_path / 'data.zarr.zip'
    with zipfile.ZipFile(out) as zf:
        assert set(zf.namelist()) == files
        assert zf.read('a/c/0.0') == b'a/c/0.0'
    assert not store.exists()


def test_zipper_archive_can_keep_the_directory(tmp_path):
    store = tmp_path / 'data.zarr'
    make_tree(store)
    sinks.Zipper.from_zarr(store).archive(remove=False)
    assert store.is_dir()
    assert (tmp_path / 'data.zarr.zip').exists()


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
