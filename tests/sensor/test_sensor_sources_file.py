"""striqt.sensor.lib.sources.file: the zarr, MAT and TDMS file sources.

All three share the uniform time origin of `VirtualSource.get_waveform`: file index 0
is corrected output sample 0, so the lead overlap that `correct_iq` trims is a zero
pre-roll rather than file data. The oracle for each round trip is the array that was
written to the file, read back independently of the source under test.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
from numeric_checks import (
    FIR_LEAKAGE,
    ROUNDOFF_SAFETY,
    assert_close,
    interior,
    to_numpy,
    unit_roundoff,
)
from synthetic_sources import (
    FILTER_SIZE,
    SCALE_ONLY,
    acquire_corrected,
    make_capture,
    make_sweep,
    run_in_memory,
)

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.analysis import testing
from striqt.sensor.lib.sources.file import _split_preroll

FS = SCALE_ONLY['sample_rate']
CENTER_FREQUENCY = 3.7e9
START_TIME = np.datetime64('2026-01-01T00:00:00', 'ns')
# the file holds 2 ms; the captures read 0.5 ms so that the tail overlap of a
# filtered capture (16640 samples at FS) still lies inside the file
FILE_DURATION = 2e-3
FILE_SAMPLES = round(FILE_DURATION * FS)
READ_DURATION = 0.5e-3
READ_SAMPLES = round(READ_DURATION * FS)
TONE_FREQUENCIES = (-1e6, 2e6)

READ_SCALE_ONLY = {**SCALE_ONLY, 'duration': READ_DURATION}
READ_FILTER_ONLY = {**READ_SCALE_ONLY, 'analysis_bandwidth': 5e6}

PREROLL_COUNT = 12
PREROLL_FILLS = pytest.mark.parametrize(
    'fill', [0, 5, PREROLL_COUNT], ids=['none', 'straddling', 'all']
)


def file_capture(**kws):
    return ss.specs.FileCapture(**{**READ_SCALE_ONLY, **kws})


def assert_preroll_then_file(waveform, fill, file_rows, **tol):
    """`waveform` is `fill` zeros followed by the start of `file_rows`"""
    waveform = to_numpy(waveform)
    assert not waveform[..., :fill].any(), 'pre-roll is not zero'
    assert_close(
        waveform[..., fill:], file_rows[..., : waveform.shape[-1] - fill], **tol
    )


# %% _split_preroll


@pytest.mark.parametrize(
    'start_index, count, expected',
    [
        (-5, 3, (3, 0, 0)),
        (-5, 8, (5, 0, 3)),
        (-8, 8, (8, 0, 0)),
        (0, 8, (0, 0, 8)),
        (7, 8, (0, 7, 8)),
    ],
    ids=['fully_negative', 'straddling', 'ends_at_zero', 'from_zero', 'positive'],
)
def test_split_preroll(start_index, count, expected):
    """the fill is the number of requested indices below zero, the file read
    starts at the first non-negative one and supplies the rest"""
    assert _split_preroll(start_index, count) == expected


# %% ZarrIQSource


@pytest.fixture(scope='module')
def zarr_iq_file(tmp_path_factory):
    """a zarr store holding the corrected iq_waveform of two 2-port single-tone
    captures at FS, with the coordinates ZarrIQSource.arm reads from a sensor store"""
    import xarray as xr

    fields = {**SCALE_ONLY, 'duration': FILE_DURATION, 'snr': None}
    captures = [
        make_capture('single_tone', **fields, frequency_offset=f)
        for f in TONE_FREQUENCIES
    ]
    ds = xr.concat(run_in_memory(make_sweep('single_tone', captures)), dim='capture')
    rows = ds.sizes['capture']
    ds = ds.assign_coords(
        center_frequency=('capture', np.full(rows, CENTER_FREQUENCY)),
        gain=('capture', np.zeros(rows)),
        start_time=('capture', np.full(rows, START_TIME)),
    )
    path = tmp_path_factory.mktemp('zarr_iq') / 'iq.zarr'
    sa.dump(sa.open_store(path, mode='w'), ds)
    stored = sa.load(path).iq_waveform.values
    assert stored.shape == (2 * len(captures), FILE_SAMPLES)
    return SimpleNamespace(path=path, stored=stored, captures=captures)


def zarr_spec(zarr_iq_file, **select):
    return ss.specs.ZarrIQSource(
        path=str(zarr_iq_file.path),
        center_frequency=CENTER_FREQUENCY,
        master_clock_rate=FS,
        select=select,
    )


def zarr_rows(zarr_iq_file, capture_index):
    """the stored rows of one capture, in port order"""
    return zarr_iq_file.stored[2 * capture_index : 2 * capture_index + 2]


@PREROLL_FILLS
def test_zarr_get_waveform_preroll(zarr_iq_file, xp, fill):
    source = ss.lib.sources.ZarrIQSource(zarr_spec(zarr_iq_file, capture_index=1))
    waveform = source.get_waveform(PREROLL_COUNT, -fill, port=1, xp=xp)
    assert waveform.shape == (1, PREROLL_COUNT)
    assert_preroll_then_file(waveform[0], fill, zarr_rows(zarr_iq_file, 1)[1])


def test_zarr_select_narrows_rows(zarr_iq_file):
    everything = ss.lib.sources.ZarrIQSource(zarr_spec(zarr_iq_file))
    assert everything.get_info().num_rx_ports == zarr_iq_file.stored.shape[0]

    second = ss.lib.sources.ZarrIQSource(zarr_spec(zarr_iq_file, capture_index=1))
    assert second.get_info().num_rx_ports == 2
    waveform = second.get_waveform(16, 0, port=0, xp=np)
    assert_close(waveform[0], zarr_rows(zarr_iq_file, 1)[0, :16])


@pytest.mark.parametrize(
    'port, ok', [(1, True), (2, False), (5, False)], ids=['last', 'one_past', 'far']
)
def test_zarr_port_bound(zarr_iq_file, port, ok):
    source = ss.lib.sources.ZarrIQSource(zarr_spec(zarr_iq_file, capture_index=0))
    if ok:
        source.get_waveform(4, 0, port=port, xp=np)
    else:
        with pytest.raises(ValueError, match='exceeds data channel count'):
            source.get_waveform(4, 0, port=port, xp=np)


def test_zarr_request_past_end(zarr_iq_file):
    source = ss.lib.sources.ZarrIQSource(zarr_spec(zarr_iq_file, capture_index=0))
    source.get_waveform(8, FILE_SAMPLES - 8, port=0, xp=np)
    with pytest.raises(ValueError, match='file capture length'):
        source.get_waveform(8, FILE_SAMPLES - 7, port=0, xp=np)


@pytest.mark.parametrize(
    'dtype, expected', [(None, 'complex64'), ('complex128', 'complex128')], ids=str
)
def test_zarr_get_waveform_dtype(zarr_iq_file, dtype, expected):
    source = ss.lib.sources.ZarrIQSource(zarr_spec(zarr_iq_file, capture_index=0))
    waveform = source.get_waveform(8, 0, port=0, xp=np, dtype=dtype)
    assert waveform.dtype == np.dtype(expected)
    assert_close(waveform[0], zarr_rows(zarr_iq_file, 0)[0, :8])


def test_zarr_read_timestamps(zarr_iq_file):
    """the first chunk is stamped with the store's start_time; later chunks return 0
    (current behaviour, not a contract: Controller.read_iq only keeps the first)"""
    source = ss.lib.sources.ZarrIQSource(zarr_spec(zarr_iq_file, capture_index=0))
    source.setup()
    source.arm(file_capture())
    source.trigger((0, 0))
    bufs = [np.empty(16, dtype='complex64') for _ in range(2)]

    count, first_ns = source.read(bufs, 0, 8)
    assert count == 8
    assert first_ns == int(START_TIME.astype('int64'))
    _, later_ns = source.read(bufs, 8, 8)
    assert later_ns == 0


def test_zarr_acquisition_info(zarr_iq_file):
    """arm fills the acquisition metadata from the store's coordinates"""
    source = zarr_spec(zarr_iq_file, capture_index=1)
    raw = acquire_corrected(ss.bindings.zarr_iq, file_capture(), source=source).raw
    assert raw.info.center_frequency == CENTER_FREQUENCY
    assert raw.info.backend_sample_rate == FS
    assert raw.info.port == (0, 1)
    assert raw.info.gain == (0.0, 0.0)


def test_zarr_acquire_layout(zarr_iq_file):
    """the raw acquisition is [zero pre-roll | file from index 0 | file tail]"""
    spec = zarr_spec(zarr_iq_file, capture_index=1)
    capture = file_capture()
    lead, tail = ss.lib.compute.get_correction_overlaps(capture, spec, None)
    raw = acquire_corrected(ss.bindings.zarr_iq, capture, source=spec).raw
    assert raw.pre_align.shape == (2, lead + READ_SAMPLES + tail)
    assert_preroll_then_file(raw.pre_align, lead, zarr_rows(zarr_iq_file, 1))


@pytest.mark.parametrize('capture_index', [0, 1], ids=['first', 'second'])
def test_zarr_round_trip_scale_only(zarr_iq_file, capture_index):
    """without resampling or filtering the corrected capture is the stored waveform
    from index 0, bit for bit"""
    source = zarr_spec(zarr_iq_file, capture_index=capture_index)
    stages = acquire_corrected(ss.bindings.zarr_iq, file_capture(), source=source)
    corrected = stages.corrected
    expected = zarr_rows(zarr_iq_file, capture_index)[:, :READ_SAMPLES]
    assert corrected.pre_align.dtype == expected.dtype
    np.testing.assert_array_equal(to_numpy(corrected.pre_align), expected)


def test_zarr_round_trip_filtered(zarr_iq_file):
    """the FIR sees the zero pre-roll as a step at index 0, so its ringing spans the
    first FILTER_SIZE//2 output samples; the interior is the stored tone (in the
    passband) to within the FIR's passband ripple"""
    source = zarr_spec(zarr_iq_file, capture_index=1)
    capture = file_capture(**READ_FILTER_ONLY)
    corrected = acquire_corrected(ss.bindings.zarr_iq, capture, source=source).corrected
    expected = zarr_rows(zarr_iq_file, 1)[:, :READ_SAMPLES]
    assert corrected.pre_align.shape == expected.shape
    assert_close(
        interior(corrected.pre_align, FILTER_SIZE // 2),
        interior(expected, FILTER_SIZE // 2),
        sigma=FIR_LEAKAGE,
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason=(
        'ZarrIQSource.get_resampler passes the file sample rate as master_clock_rate '
        'without pinning backend_sample_rate, so design_resampler is free to choose '
        'an integer divisor of the file rate as fs_sdr that the file cannot supply'
    ),
)
def test_zarr_resampler_pins_file_rate(zarr_iq_file):
    """a capture at half the file rate must be resampled from the file rate"""
    spec = zarr_spec(zarr_iq_file, capture_index=0)
    capture = ss.specs.FileCapture(
        port=(0, 1),
        sample_rate=FS / 2,
        duration=READ_DURATION,
        analysis_bandwidth=float('inf'),
    )
    with ss.bindings.zarr_iq.from_source_spec(spec) as ctrl:
        ctrl._arm_spec(capture)
        design = ctrl.get_resampler()
    assert design['fs_sdr'] == FS
    assert design['nfft'] == 2 * design['nfft_out']


# %% MATSource


@pytest.fixture(scope='module')
def mat_files(tmp_path_factory):
    """legacy .mat files holding a 'waveform' matrix of one and of two tone rows"""
    from scipy.io import savemat

    root = tmp_path_factory.mktemp('mat')
    files = {}
    for rows in (1, 2):
        tone = {'frequency_offset': TONE_FREQUENCIES[0], 'count': FILE_SAMPLES}
        waveform = testing.single_tone(None, FS, ports=rows, **tone)
        path = root / f'{rows}row.mat'
        savemat(path, {'waveform': waveform})
        files[rows] = SimpleNamespace(path=path, waveform=waveform)
    return files


def mat_spec(mat_file, **kws):
    return ss.specs.MATSource(path=str(mat_file.path), master_clock_rate=FS, **kws)


@pytest.mark.parametrize('key', [None, 'waveform'], ids=['default_key', 'explicit_key'])
@PREROLL_FILLS
def test_mat_get_waveform_preroll(mat_files, array_backend, key, fill):
    """the stream's 'waveform' default applies when the spec leaves `key` unset"""
    spec = mat_spec(mat_files[1], key=key, array_backend=array_backend)
    source = ss.lib.sources.MATSource(spec)
    try:
        waveform = source.get_waveform(PREROLL_COUNT, -fill, port=0, xp=np)
    finally:
        source.close()
    assert waveform.shape == (1, PREROLL_COUNT)
    assert_preroll_then_file(waveform, fill, mat_files[1].waveform)


@pytest.mark.xfail(
    strict=True,
    reason=(
        'MATLegacyFileStream.read re-lists the whole matrix as a fresh ref on every '
        'call, so a request past the end wraps to file index 0 regardless of loop'
    ),
)
def test_mat_request_past_end(mat_files):
    """without loop, a read past the end of the file is an error, as for the zarr
    and TDMS sources"""
    source = ss.lib.sources.MATSource(mat_spec(mat_files[1]))
    try:
        source.get_waveform(8, FILE_SAMPLES - 8, port=0, xp=np)
        with pytest.raises(ValueError, match='too few samples'):
            source.get_waveform(8, FILE_SAMPLES - 7, port=0, xp=np)
    finally:
        source.close()


def test_mat_missing_file(tmp_path):
    spec = ss.specs.MATSource(path=str(tmp_path / 'absent.mat'), master_clock_rate=FS)
    with pytest.raises(IOError, match='does not exist'):
        ss.lib.sources.MATSource(spec)


def test_mat_round_trip_scale_only(mat_files):
    capture = file_capture(port=0)
    source = mat_spec(mat_files[1])
    stages = acquire_corrected(ss.bindings.mat_file, capture, source=source)
    assert stages.raw.info.backend_sample_rate == FS
    np.testing.assert_array_equal(
        to_numpy(stages.corrected.pre_align), mat_files[1].waveform[:, :READ_SAMPLES]
    )


def test_mat_loop_repeats_file(mat_files):
    """with loop=True a capture longer than the file wraps around to index 0"""
    duration = 1.5 * FILE_DURATION
    capture = file_capture(port=0, duration=duration)
    stages = acquire_corrected(
        ss.bindings.mat_file, capture, source=mat_spec(mat_files[1], loop=True)
    )
    corrected = stages.corrected
    file = mat_files[1].waveform
    count = round(duration * FS)
    expected = np.concatenate([file, file[:, : count - FILE_SAMPLES]], axis=1)
    np.testing.assert_array_equal(to_numpy(corrected.pre_align), expected)


@pytest.mark.xfail(
    strict=True,
    raises=ValueError,
    reason=(
        'MATSource.get_waveform returns every row of the file regardless of `port`, '
        'so VirtualSource.read cannot assign the (rows, count) result into one '
        'port buffer'
    ),
)
def test_mat_two_port(mat_files):
    """a two-row file supplies ports 0 and 1 from rows 0 and 1"""
    capture = file_capture()
    stages = acquire_corrected(
        ss.bindings.mat_file, capture, source=mat_spec(mat_files[2])
    )
    expected = mat_files[2].waveform[:, :READ_SAMPLES]
    np.testing.assert_array_equal(to_numpy(stages.corrected.pre_align), expected)


# %% TDMSSource

TDMS_REFERENCE_LEVEL_DBM = 10.0
INT16_FULL_SCALE = np.iinfo(np.int16).max


@pytest.fixture(scope='module')
def tdms_file(tmp_path_factory):
    """a TDMS file in the header + interleaved int16 I/Q layout TDMSSource reads,
    quantized from a tone: full scale int16 is the reference level in dBm"""
    from nptdms import ChannelObject, GroupObject, TdmsWriter

    tone = testing.single_tone(
        None, FS, frequency_offset=TONE_FREQUENCIES[0], ports=1, count=FILE_SAMPLES
    )[0]
    amplitude = 10 ** (TDMS_REFERENCE_LEVEL_DBM / 20)
    i = np.round(tone.real / amplitude * INT16_FULL_SCALE).astype('int16')
    q = np.round(tone.imag / amplitude * INT16_FULL_SCALE).astype('int16')

    path = tmp_path_factory.mktemp('tdms') / 'iq.tdms'
    reference_level = np.array([TDMS_REFERENCE_LEVEL_DBM])
    with TdmsWriter(str(path)) as writer:
        writer.write_segment([
            GroupObject('header'),
            ChannelObject('header', 'IQ_samples_per_second', np.array([FS])),
            ChannelObject('header', 'carrier_frequency', np.array([CENTER_FREQUENCY])),
            ChannelObject('header', 'total_samples', np.array([FILE_SAMPLES])),
            ChannelObject('header', 'reference_level_dBm', reference_level),
            GroupObject('iq'),
            ChannelObject('iq', 'I', i),
            ChannelObject('iq', 'Q', q),
        ])

    waveform = (i.astype('float64') + 1j * q) / INT16_FULL_SCALE * amplitude
    return SimpleNamespace(path=path, waveform=waveform)


def tdms_spec(tdms_file, **kws):
    kws.setdefault('master_clock_rate', FS)
    return ss.specs.TDMSSource(path=str(tdms_file.path), **kws)


# int16 -> float32 is exact; the scale and the product each round once
TDMS_SCALING_TOL = {'rtol': ROUNDOFF_SAFETY * 2 * unit_roundoff(np.float32)}


def test_tdms_file_info(tdms_file):
    source = ss.lib.sources.TDMSSource(tdms_spec(tdms_file))
    assert source._file_info.backend_sample_rate == FS
    assert source._file_info.center_frequency == CENTER_FREQUENCY
    assert source.get_info().num_rx_ports == 1


@PREROLL_FILLS
def test_tdms_get_waveform_preroll(tdms_file, xp, fill):
    source = ss.lib.sources.TDMSSource(tdms_spec(tdms_file))
    waveform = source.get_waveform(PREROLL_COUNT, -fill, port=0, xp=xp)
    assert waveform.shape == (PREROLL_COUNT,)
    assert waveform.dtype == np.dtype('complex64')
    assert_preroll_then_file(waveform, fill, tdms_file.waveform, **TDMS_SCALING_TOL)


def test_tdms_request_past_end(tdms_file):
    source = ss.lib.sources.TDMSSource(tdms_spec(tdms_file))
    source.get_waveform(8, FILE_SAMPLES - 8, port=0, xp=np)
    with pytest.raises(ValueError, match='file capture length'):
        source.get_waveform(8, FILE_SAMPLES - 7, port=0, xp=np)


def test_tdms_resampler_uses_header_rate(tdms_file):
    """the file, not the spec's master clock, fixes the resampler's input rate"""
    capture = ss.specs.FileCapture(
        port=0, sample_rate=FS / 2, duration=READ_DURATION, analysis_bandwidth=math.inf
    )
    with ss.bindings.tdms_file.from_source_spec(
        tdms_spec(tdms_file, master_clock_rate=2 * FS)
    ) as ctrl:
        ctrl._arm_spec(capture)
        design = ctrl.get_resampler()
    assert design['fs_sdr'] == FS
    assert design['nfft'] == 2 * design['nfft_out']


def test_tdms_round_trip_scale_only(tdms_file):
    capture = file_capture(port=0)
    source = tdms_spec(tdms_file)
    stages = acquire_corrected(ss.bindings.tdms_file, capture, source=source)
    assert stages.raw.info.center_frequency == CENTER_FREQUENCY
    assert_close(
        stages.corrected.pre_align[0],
        tdms_file.waveform[:READ_SAMPLES],
        **TDMS_SCALING_TOL,
    )
