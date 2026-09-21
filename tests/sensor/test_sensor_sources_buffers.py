"""striqt.sensor.lib.sources.buffers: read-count arithmetic, trigger holdoff
alignment, transport dtype casting, IQ reuse compatibility, gapless carryover and
receive buffer allocation"""

from __future__ import annotations

from math import ceil
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from numeric_checks import assert_close
from soapy_factories import soapy_capture, source_spec
from sweep_strategies import SOURCE
from synthetic_sources import (
    PRESETS,
    RESAMPLE_FILTER,
    SCALE_ONLY,
    fs_sdr,
    make_capture,
)

import striqt.sensor as ss
import striqt.waveform as sw
from striqt.sensor.lib.sources import buffers

# exact in binary, so the holdoff sample counts derived here match to the last bit
HOLDOFF_TIME = 2**-10
STROBE_TIME = 2**-12


def stub_controller(capture, source=SOURCE):
    """the controller attributes that ReceiveBuffers and _alloc_empty_iq read"""
    return SimpleNamespace(
        capture_spec=capture,
        source_spec=source,
        source_info=ss.specs.SourceInfo(num_rx_ports=None),
    )


# %% get_read_count


@pytest.mark.parametrize('preset', list(PRESETS), ids=list(PRESETS))
@pytest.mark.parametrize('overlap', [0, 2, 100], ids=lambda n: f'overlap{n}')
def test_get_read_count_is_the_source_rate_sample_count_plus_overlap(preset, overlap):
    """the source must deliver duration * fs_sdr samples (rounded up when the
    resampler ratio is fractional) plus the requested overlap"""
    capture = make_capture('single_tone', **PRESETS[preset])
    samples_out = round(capture.duration * capture.sample_rate)
    ratio = fs_sdr(capture) / capture.sample_rate
    expected = ceil(samples_out * ratio) + overlap

    assert buffers.get_read_count(capture, SOURCE, overlap=overlap) == expected


def test_get_read_count_holdoff_adds_transient_and_two_strobe_periods():
    """the holdoff padding covers the transient plus two strobe periods at the
    source rate; without include_holdoff neither setting contributes"""
    source = source_spec(
        transient_holdoff_time=HOLDOFF_TIME, trigger_strobe=STROBE_TIME
    )
    capture = soapy_capture()
    base = round(capture.duration * capture.sample_rate)
    padding = ceil(fs_sdr(capture, source) * (HOLDOFF_TIME + 2 * STROBE_TIME))

    assert (
        buffers.get_read_count(capture, source, include_holdoff=True) == base + padding
    )
    assert buffers.get_read_count(capture, source, include_holdoff=False) == base


@pytest.mark.parametrize('overlap', [1, 3, -2], ids=['odd1', 'odd3', 'negative'])
def test_get_read_count_rejects_invalid_overlap(overlap):
    capture = make_capture('single_tone', **SCALE_ONLY)
    with pytest.raises(ValueError, match='non-negative even'):
        buffers.get_read_count(capture, SOURCE, overlap=overlap)


# %% find_trigger_holdoff


@given(
    start_time_ns=st.integers(min_value=0, max_value=10**13),
    strobe_us=st.integers(min_value=1, max_value=200),
    start_overlap=st.integers(min_value=0, max_value=5000).map(lambda k: 2 * k),
    rearmed=st.booleans(),
)
def test_find_trigger_holdoff_aligns_to_the_strobe(
    start_time_ns, strobe_us, start_overlap, rearmed
):
    """the holdoff ends on a strobe edge (to within one sample), covers the
    overlap and, on a fresh arm, the transient holdoff.

    The upper bound records current behaviour rather than a contract: the
    implementation may skip up to one extra strobe period beyond the first edge
    that satisfies the minimum.
    """
    strobe = strobe_us * 1e-6
    source = source_spec(transient_holdoff_time=HOLDOFF_TIME, trigger_strobe=strobe)
    capture = soapy_capture()
    fs = fs_sdr(capture, source)
    state = SimpleNamespace(start_time_ns=0 if rearmed else None)

    holdoff = buffers.find_trigger_holdoff(
        source, capture, state, start_time_ns, start_overlap=start_overlap
    )

    min_holdoff = start_overlap + (0 if rearmed else round(HOLDOFF_TIME * fs))
    strobe_ns = strobe_us * 1000
    assert holdoff >= min_holdoff
    assert holdoff < min_holdoff + 2 * round(strobe * fs)
    end_ns = start_time_ns + round(holdoff * 1e9 / fs)
    excess_ns = end_ns % strobe_ns
    assert min(excess_ns, strobe_ns - excess_ns) <= 1e9 / fs


@pytest.mark.parametrize('trigger_strobe', [None, 0], ids=['none', 'zero'])
def test_find_trigger_holdoff_without_a_strobe_is_overlap_plus_transient(
    trigger_strobe,
):
    source = source_spec(
        transient_holdoff_time=HOLDOFF_TIME, trigger_strobe=trigger_strobe
    )
    capture = soapy_capture()
    transient = round(HOLDOFF_TIME * fs_sdr(capture, source))

    fresh = SimpleNamespace(start_time_ns=None)
    holdoff = buffers.find_trigger_holdoff(source, capture, fresh, 12345, 40)
    assert holdoff == 40 + transient
    rearmed = SimpleNamespace(start_time_ns=12345)
    assert buffers.find_trigger_holdoff(source, capture, rearmed, 12345, 40) == 40


# %% cast_iq


def test_cast_iq_complex64_returns_an_equal_copy():
    buffer = (np.arange(24, dtype='float32') * (1 - 2j)).reshape(2, 12)
    buffer = buffer.astype('complex64')

    out = buffers.cast_iq(SOURCE, buffer, buffer.shape[1])

    assert_close(out, buffer)
    assert not np.shares_memory(out, buffer)


def test_cast_iq_int16_interleaved_pairs_become_unscaled_complex64():
    """transport int16 holds (I, Q) pairs; the cast is exact and leaves the
    full-scale factor to voltage_scale"""
    source = source_spec(transport_dtype='int16')
    ports, acquired = 2, 300
    pairs = (np.arange(ports * 2 * acquired) * 7919 - 32768) % 65536 - 32768
    pairs = pairs.reshape(ports, 2 * acquired).astype('int16')
    pairs[0, :2] = (-32768, 32767)

    # the receive buffer is complex64-sized with the int16 stream at its head
    buffer = np.zeros((ports, 2 * acquired), dtype='complex64')
    buffer.view('int16')[:, : 2 * acquired] = pairs

    out = buffers.cast_iq(source, buffer, acquired)

    expected = pairs[:, 0::2].astype('float32') + 1j * pairs[:, 1::2].astype('float32')
    assert out.dtype == np.complex64
    assert out.shape == (ports, acquired)
    np.testing.assert_array_equal(out, expected.astype('complex64'))


@pytest.mark.namespaces('cupy')
def test_cast_iq_cupy_returns_a_device_array(xp):
    source = SOURCE.replace(array_backend='cupy')
    buffer = (np.arange(24, dtype='float32') * (1 - 2j)).reshape(2, 12)
    buffer = buffer.astype('complex64')

    out = buffers.cast_iq(source, buffer, buffer.shape[1])

    assert sw.is_cupy_array(out)
    assert_close(out, buffer)


# %% get_dtype_scale


@pytest.mark.parametrize(
    'transport_dtype, expected',
    [('int16', 1 / (2**15 - 1)), ('float32', 1.0), ('complex64', 1.0)],
    ids=['int16', 'float32', 'complex64'],
)
def test_get_dtype_scale(transport_dtype, expected):
    assert buffers.get_dtype_scale(transport_dtype) == expected


# %% is_reusable

_C1 = make_capture('single_tone', **RESAMPLE_FILTER)
# host_resample changes fs_sdr unless the rate divides the master clock
_C_DIVISOR = make_capture(
    'single_tone', **{**SCALE_ONLY, 'sample_rate': 12.5e6, 'duration': 1e-3}
)

REUSE_CASES = {
    'identical': (_C1, _C1, True),
    'analysis_bandwidth': (_C1, _C1.replace(analysis_bandwidth=float('inf')), True),
    'sample_rate_same_fs_sdr': (
        _C1,
        _C1.replace(sample_rate=6e6, analysis_bandwidth=float('inf')),
        True,
    ),
    'adjust_analysis': (
        _C1,
        _C1.replace(adjust_analysis={'power_spectral_density': {'window': 'hann'}}),
        True,
    ),
    'host_resample_same_fs_sdr': (
        _C_DIVISOR,
        _C_DIVISOR.replace(host_resample=True),
        True,
    ),
    'host_resample_new_fs_sdr': (
        make_capture('single_tone', **SCALE_ONLY),
        make_capture('single_tone', **{**SCALE_ONLY, 'host_resample': True}),
        False,
    ),
    'port': (_C1, _C1.replace(port=0), False),
    'frequency_offset': (_C1, _C1.replace(frequency_offset=1e5), False),
    'duration': (_C1, _C1.replace(duration=1e-3), False),
    'none_first': (None, _C1, False),
    'none_second': (_C1, None, False),
}


@pytest.mark.parametrize('case', list(REUSE_CASES), ids=list(REUSE_CASES))
def test_is_reusable(case):
    c1, c2, expected = REUSE_CASES[case]
    if expected and c1 is not None:
        assert fs_sdr(c1) == fs_sdr(c2)
    assert buffers.is_reusable(c1, c2, SOURCE.master_clock_rate) is expected


# %% ReceiveBuffers carryover


def test_carryover_round_trip_copies_the_tail_into_the_next_head(receive_buffers):
    capture = soapy_capture(duration=2e-3)
    rb = receive_buffers(capture)
    unused = 7
    t0 = 1_000_000_000_123
    previous = (np.arange(2 * 40) * (1 + 2j)).reshape(2, 40).astype('complex64')
    rb.stash_carryover(previous, t0, unused_sample_count=unused, capture=capture)

    samples = np.full((2, 40), -1 - 1j, dtype='complex64')
    start_ns, count = rb.apply(samples)

    assert count == unused
    assert start_ns == t0 + round(1e9 * capture.duration)
    np.testing.assert_array_equal(samples[:, :unused], previous[:, -unused:])
    np.testing.assert_array_equal(samples[:, unused:], -1 - 1j)


def test_carryover_is_a_copy_of_the_stashed_samples(receive_buffers):
    capture = soapy_capture()
    rb = receive_buffers(capture)
    previous = np.ones((1, 20), dtype='complex64')
    rb.stash_carryover(previous, 0, unused_sample_count=5, capture=capture)
    previous[:] = 0

    samples = np.zeros((1, 20), dtype='complex64')
    rb.apply(samples)
    assert (samples[:, :5] == 1).all()


def test_carryover_without_a_stash_returns_the_timestamp_only(receive_buffers):
    rb = receive_buffers(soapy_capture())
    samples = np.zeros((1, 8), dtype='complex64')
    assert rb.apply(samples) == (None, 0)
    rb.start_time_ns = 55
    assert rb.apply(samples) == (55, 0)


def test_carryover_is_disabled_when_not_gapless(receive_buffers):
    capture = soapy_capture()
    rb = receive_buffers(capture, source_spec())
    previous = np.ones((1, 20), dtype='complex64')
    rb.stash_carryover(previous, 0, unused_sample_count=5, capture=capture)

    samples = np.zeros((1, 20), dtype='complex64')
    assert rb.carryover_samples is None
    assert rb.apply(samples) == (None, 0)
    assert not samples.any()


def test_carryover_without_a_timestamp_is_an_error(receive_buffers):
    rb = receive_buffers(soapy_capture())
    rb.carryover_samples = np.ones((1, 4), dtype='complex64')
    rb.start_time_ns = None
    with pytest.raises(ValueError, match='timestamp'):
        rb.apply(np.zeros((1, 8), dtype='complex64'))


def test_clear_drops_the_carryover(receive_buffers):
    capture = soapy_capture()
    rb = receive_buffers(capture)
    rb.stash_carryover(
        np.ones((1, 20), dtype='complex64'), 0, unused_sample_count=5, capture=capture
    )
    rb.clear()
    assert rb.carryover_samples is None and rb.start_time_ns is None


# %% _alloc_empty_iq

_TWO_PORT = make_capture('sawtooth', **SCALE_ONLY)
_ONE_PORT = _TWO_PORT.replace(port=0)
_THREE_PORT = _TWO_PORT.replace(port=(0, 1, 2))
_LONG = _TWO_PORT.replace(duration=2e-3)
_ONE_PORT_LONG = _ONE_PORT.replace(duration=2e-3)


def _port_count(capture) -> int:
    return len(capture.port) if isinstance(capture.port, tuple) else 1


def _read_count(capture):
    return buffers.get_read_count(capture, SOURCE, include_holdoff=True)


def test_alloc_shape_is_ports_by_read_count_with_one_stream_buffer_per_port():
    capture = _TWO_PORT
    all_samples, (samples, streams) = buffers._alloc_empty_iq(
        stub_controller(capture), capture
    )
    count = _read_count(capture)
    assert all_samples is samples
    assert samples.shape == (2, count)
    assert samples.dtype == np.complex64
    assert len(streams) == 2
    for i, stream in enumerate(streams):
        assert np.shares_memory(stream, samples[i])


def test_alloc_overlaps_extend_the_read_count():
    capture = _TWO_PORT
    _, (samples, _) = buffers._alloc_empty_iq(
        stub_controller(capture), capture, overlaps=(100, 200)
    )
    assert samples.shape[1] == _read_count(capture) + 300


def test_alloc_reuses_a_prior_buffer_that_is_large_enough():
    ctrl = stub_controller(_TWO_PORT)
    prior = np.empty((2, _read_count(_LONG)), dtype='complex64')
    all_samples, (samples, _) = buffers._alloc_empty_iq(ctrl, _TWO_PORT, prior)
    assert all_samples is prior
    assert np.shares_memory(samples, prior)
    assert samples.shape[0] == 2
    assert samples.shape[1] >= _read_count(_TWO_PORT)


@pytest.mark.parametrize(
    'prior_capture, capture',
    [
        (_TWO_PORT, _LONG),
        (_TWO_PORT, _THREE_PORT),
        (_TWO_PORT, _ONE_PORT),
        (_TWO_PORT, _ONE_PORT_LONG),
    ],
    ids=[
        'too_few_samples',
        'too_few_ports',
        'too_many_ports',
        'fewer_ports_more_samples',
    ],
)
def test_alloc_does_not_reuse_a_prior_buffer_of_the_wrong_shape(prior_capture, capture):
    """a reused buffer is returned whole along the port axis, so it must have
    exactly one row per port and at least the read count of columns"""
    prior = np.empty((2, _read_count(prior_capture)), dtype='complex64')
    ports = _port_count(capture)
    _, (samples, streams) = buffers._alloc_empty_iq(
        stub_controller(capture), capture, prior
    )
    assert samples.shape[0] == ports
    assert samples.shape[1] >= _read_count(capture)
    assert len(streams) == ports
    assert not np.shares_memory(samples, prior)


@pytest.mark.namespaces('cupy')
def test_alloc_for_cupy_is_pinned_host_memory(xp):
    source = SOURCE.replace(array_backend='cupy')
    capture = _TWO_PORT
    all_samples, _ = buffers._alloc_empty_iq(stub_controller(capture, source), capture)
    assert isinstance(all_samples, np.ndarray)
    assert isinstance(all_samples.base, xp.cuda.PinnedMemoryPointer)


# %% get_array_namespace


def test_get_array_namespace_numpy():
    assert buffers.get_array_namespace('numpy') is np


@pytest.mark.namespaces('cupy')
def test_get_array_namespace_cupy(xp):
    assert buffers.get_array_namespace('cupy') is xp


def test_get_array_namespace_cupy_unavailable_is_an_import_error(monkeypatch):
    monkeypatch.setattr(sw.arrays, 'cp', None)
    with pytest.raises(ImportError):
        buffers.get_array_namespace('cupy')


def test_get_array_namespace_rejects_other_backends():
    with pytest.raises(TypeError):
        buffers.get_array_namespace('dask')
