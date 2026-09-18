"""striqt.sensor.lib.sources.base: the read contract of VirtualSource (buffer position
to generator index, contiguity across reads, timestamps, re-arming) and the empty
NoSource"""

from __future__ import annotations

import numpy as np

import striqt.sensor as ss
from striqt.analysis import testing
from striqt.sensor.lib import sources

MCR = 125e6
# served by the firmware rate (host_resample=False), so the generator runs at the
# capture's own sample rate and no design decision enters the expected values
SAMPLE_RATE = 15.36e6
PERIOD = 1e-4
# asymmetric, so that a read positioned by the tail overlap instead of the lead shows
OVERLAPS = (512, 256)
COUNT = 1000
NOISE_PSD = 1e-17

FUNCTION_SOURCE = ss.specs.FunctionSource(master_clock_rate=MCR, num_rx_ports=2)
NO_SOURCE = ss.specs.NoSource(master_clock_rate=MCR, num_rx_ports=2)


def _armed(source_cls, spec, capture_cls, port=(0, 1), **capture_kws):
    capture = capture_cls(
        port=port,
        sample_rate=SAMPLE_RATE,
        duration=1e-3,
        host_resample=False,
        **capture_kws,
    )
    source = source_cls(spec)
    source.setup()
    source.arm(capture)
    source.trigger(OVERLAPS)
    return source, capture


def _sawtooth_source():
    return _armed(
        sources.SawtoothSource, FUNCTION_SOURCE, ss.specs.SawtoothCapture, period=PERIOD
    )


def _buffers(count, ports=2):
    """one NaN-filled row per port, so that an unwritten sample is visible"""
    return [np.full(count, np.nan, dtype='complex64') for _ in range(ports)]


def _expected_sawtooth(start_index, count):
    return testing.sawtooth(
        None, SAMPLE_RATE, period=PERIOD, ports=2, start_index=start_index, count=count
    )


def _sample_period_ns(count):
    return round(count * 1e9 / SAMPLE_RATE)


# %% VirtualSource.read


def test_buffer_position_zero_is_the_lead_overlap_before_index_zero():
    source, _ = _sawtooth_source()
    buffers = _buffers(COUNT)

    count, _ = source.read(buffers, 0, COUNT)

    assert count == COUNT
    assert np.array_equal(np.stack(buffers), _expected_sawtooth(-OVERLAPS[0], COUNT))


def test_consecutive_reads_are_contiguous():
    source, _ = _sawtooth_source()
    buffers = _buffers(COUNT)
    first = 600

    source.read(buffers, 0, first)
    source.read(buffers, first, COUNT - first)

    assert np.array_equal(np.stack(buffers), _expected_sawtooth(-OVERLAPS[0], COUNT))


def test_timestamps_advance_by_the_samples_read():
    source, _ = _sawtooth_source()

    _, first_ns = source.read(_buffers(COUNT), 0, COUNT)
    _, second_ns = source.read(_buffers(24), 0, 24)

    assert second_ns - first_ns == _sample_period_ns(COUNT)


def test_arm_restarts_the_stream():
    source, capture = _sawtooth_source()
    first = _buffers(COUNT)
    source.read(first, 0, COUNT)

    source.arm(capture)
    source.trigger(OVERLAPS)
    again = _buffers(COUNT)
    source.read(again, 0, COUNT)

    assert np.array_equal(np.stack(again), np.stack(first))


def test_rows_follow_the_capture_port_order():
    """buffers[i] is the capture's i-th port, which is port 1 for port=(1, 0)"""
    source, _ = _armed(
        sources.NoiseSource,
        FUNCTION_SOURCE,
        ss.specs.NoiseCapture,
        port=(1, 0),
        noise_psd=NOISE_PSD,
    )
    buffers = _buffers(COUNT)

    source.read(buffers, 0, COUNT)

    expected = testing.noise(
        None,
        SAMPLE_RATE,
        noise_psd=NOISE_PSD,
        ports=2,
        start_index=-OVERLAPS[0],
        count=COUNT,
    )
    assert np.array_equal(np.stack(buffers), expected)
    assert not np.array_equal(buffers[0], buffers[1])


# %% NoSource


def _no_source():
    return _armed(sources.NoSource, NO_SOURCE, ss.specs.SensorCapture)


def test_no_source_read_leaves_the_buffers_alone():
    source, _ = _no_source()
    buffers = _buffers(COUNT)

    count, _ = source.read(buffers, 0, COUNT)

    assert count == COUNT
    assert np.isnan(np.stack(buffers)).all()


def test_no_source_identity():
    source, _ = _no_source()
    assert source.get_id() == 'null'
    assert source.get_info().num_rx_ports == NO_SOURCE.num_rx_ports


def test_no_source_timestamps_advance_by_the_samples_read():
    source, _ = _no_source()

    _, first_ns = source.read(_buffers(COUNT), 0, COUNT)
    _, second_ns = source.read(_buffers(24), 0, 24)

    assert second_ns - first_ns == _sample_period_ns(COUNT)
