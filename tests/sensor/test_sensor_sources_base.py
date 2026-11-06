"""striqt.sensor.lib.sources.base: the read contract of the virtual sources"""

from __future__ import annotations

import numpy as np
import pytest
from soapy_factories import MCR
from sweep_strategies import SOURCE as FUNCTION_SOURCE

import striqt.sensor as ss
from striqt.analysis import testing
from striqt.sensor.lib import sources

# served by the firmware rate (host_resample=False), so the generator runs at the
# sample rate of the capture and no design decision enters the expected values
SAMPLE_RATE = 15.36e6
PERIOD = 1e-4
# asymmetric, so that a read positioned by the tail overlap instead of the lead shows
OVERLAPS = (512, 256)
COUNT = 1000

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


def _no_source():
    return _armed(sources.NoSource, NO_SOURCE, ss.specs.SensorCapture)


def _buffers(count, ports=2):
    """return one NaN-filled complex64 row per port, in which an unwritten sample stays
    NaN"""
    return [np.full(count, np.nan, dtype='complex64') for _ in range(ports)]


def _expected_sawtooth(start_index, count):
    return testing.sawtooth(
        None, SAMPLE_RATE, period=PERIOD, ports=2, start_index=start_index, count=count
    )


def _sample_period_ns(count, rate=SAMPLE_RATE):
    return round(count * 1e9 / rate)


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


@pytest.mark.parametrize(
    'armed', [_sawtooth_source, _no_source], ids=['sawtooth', 'no_source']
)
def test_timestamps_advance_by_the_samples_read(armed):
    source, _ = armed()

    _, first_ns = source.read(_buffers(COUNT), 0, COUNT)
    _, second_ns = source.read(_buffers(24), 0, 24)

    assert second_ns - first_ns == _sample_period_ns(COUNT)


def test_first_read_is_stamped_at_the_lead_overlap_before_the_time_origin():
    """buffer position 0 is source sample ``-OVERLAPS[0]``, and the first read is
    stamped that many sample periods before the time origin of the source"""
    source, _ = _sawtooth_source()

    _, first_ns = source.read(_buffers(COUNT), 0, COUNT)

    assert first_ns - source._sync_time_ns == -_sample_period_ns(OVERLAPS[0])


def test_arm_restarts_the_stream():
    source, capture = _sawtooth_source()
    first = _buffers(COUNT)
    source.read(first, 0, COUNT)

    source.arm(capture)
    source.trigger(OVERLAPS)
    again = _buffers(COUNT)
    source.read(again, 0, COUNT)

    assert np.array_equal(np.stack(again), np.stack(first))


@pytest.mark.parametrize('resetup', [False, True], ids=['arm', 'setup_arm'])
@pytest.mark.parametrize(
    'rearm_rate', [SAMPLE_RATE, SAMPLE_RATE / 2], ids=['same_rate', 'half_rate']
)
def test_rearm_continues_the_time_base_at_the_previous_rate(rearm_rate, resetup):
    """the re-armed stream is stamped where the previous stream ended, with the samples
    read converted at the rate of the previous capture. The time origin of the
    re-armed capture is one lead overlap at the new rate after that stamp."""
    source, capture = _sawtooth_source()
    _, first_ns = source.read(_buffers(COUNT), 0, COUNT)

    if resetup:
        source.setup()
    source.arm(capture.replace(sample_rate=rearm_rate))
    source.trigger(OVERLAPS)
    _, rearmed_ns = source.read(_buffers(COUNT), 0, COUNT)

    assert rearmed_ns - first_ns == _sample_period_ns(COUNT)
    assert source._sync_time_ns - rearmed_ns == _sample_period_ns(
        OVERLAPS[0], rate=rearm_rate
    )


def test_reads_after_one_arm_are_consecutive_segments():
    """each read at buffer position 0 after one `arm` continues the generator where
    the previous read ended and is stamped that many sample periods later"""
    source, _ = _sawtooth_source()
    stamps = []

    for k in range(3):
        buffers = _buffers(COUNT)
        _, time_ns = source.read(buffers, 0, COUNT)
        stamps.append(time_ns)
        expected = _expected_sawtooth(k * COUNT - OVERLAPS[0], COUNT)
        assert np.array_equal(np.stack(buffers), expected), f'read {k}'

    assert np.diff(stamps).tolist() == [_sample_period_ns(COUNT)] * 2


# %% NoSource


def test_no_source_read_leaves_the_buffers_alone():
    source, _ = _no_source()
    buffers = _buffers(COUNT)

    count, _ = source.read(buffers, 0, COUNT)

    assert count == COUNT
    assert np.isnan(np.stack(buffers)).all()


def test_no_source_power_units_are_arbitrary():
    source, _ = _no_source()
    assert source.power_units() == 'a.u.'
