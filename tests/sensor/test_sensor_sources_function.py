"""striqt.sensor.lib.sources.function: the mapping from a capture's ports onto the
`striqt.analysis.testing` generators that the synthetic sources call"""

from __future__ import annotations

import numpy as np
import pytest
from numeric_checks import assert_close, tone_frequency
from soapy_factories import MCR
from sweep_strategies import SOURCE
from synthetic_sources import BINDINGS, LO_SHIFT_CAPTURE, generator

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.sensor.lib import sources
from striqt.waveform.lib.arrays import ROUNDOFF_SAFETY, unit_roundoff

SAMPLE_RATE = 15.36e6
COUNT = 4096
NOISE_PSD = 1e-17


def _armed(source_cls, capture_cls, **capture_kws):
    spec = ss.specs.FunctionSource(master_clock_rate=MCR, num_rx_ports=4)
    capture = capture_cls(sample_rate=SAMPLE_RATE, duration=1e-3, **capture_kws)
    source = source_cls(spec)
    source.setup()
    source.arm(capture)
    return source, capture


def _rows(source, ports):
    return [
        source.get_waveform(COUNT, 0, port=port, xp=np, dtype='complex64')
        for port in ports
    ]


def _generator_kws(source, capture):
    return {
        'duration': None,
        'sample_rate': source.get_resampler(capture)['fs_sdr'],
        'start_index': 0,
        'count': COUNT,
    }


# %% NoiseSource


def test_noise_ports_are_independent_and_match_the_generator():
    source, capture = _armed(
        sources.NoiseSource, ss.specs.NoiseCapture, port=(0, 1), noise_psd=NOISE_PSD
    )
    rows = _rows(source, (0, 1))
    expected = sa.testing.noise(
        noise_psd=NOISE_PSD, ports=2, **_generator_kws(source, capture)
    )

    for i, row in enumerate(rows):
        assert row.shape == (1, COUNT)
        assert np.array_equal(row, expected[i : i + 1])

    assert not np.array_equal(rows[0], rows[1])


def test_noise_rows_follow_the_capture_port_order():
    source, capture = _armed(
        sources.NoiseSource, ss.specs.NoiseCapture, port=(3, 2), noise_psd=NOISE_PSD
    )
    expected = sa.testing.noise(
        noise_psd=NOISE_PSD, ports=2, **_generator_kws(source, capture)
    )

    # the port number is not its own index: port 3 is the capture's first row
    for i, row in enumerate(_rows(source, (3, 2))):
        assert np.array_equal(row, expected[i : i + 1])


def test_noise_from_a_scalar_port_capture():
    source, capture = _armed(
        sources.NoiseSource, ss.specs.NoiseCapture, port=1, noise_psd=NOISE_PSD
    )
    expected = sa.testing.noise(
        noise_psd=NOISE_PSD, ports=1, **_generator_kws(source, capture)
    )
    assert np.array_equal(_rows(source, (1,))[0], expected)


# %% SingleToneSource


def test_single_tone_noise_is_independent_per_port():
    kws = {'frequency_offset': -3e6, 'snr': 20}
    source, capture = _armed(
        sources.SingleToneSource, ss.specs.SingleToneCapture, port=(0, 1), **kws
    )
    rows = _rows(source, (0, 1))
    expected = sa.testing.single_tone(
        lo_offset=source.get_resampler(capture)['lo_offset'],
        ports=2,
        **kws,
        **_generator_kws(source, capture),
    )

    for i, row in enumerate(rows):
        assert np.array_equal(row, expected[i : i + 1])

    assert not np.array_equal(rows[0], rows[1])


# %% get_waveform: the origin contract for every synthetic binding

# capture fields that exercise each generator, at a firmware-served 15.36 MS/s so
# that the source rate is the capture rate
SIGNALS = {
    'single_tone': {'frequency_offset': 1e6, 'snr': 30},
    'noise': {'noise_psd': NOISE_PSD},
    'sawtooth': {'period': 1e-4},
    'dirac_delta': {'time': 1e-4, 'power': -3},
}


def _armed_binding(binding, **capture_kws):
    capture = BINDINGS[binding].schema.capture(**{
        'port': (0, 1),
        'sample_rate': SAMPLE_RATE,
        'duration': 1e-3,
        'host_resample': False,
        **SIGNALS[binding],
        **capture_kws,
    })
    source = BINDINGS[binding].sensor.source_cls(SOURCE)
    source.setup()
    source.arm(capture)
    return source, capture


@pytest.mark.parametrize('binding', list(BINDINGS))
@pytest.mark.parametrize(
    'start_index', [-700, 0, 300], ids=['preroll', 'origin', 'interior']
)
def test_get_waveform_is_the_generator_at_the_absolute_index(binding, start_index):
    source, capture = _armed_binding(binding)
    expected = generator(
        binding, capture, sample_rate=SAMPLE_RATE, start_index=start_index, count=COUNT
    )

    for port in (0, 1):
        row = source.get_waveform(COUNT, start_index, port=port, xp=np)
        assert np.array_equal(row, expected[port : port + 1])


def test_host_resampled_capture_generates_at_the_source_rate():
    """the source runs at the radio's rate, the smallest integer division of the
    master clock at or above the capture rate: 125 MHz / 20 for 6 MS/s"""
    source, _ = _armed_binding(
        'single_tone', sample_rate=6e6, host_resample=True, snr=None
    )
    fs_sdr = MCR / (MCR // 6e6)

    row = source.get_waveform(COUNT, 0, port=0, xp=np)

    expected = sa.testing.tone(None, fs_sdr, frequency=1e6, count=COUNT)
    assert np.array_equal(row, expected)


def test_sawtooth_closed_form_on_a_binary_grid():
    """(t % period) / period at a power-of-two rate and period is the exact
    fraction (i mod 2**14) / 2**14, so the source must reproduce it bit for bit"""
    rate = 2.0**24
    spec = ss.specs.FunctionSource(master_clock_rate=rate, num_rx_ports=1)
    capture = ss.specs.SawtoothCapture(
        port=0,
        sample_rate=rate,
        duration=2.0**-10,
        host_resample=False,
        period=2.0**-10,
    )
    source = sources.SawtoothSource(spec)
    source.setup()
    source.arm(capture)
    start_index = -700

    row = source.get_waveform(COUNT, start_index, port=0, xp=np)

    ramp = np.arange(start_index, start_index + COUNT) % 2**14
    expected = (ramp / 2**14).astype('complex64')[np.newaxis]
    assert np.array_equal(row, expected)


def test_single_tone_sits_at_the_lo_shifted_frequency():
    """a lo_shift design moves the radio LO by lo_offset, so what the radio sees at
    fs_sdr is the tone at frequency_offset + lo_offset"""
    frequency_offset = -1e6
    source, capture = _armed_binding(
        'single_tone',
        **LO_SHIFT_CAPTURE,
        lo_shift='right',
        host_resample=True,
        frequency_offset=frequency_offset,
        snr=None,
    )
    design = source.get_resampler(capture)
    fs_sdr = design['fs_sdr']
    count = 25000

    row = source.get_waveform(count, 0, port=0, xp=np)

    frequency = frequency_offset + design['lo_offset']
    assert abs(tone_frequency(row[0], fs_sdr) - frequency) < fs_sdr / count
    # the source multiplies two complex64 tones, each within one unit roundoff, and
    # the product rounds twice more
    expected = sa.testing.tone(None, fs_sdr, frequency=frequency, count=count)
    assert_close(row, expected, atol=ROUNDOFF_SAFETY * 4 * unit_roundoff(np.float32))


def test_dirac_delta_index_is_not_compensated():
    """the impulse of a pre-roll read sits at round(time*fs_sdr) - start_index"""
    source, capture = _armed_binding('dirac_delta')
    start_index = -700

    row = source.get_waveform(COUNT, start_index, port=0, xp=np)

    expected = np.zeros((1, COUNT), dtype='complex64')
    expected[0, round(capture.time * SAMPLE_RATE) - start_index] = 10 ** (
        capture.power / 20
    )
    assert np.array_equal(row, expected)
