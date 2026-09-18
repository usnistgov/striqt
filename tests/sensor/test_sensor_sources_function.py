"""striqt.sensor.lib.sources.function: the mapping from a capture's ports onto the
`striqt.analysis.testing` generators that the synthetic sources call"""

from __future__ import annotations

import numpy as np

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.sensor.lib import sources

MCR = 125e6
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
        source.get_waveform(COUNT, 0, 0, port=port, xp=np, dtype='complex64')
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
