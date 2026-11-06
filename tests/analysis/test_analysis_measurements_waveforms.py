"""striqt.analysis.measurements.waveforms: the IQ pass-through measurement.

The tests cover what the measurement wrapping adds to a slice of the corrected IQ:
the conversion of the time bounds to sample indices, the index coordinate, and the
registered tolerance. The tolerance passes `input_error` through unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
from analysis_strategies import DURATION, FS, capture, tolerance

import striqt.analysis as sa
import striqt.waveform as sw
from striqt.analysis import testing

SIZE = round(DURATION * FS)


# %% iq_waveform


class TestIqWaveform:
    @staticmethod
    def waveform(duration=DURATION):
        return testing.single_tone(duration, FS, frequency_offset=1e5, snr=10)

    @pytest.mark.parametrize(
        ('start_time_sec', 'stop_time_sec', 'start', 'stop'),
        [
            (None, None, 0, SIZE),
            (2e-5, 6e-5, 20, 60),
            (None, 2.5e-5, 0, 25),
            (9.6e-5, None, 96, SIZE),
            (None, 2 * DURATION, 0, SIZE),
            (None, DURATION, 0, SIZE),
            (2 * DURATION, None, SIZE, SIZE),
            (2 * DURATION, 3 * DURATION, SIZE, SIZE),
        ],
        ids=[
            'unbounded',
            'both_bounds',
            'stop_only',
            'start_only',
            'stop_past_end',
            'stop_at_end',
            'start_past_end',
            'both_past_end',
        ],
    )
    def test_time_bounds_slice_by_sample_index(
        self, start_time_sec, stop_time_sec, start, stop
    ):
        iq = self.waveform()
        da = sa.measurements.iq_waveform(
            iq, capture(), start_time_sec=start_time_sec, stop_time_sec=stop_time_sec
        )

        assert da.sizes['iq_index'] == stop - start
        assert np.array_equal(da.values, iq[:, start:stop])
        indices = da.coords['iq_index']
        assert indices.dtype == np.dtype('uint64')
        np.testing.assert_array_equal(indices.values, np.arange(start, stop))

    def test_reversed_window_is_empty(self):
        """a window that ends before it starts yields an empty result with a 0-size
        index coordinate."""
        iq = self.waveform()
        da = sa.measurements.iq_waveform(
            iq, capture(), start_time_sec=6e-5, stop_time_sec=2e-5
        )

        assert da.sizes['iq_index'] == 0
        assert da.coords['iq_index'].size == 0

    def test_tolerance_is_the_input_error_on_the_envelope_level(self):
        """the slice adds no error, so exact IQ passes through with a zero budget. An
        rms amplitude error `r` in the IQ is an rms error of ``20*log10(1 + r)`` dB on
        the envelope level ``20*log10|iq|``."""
        exact = tolerance(sa.specs.IQWaveform())
        assert exact == sa.specs.Tolerance(
            units='dB', rtol=0.0, on_peak=sa.specs.ErrorBound(rms=0.0, peak=0.0)
        )

        r = 1e-4
        noisy = tolerance(sa.specs.IQWaveform(), input_error=r)
        assert noisy.units == 'dB'
        assert noisy.rtol == 0.0
        assert noisy.on_peak.rms == pytest.approx(
            sw.power_analysis.level_tolerance_dB(r)
        )
