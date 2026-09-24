"""the waveform pass-through measurement: iq_waveform.

`iq_waveform` slices the corrected IQ between `start_time_sec` and `stop_time_sec`
without transforming it, so the tests cover what the measurement wrapping adds: the
conversion of the time bounds to sample indices (open, closed, past the capture end,
reversed), the `iq_index` coordinate and its dtype, and the registered tolerance,
which passes the caller's input error through to the envelope level with no budget of
its own.
"""

from __future__ import annotations

import numpy as np
import pytest
from analysis_strategies import registered_tolerance

import striqt.analysis as sa
import striqt.waveform as sw
from striqt.analysis import testing

FS = 1e6

DURATION = 1e-4
SIZE = round(DURATION * FS)


def capture(duration=DURATION, sample_rate=FS) -> sa.specs.Capture:
    return sa.specs.Capture(duration=duration, sample_rate=sample_rate)


def tolerance(spec, duration=DURATION, **kws) -> sa.specs.Tolerance:
    """the tolerance registered for `spec` on a capture of `duration`"""
    return registered_tolerance(capture(duration=duration), spec, **kws)


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
        ],
        ids=[
            'unbounded',
            'both_bounds',
            'stop_only',
            'start_only',
            'stop_past_end',
            'stop_at_end',
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

    @pytest.mark.parametrize(
        'stop_time_sec', [None, 3 * DURATION], ids=['start_only', 'both_bounds']
    )
    def test_bounds_past_the_capture_end_are_empty(self, stop_time_sec):
        iq = self.waveform()
        da = sa.measurements.iq_waveform(
            iq, capture(), start_time_sec=2 * DURATION, stop_time_sec=stop_time_sec
        )

        assert da.sizes['iq_index'] == 0
        assert da.coords['iq_index'].size == 0

    def test_reversed_window_is_empty(self):
        """current behaviour, not a validated contract: a window that ends before it
        starts yields an empty result rather than an error"""
        iq = self.waveform()
        da = sa.measurements.iq_waveform(
            iq, capture(), start_time_sec=6e-5, stop_time_sec=2e-5
        )

        assert da.sizes['iq_index'] == 0
        assert da.coords['iq_index'].size == 0

    def test_tolerance_is_the_input_error_on_the_envelope_level(self):
        """the slice adds no error of its own, so exact IQ passes through with a zero
        budget, and an rms amplitude error `r` in the IQ reads on the envelope level
        ``20*log10|iq|`` as an rms of ``20*log10(1 + r)`` dB with a larger peak over
        the slice"""
        exact = tolerance(sa.specs.IQWaveform())
        assert exact == sa.specs.Tolerance(
            units='dB', rtol=0.0, on_peak=sa.specs.ErrorBound(rms=0.0, peak=0.0)
        )

        r = 1e-4
        noisy = tolerance(sa.specs.IQWaveform(), input_error=r)
        assert noisy.units == 'dB'
        assert noisy.rtol == exact.rtol
        assert noisy.on_peak.rms == pytest.approx(
            sw.power_analysis.level_tolerance_dB(r)
        )
        assert noisy.on_peak.peak > noisy.on_peak.rms
        assert noisy.off_peak_dBc is not None and noisy.off_peak_dBc.peak < 0

        shorter = tolerance(sa.specs.IQWaveform(stop_time_sec=1e-5), input_error=r)
        assert shorter.on_peak.rms == noisy.on_peak.rms
        assert shorter.on_peak.peak < noisy.on_peak.peak
