"""factories and oracles for sweeps over the synthetic (function generator) sources.

Not a conftest: importable by bare name from the sensor test modules only.

The four synthetic bindings call the `striqt.analysis.testing` generators at the
source sample rate `fs_sdr` with absolute sample indices referenced to the corrected
capture, so the corrected output of a capture is the same generator evaluated at
`capture.sample_rate` from index 0 (`expected_corrected`). The presets below were
chosen from a 125 MHz master clock so that every acquisition stays under 5e4 samples
per port and the correction overlaps come out even, which `Controller.read_iq`
requires.
"""

from __future__ import annotations

from fractions import Fraction
from typing import NamedTuple

import numpy as np
from sweep_strategies import SOURCE

import striqt.sensor as ss
import striqt.waveform as sw
from striqt.analysis import testing

BINDINGS = {
    'single_tone': ss.bindings.single_tone,
    'noise': ss.bindings.noise,
    'sawtooth': ss.bindings.sawtooth,
    'dirac_delta': ss.bindings.dirac_delta,
}

GENERATORS = {
    'single_tone': testing.single_tone,
    'noise': testing.noise,
    'sawtooth': testing.sawtooth,
    'dirac_delta': testing.dirac_delta,
}

# capture fields of each binding that are also generator keywords
SIGNAL_KWS = {
    'single_tone': ('frequency_offset', 'snr'),
    'noise': ('noise_psd',),
    'sawtooth': ('period', 'power'),
    'dirac_delta': ('time', 'power'),
}

# the four signal paths of correct_iq. Measured overlaps from SOURCE: 6250/6250,
# 350/350, 512/512 and 12800/12800 at fs_sdr 6.25e6, 6.25e6, 15.36e6 and 7.68e6.
# Both durations are multiples of the 1/8000 s detector period.
RESAMPLE_FILTER = {
    'port': (0, 1),
    'sample_rate': 6.144e6,
    'duration': 2e-3,
    'analysis_bandwidth': 5e6,
}
RESAMPLE_ONLY = {
    'port': (0, 1),
    'sample_rate': 6e6,
    'duration': 2e-3,
    'analysis_bandwidth': float('inf'),
}
SCALE_ONLY = {
    'port': (0, 1),
    'sample_rate': 15.36e6,
    'duration': 1e-3,
    'analysis_bandwidth': float('inf'),
    'host_resample': False,
}
FILTER_ONLY = {
    'port': (0, 1),
    'sample_rate': 7.68e6,
    'duration': 2e-3,
    'analysis_bandwidth': 5e6,
    'host_resample': False,
}
PRESETS = {
    'resample_filter': RESAMPLE_FILTER,
    'resample_only': RESAMPLE_ONLY,
    'scale_only': SCALE_ONLY,
    'filter_only': FILTER_ONLY,
}
ONE_PORT = {**SCALE_ONLY, 'port': 0}

# the smallest lo_shift design from the 125 MHz clock whose overlaps are even:
# fs_sdr 9.615 MS/s, lo_offset 1.661 MHz, 25000 samples per port
LO_SHIFT_CAPTURE = {
    'port': (0, 1),
    'sample_rate': 7.68e6,
    'duration': 0.5e-3,
    'analysis_bandwidth': 3.072e6,
}

FILTER_SIZE = ss.lib.compute.corrections.FILTER_SIZE

# 8 kHz divides all four preset sample rates
PSD_RESOLUTION = 8e3
DETECTOR_PERIOD = Fraction(1, 8000)
ANALYSIS = ss.specs.BundledAnalysis.from_dict({
    'iq_waveform': {},
    'power_spectral_density': {
        'window': 'hann',
        'frequency_resolution': PSD_RESOLUTION,
    },
    'channel_power_time_series': {'detector_period': DETECTOR_PERIOD},
})

IQ_ONLY = ss.specs.BundledAnalysis.from_dict({'iq_waveform': {}})
SPECTROGRAM = ss.specs.BundledAnalysis.from_dict({
    'spectrogram': {'window': 'hann', 'frequency_resolution': PSD_RESOLUTION}
})

NO_SINK = ss.specs.Extension(sink='striqt.sensor.sinks.NoSink')


def make_capture(binding: str, **kws):
    """a capture spec for `binding` from RESAMPLE_FILTER updated with `kws`"""
    return BINDINGS[binding].schema.capture(**{**RESAMPLE_FILTER, **kws})


def make_sweep(
    binding: str,
    captures,
    *,
    analysis=ANALYSIS,
    loops=(),
    source=SOURCE,
    **replace,
):
    """a sweep spec bound to `binding` that writes nothing (NO_SINK)"""
    cls = BINDINGS[binding].sensor.sweep_spec_cls
    return cls(
        source=source,
        captures=tuple(captures),
        loops=tuple(loops),
        analysis=analysis,
        extensions=NO_SINK,
        **replace,
    )


def tone_captures(offsets, **kws):
    """a ONE_PORT single_tone capture per frequency offset, with no added noise so
    that a result attributed to the wrong capture disagrees with its generator oracle"""
    return tuple(
        make_capture(
            'single_tone', **{**ONE_PORT, 'snr': None, **kws}, frequency_offset=f
        )
        for f in offsets
    )


def tone_sweep(offsets, *, analysis=IQ_ONLY, **replace):
    """a NO_SINK sweep over tone_captures(offsets)"""
    return make_sweep(
        'single_tone', tone_captures(offsets), analysis=analysis, **replace
    )


def fs_sdr(capture, source=SOURCE) -> float:
    """the source sample rate that correct_iq resamples `capture` from"""
    return float(
        ss.lib.compute.design_resampler(capture, source.master_clock_rate)['fs_sdr']
    )


def resampler_nffts(capture, source=SOURCE) -> tuple[int, int]:
    """the (input, output) FFT sizes of the resampler design for `capture`"""
    design = ss.lib.compute.design_resampler(capture, source.master_clock_rate)
    return design['nfft'], design['nfft_out']


def spectrogram_frames(capture, spec) -> tuple[int, float]:
    """(frame count, hop period) of the spectrogram of `capture` under `spec`"""
    nfft = round(capture.sample_rate / spec.frequency_resolution)
    hop = nfft - round(Fraction(spec.fractional_overlap) * nfft)
    count = round(capture.duration * capture.sample_rate)
    return (count - nfft) // hop + 1, hop / capture.sample_rate


def expected_raw(binding: str, capture, source=SOURCE, *, overlaps=None, xp=np):
    """the acquisition buffer [lead overlap | round(duration*fs_sdr) | tail overlap] as
    one generator call at fs_sdr starting at index -lead"""
    if overlaps is None:
        overlaps = ss.lib.compute.get_correction_overlaps(capture, source, None)
    fs = fs_sdr(capture, source)
    count = overlaps[0] + round(capture.duration * fs) + overlaps[1]
    return generator(
        binding, capture, sample_rate=fs, start_index=-overlaps[0], count=count, xp=xp
    )


def build_acquired_iq(x, capture, source=SOURCE, *, info=None):
    """`x` packaged as the AcquiredIQ that a controller returns before any correction"""
    return ss.specs.AcquiredIQ(
        pre_align=x,
        pre_filter=None,
        aligned=None,
        capture=capture,
        info=ss.specs.AcquisitionInfo() if info is None else info,
        extra_data={},
        source_spec=source,
        resampler=ss.lib.compute.design_resampler(capture, source.master_clock_rate),
    )


class Stages(NamedTuple):
    """the acquisition and correction of one capture"""

    raw: ss.specs.AcquiredIQ
    corrected: ss.specs.AcquiredIQ
    overlaps: tuple[int, int]
    resampler: dict


def acquire_corrected(
    binding,
    capture,
    *,
    source=SOURCE,
    analysis=None,
    array_backend=None,
    signal_trigger=None,
    overwrite_x=False,
) -> Stages:
    """acquire `capture` from a fresh controller and run it through correct_iq.

    `binding` is a key of BINDINGS or a bound sensor (the file sources are not
    synthetic and are passed directly); `array_backend` replaces that field of
    `source` when given.

    `raw.pre_align` is [lead overlap | round(duration*fs_sdr) | tail overlap] at
    fs_sdr; `corrected` holds the pre_filter, pre_align and (with a trigger) aligned
    stages at capture.sample_rate.
    """
    if isinstance(binding, str):
        binding = BINDINGS[binding]
    if array_backend is not None:
        source = source.replace(array_backend=array_backend)
    with binding.from_source_spec(source, reuse_iq=False) as ctrl:
        ctrl.target_analysis(analysis)
        ctrl._arm_spec(capture)
        raw = ctrl.acquire()
        overlaps = ss.lib.compute.get_correction_overlaps(capture, source, analysis)
        resampler = dict(ctrl.get_resampler())
    corrected = ss.correct_iq(
        raw, signal_trigger=signal_trigger, overwrite_x=overwrite_x
    )
    return Stages(raw, corrected, overlaps, resampler)


def generator(
    binding: str, capture, *, sample_rate, start_index, count, xp=np, ports=None
):
    """the `striqt.analysis.testing` waveform of `capture`, shape (ports, count).

    `ports` defaults to the capture's port count; the noise-bearing generators need
    exactly that to reproduce the acquisition row for row.
    """
    if ports is None:
        ports = len(capture.port) if isinstance(capture.port, tuple) else 1
    kws = {name: getattr(capture, name) for name in SIGNAL_KWS[binding]}
    kws.update(ports=ports, start_index=start_index, count=count, xp=xp)
    return GENERATORS[binding](None, sample_rate, **kws)


def expected_corrected(binding: str, capture, *, count=None, xp=np):
    """the corrected waveform of `capture` under the uniform time origin: the
    generator at capture.sample_rate from index 0.

    None for noise: only its power survives resampling, since the source draws it
    at fs_sdr.
    """
    if binding == 'noise':
        return None
    if count is None:
        count = round(capture.duration * capture.sample_rate)
    return generator(
        binding,
        capture,
        sample_rate=capture.sample_rate,
        start_index=0,
        count=count,
        xp=xp,
    )


def run_in_memory(sweep, *, loop=False, take=None, **replace) -> list:
    """run `sweep` through open_resources and iterate_sweep with NoSink, returning
    one xarray.Dataset per capture (the first `take` with `loop`)"""
    datasets = []
    with ss.open_resources(sweep, None) as resources:
        # NoSink looks up the controller that open_resources registers
        replace.setdefault('sink', ss.sinks.NoSink(sweep))
        results = ss.iterate_sweep(resources, loop=loop, **replace)
        for dd in results:
            if dd is None:
                continue
            datasets.append(ss.lib.compute.from_delayed(dd))
            if take is not None and len(datasets) >= take:
                break
    return datasets


class FakeTrigger:
    """stands in for `striqt.analysis.Trigger`: shifts each port by a fixed lag.

    `lags` is one lag per port in seconds; a scalar applies to every port.
    """

    def __init__(self, lags, max_lag=None):
        self.lags = lags
        self._max_lag = max_lag

    def max_lag(self, capture) -> float:
        if self._max_lag is not None:
            return self._max_lag
        return float(np.max(np.abs(self.lags)))

    def __call__(self, iq, capture):
        xp = sw.array_namespace(iq)
        lags = xp.asarray(self.lags, dtype='float64')
        return xp.broadcast_to(lags, (iq.shape[0],)).copy()
