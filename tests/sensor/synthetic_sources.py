"""factories and oracles for sweeps over the synthetic (function generator) sources.

The synthetic bindings call the `striqt.analysis.testing` generators at the source
sample rate with absolute indices referenced to the corrected capture. Thus, the
corrected output is the same generator evaluated at `capture.sample_rate` from
index 0. The presets keep every acquisition under 5e4 samples per port.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from fractions import Fraction
from typing import NamedTuple

import numpy as np
from sweep_strategies import SOURCE

import striqt.analysis as sa
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
    'noise': testing.circular_awgn,
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

# the 4 signal paths of the conditioning stages. Measured overlaps from SOURCE: 3125/3125,
# 50/50, 512/512 and 4608/4608 at fs_sdr 6.25e6, 6.25e6, 15.36e6 and 7.68e6.
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

# an lo_shift design from the 125 MHz clock: fs_sdr 11.364 MS/s, 18750 samples per
# port; lo_offset 1.6618 MHz (914 bins of 6250/4224-point FFTs) on the default path
# and 1.6618 MHz (3656 bins of 25000/16896-point FFTs) on the oaresample path
LO_SHIFT_CAPTURE = {
    'port': (0, 1),
    'sample_rate': 7.68e6,
    'duration': 1e-3,
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

NO_SINK = ss.specs.ZarrSink(path=None)


def preset_capture(binding: str, **kws):
    """return the capture spec of `binding` built from `RESAMPLE_FILTER` updated with
    `kws`"""
    return BINDINGS[binding].schema.capture(**{**RESAMPLE_FILTER, **kws})


def preset_sweep(
    binding: str,
    captures,
    *,
    analysis=ANALYSIS,
    loops=(),
    source=SOURCE,
    **replace,
):
    """return a sweep spec bound to `binding` with `NO_SINK` unless `replace` gives
    `sink`"""
    cls = BINDINGS[binding].sensor.sweep_spec_cls
    replace.setdefault('sink', NO_SINK)
    return cls(
        source=source,
        captures=tuple(captures),
        loops=tuple(loops),
        analysis=analysis,
        **replace,
    )


def tone_capture(preset: dict, **kws) -> ss.specs.SingleToneCapture:
    """return a `single_tone` capture from `preset` at a 1 MHz offset with no added
    noise, with `kws` overriding both.

    A result attributed to the wrong capture disagrees with the generator oracle of
    that capture.
    """
    fields = {**preset, 'frequency_offset': 1e6, 'snr': None, **kws}
    return preset_capture('single_tone', **fields)


def tone_captures(
    offsets: Iterable[float], **kws
) -> tuple[ss.specs.SingleToneCapture, ...]:
    """return one `ONE_PORT` `tone_capture` per entry of `offsets` (Hz)"""
    return tuple(tone_capture(ONE_PORT, **kws, frequency_offset=f) for f in offsets)


def tone_sweep(offsets, *, analysis=IQ_ONLY, **replace):
    """return a `single_tone` sweep with `NO_SINK` over ``tone_captures(offsets)``"""
    return preset_sweep(
        'single_tone', tone_captures(offsets), analysis=analysis, **replace
    )


def fs_sdr(capture, source=SOURCE) -> float:
    """return the source sample rate (S/s) from which `capture` is resampled"""
    return float(
        ss.lib.compute.design_resampler(capture, source.master_clock_rate)['fs_sdr']
    )


def spectrogram_frames(capture, spec) -> tuple[int, float]:
    """return the frame count and hop period (s) of the spectrogram of `capture` under
    `spec`"""
    nfft = round(capture.sample_rate / spec.frequency_resolution)
    hop = nfft - round(Fraction(spec.fractional_overlap) * nfft)
    count = round(capture.duration * capture.sample_rate)
    return (count - nfft) // hop + 1, hop / capture.sample_rate


def source_count(capture, source=SOURCE) -> int:
    """return the source-rate sample count of `capture.duration`, from the exact
    rational resampler ratio with a half sample rounded up"""
    design = ss.lib.compute.design_resampler(capture, source.master_clock_rate)
    samples_out = round(capture.duration * capture.sample_rate)
    exact = Fraction(samples_out * design['nfft'], design['nfft_out'])
    return math.floor(exact + Fraction(1, 2))


def expected_raw(binding: str, capture, source=SOURCE, *, overlaps=None, xp=np):
    """return the acquisition buffer ``[lead overlap | source_count | tail overlap]`` as
    one generator call at `fs_sdr` starting at index ``-lead``"""
    if overlaps is None:
        overlaps = ss.lib.compute.get_correction_overlaps(capture, source, None)
    fs = fs_sdr(capture, source)
    count = overlaps[0] + source_count(capture, source) + overlaps[1]
    return generator(
        binding, capture, sample_rate=fs, start_index=-overlaps[0], count=count, xp=xp
    )


def build_acquired_iq(x, capture, source=SOURCE, *, info=None):
    """return `x` packaged as the `AcquiredIQ` of `capture` before any correction"""
    return ss.specs.AcquiredIQ(
        x=x,
        capture=capture,
        info=ss.specs.AcquisitionInfo() if info is None else info,
        extra_data={},
        source_spec=source,
        resampler=ss.lib.compute.design_resampler(capture, source.master_clock_rate),
    )


class Stages(NamedTuple):
    """the acquisition of one capture and the same acquisition set up for conditioning"""

    raw: ss.specs.AcquiredIQ
    """``[lead overlap | round(duration*fs_sdr) | tail overlap]`` at `fs_sdr`"""
    corrected: ss.specs.AcquiredIQ
    """`raw` with the trigger and `overwrite_x` attached; `unfiltered`, `filtered`,
    and `aligned` select the conditioned waveforms at `capture.sample_rate`"""
    overlaps: tuple[int, int]
    """lead and tail overlap sample counts at `fs_sdr`"""
    resampler: dict
    """the resampler design of the controller"""


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
    """acquire `capture` from a fresh controller and attach the conditioning.

    Args:
        binding: a key of `BINDINGS` or a bound sensor
        capture: the capture spec to arm
        source: the function source spec
        analysis: the analysis targeted by the controller, or None
        array_backend: replaces that field of `source` when given
        signal_trigger: attached to `corrected`
        overwrite_x: attached to `corrected`
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
    corrected = ss.lib.compute.attach_corrections(
        raw, signal_trigger=signal_trigger, overwrite_x=overwrite_x
    )
    return Stages(raw, corrected, overlaps, resampler)


def unfiltered(iq: sa.AcquiredIQ) -> sw.typing.Array:
    """return the resampled waveform before the analysis filter, trigger-shifted when
    a trigger is set"""
    return iq.preprocess(prefer_filtered=False, prefer_aligned=True)


def filtered(iq: sa.AcquiredIQ) -> sw.typing.Array:
    """return the resampled and low-passed waveform, without the trigger shift"""
    return iq.preprocess(prefer_filtered=True, prefer_aligned=False)


def aligned(iq: sa.AcquiredIQ) -> sw.typing.Array:
    """return the resampled, low-passed, and trigger-shifted waveform"""
    return iq.preprocess(prefer_filtered=True, prefer_aligned=True)


def generator(
    binding: str, capture, *, sample_rate, start_index, count, xp=np, ports=None
):
    """return the `striqt.analysis.testing` waveform of `capture`, shape
    ``(ports, count)``.

    `ports` defaults to the port count of `capture`. The noise-bearing generators
    reproduce the acquisition row for row only at that count.
    """
    if ports is None:
        ports = len(capture.port) if isinstance(capture.port, tuple) else 1
    kws = {name: getattr(capture, name) for name in SIGNAL_KWS[binding]}
    if binding == 'noise':
        kws['power'] = kws.pop('noise_psd') * sample_rate
    kws.update(ports=ports, start_index=start_index, count=count, xp=xp)
    return GENERATORS[binding](None, sample_rate, **kws)


def expected_corrected(binding: str, capture, *, count=None, xp=np):
    """return the corrected waveform of `capture` under the uniform time origin: the
    generator at `capture.sample_rate` from index 0.

    The return is None for the noise binding. The source draws noise at `fs_sdr`, and
    only the noise power survives resampling.
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
    """run `sweep` through `open_resources` and `iterate_sweep` with a `NoSink`.

    Returns:
        one `xarray.Dataset` per capture, truncated to the first `take` when `take`
        is set
    """
    datasets = []
    with ss.open_resources(sweep, None) as resources:
        # nothing is written even when the sink of the sweep names a path
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
    """a stand-in for `striqt.analysis.Trigger` that shifts each port by a fixed lag.

    `lags` is one lag (s) per port; a scalar applies to every port.
    """

    def __init__(self, lags, max_lag=None):
        self.lags = lags
        self._max_lag = max_lag

    def max_lag(self, capture) -> float:
        if self._max_lag is not None:
            return self._max_lag
        return float(np.max(np.abs(self.lags)))

    def __call__(
        self, iq: sw.typing.Array | sa.AcquiredIQ, capture: sa.specs.Capture
    ) -> sw.typing.Array:
        if isinstance(iq, sa.AcquiredIQ):
            x = iq.preprocess(prefer_aligned=False)
        else:
            x = iq
        xp = sw.array_namespace(x)
        lags = xp.asarray(self.lags, dtype='float64')
        return xp.broadcast_to(lags, (x.shape[0],)).copy()
