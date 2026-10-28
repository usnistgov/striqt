"""per-port summaries of one analyzed capture: overload state and spectrogram SNR."""

from __future__ import annotations as __

import dataclasses
import math
from collections.abc import Sequence
from typing import Any, cast

import striqt.waveform as sw

from ... import specs
from . import cal_lookups, datasets


@dataclasses.dataclass(frozen=True)
class PortOverload:
    """the overload state of one port of a capture."""

    capture: specs.SoapyCapture
    """the single-port capture at the port"""
    adc: bool
    """True when the ADC headroom at the port is at or below 0 dB"""
    im3_victims: tuple[float, ...]
    """sorted center frequencies (Hz) of the sweep captures at the port where the
    third-order products of this capture exceed the IF headroom. Empty when
    `if_headroom` is absent from `extra_data`."""


@dataclasses.dataclass(frozen=True)
class SpectrogramSNR:
    """the peak SNR of the spectrogram of one port."""

    capture: specs.SensorCapture
    """the single-port capture"""
    snr_dB: float
    """peak bin power of the spectrogram (dB) above the calibrated system noise power
    ``k*T*B*F`` in the noise bandwidth of the spectrogram"""


def overloads(dd: datasets.DelayedDataset) -> tuple[PortOverload, ...]:
    """return the overload state of each port of `dd.capture`.

    `adc_headroom` and `if_headroom` (dB, one value per port) in `dd.extra_data`
    determine `adc` and `im3_victims`. The candidate IM3 victims are the center
    frequencies of `dd.config.sweep_spec`.

    Args:
        dd: the analyzed capture

    Returns:
        one entry per port in `split_capture_ports` order, or an empty tuple when
        `dd.capture` is not a `SoapyCapture` or neither headroom key is present
    """
    capture = dd.capture
    extra = dd.extra_data
    if not isinstance(capture, specs.SoapyCapture):
        return ()
    if 'adc_headroom' not in extra and 'if_headroom' not in extra:
        return ()

    ports = specs.captures.split_capture_ports(capture)
    adc_headroom = extra.get('adc_headroom', (math.inf,) * len(ports))
    if_headroom = extra.get('if_headroom')

    entries = []
    for i, c in enumerate(ports):
        victims: tuple[float, ...] = ()
        if if_headroom is not None:
            victims = _im3_victims(dd.config.sweep_spec, c, float(if_headroom[i]))
        entries.append(
            PortOverload(capture=c, adc=bool(adc_headroom[i] <= 0), im3_victims=victims)
        )
    return tuple(entries)


def _im3_victims(
    sweep: specs.Sweep[Any, Any, Any, Any],
    capture: specs.SoapyCapture,
    if_headroom: float,
) -> tuple[float, ...]:
    """return the sorted center frequencies (Hz) at the port of `capture` where a sweep
    capture has gain high enough for the third-order products of `capture` to exceed
    `if_headroom` (dB)"""
    sweep_captures = cast('tuple[specs.SoapyCapture, ...]', sweep.captures)
    gains = specs.sequencing.max_by_frequency('gain', sweep_captures, sweep.loops)
    assert not isinstance(capture.gain, tuple)

    # third-order products grow 2 dB per dB of gain difference relative to the signal
    victims = [
        cast(float, fc)
        for fc, gain in gains[capture.port].items()
        if if_headroom + 2 / 3 * (capture.gain - gain) <= 0
    ]
    return tuple(sorted(victims))


def describe_overloads(ports: Sequence[PortOverload]) -> str | None:
    """return a one-line description of the overloaded ports in `ports`.

    Returns:
        the description, or None when no port overloaded
    """
    adc = []
    im3 = []
    for p in ports:
        assert not isinstance(p.capture.center_frequency, tuple)
        if p.adc:
            adc.append(
                f'port {p.capture.port} ({p.capture.center_frequency / 1e6:0.0f} MHz)'
            )
        if p.im3_victims:
            freqs = ', '.join(f'{f / 1e6:0.0f}' for f in p.im3_victims)
            im3.append(f'port {p.capture.port} (onto {freqs} MHz)')

    parts = []
    if adc:
        parts.append('adc overload on ' + ', '.join(adc))
    if im3:
        parts.append('if overload at ' + ' and '.join(im3))
    if not parts:
        return None
    return ', '.join(parts)


def spectrogram_snr(
    sweep: specs.Sweep[Any, Any, Any, Any],
    capture: specs.SensorCapture,
    result: Any,
    *,
    format_path: specs.paths.PathFormatter | None = None,
) -> tuple[SpectrogramSNR, ...] | None:
    """return the peak SNR of each port in the spectrogram `result`.

    The spectrogram is linear power with the port on axis 0, and
    ``attrs['noise_bandwidth']`` is in Hz. `sweep.source.calibration` selects the
    noise figure and `sweep.source.master_clock_rate` the calibration row.

    Args:
        sweep: the sweep with the calibration
        capture: the capture of the spectrogram
        result: the cached ``(spectrogram, attrs)`` pair
        format_path: the formatter of the ``{field}`` templates in the calibration
            path, or None

    Returns:
        one entry per port with a finite SNR, in `split_capture_ports` order, or None
        when `sweep.source.calibration` is None
    """
    cal = sweep.source.calibration
    if cal is None:
        return None

    spg, attrs = result
    xp = sw.array_namespace(spg)
    peaks = spg.max(axis=tuple(range(1, spg.ndim)))
    noise = cal_lookups.lookup_system_noise_power(
        cal,
        specs.SoapyCapture.from_spec(capture),
        master_clock_rate=sweep.source.master_clock_rate,
        format_path=format_path,
        B=attrs['noise_bandwidth'],
        xp=xp,
    )

    entries = []
    ports = specs.captures.split_capture_ports(capture)
    for c, snr in zip(ports, sw.powtodB(peaks) - noise):
        if sw.is_cupy_array(snr.data):
            value = float(snr.data.get())
        else:
            value = float(snr.values)
        if not abs(value) < math.inf:
            continue
        entries.append(SpectrogramSNR(capture=c, snr_dB=value))
    return tuple(entries)


def describe_spectrogram_snr(
    entry: SpectrogramSNR,
    *,
    source_id: specs.types.SourceID | None,
    adjust_spec: specs.structs.AdjustCapturesType | None = None,
) -> str:
    """return the one-line description of `entry`.

    A field set by an adjustment in `adjust_spec` is described by the field that
    drives the adjustment.

    Args:
        source_id: the source id that selects the entries of `adjust_spec`
        adjust_spec: the `adjust_captures` of the sweep
    """
    if isinstance(entry.capture, specs.SoapyCapture):
        fields = ('center_frequency', 'port', 'gain')
    else:
        fields = ('port',)

    snr_desc = f'{round(entry.snr_dB)} dB max SNR'
    capture_desc = specs.sequencing.describe_capture(
        entry.capture, fields=fields, source_id=source_id, adjust_spec=adjust_spec
    )
    return f'spectrogram ▮ {snr_desc:<14} ▮ {capture_desc}'
