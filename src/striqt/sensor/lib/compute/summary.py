"""summaries of the acquisition and analysis of one capture for a front end to present.

Nothing here logs. Each summary function returns a dataclass or a tuple of them, and
the matching `describe_*` function renders one as the line that ``sensor-sweep`` logs.
"""

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
    """the overload state of one port of a capture.

    `capture` is the single-port capture of the port. `adc` is True when the ADC
    headroom at the port is at or below 0 dB. `im3_victims` are the sorted center
    frequencies (Hz) of the sweep captures at the port whose gain puts the third-order
    products of this capture above the IF headroom, and are empty when the acquisition
    reports no IF headroom.
    """

    capture: specs.SoapyCapture
    adc: bool
    im3_victims: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class SpectrogramSNR:
    """the peak SNR (dB) of the spectrogram of one port.

    `snr_dB` is the largest spectrogram bin of `capture` above the calibrated system
    noise power ``k*T*B*F`` in the noise bandwidth of the spectrogram.
    """

    capture: specs.SensorCapture
    snr_dB: float


def overloads(dd: datasets.DelayedDataset) -> tuple[PortOverload, ...]:
    """return the overload state of each port of `dd.capture`.

    `adc_headroom` and `if_headroom` (dB, one value per port) are read from
    `dd.extra_data`. The IM3 victims of a port are found among the captures and loops
    of `dd.config.sweep_spec`.

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
    """return the sorted center frequencies of the sweep captures at the port of
    `capture` whose gain brings the third-order products of `capture` within
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
    """return the overload line for `ports`, or None when no port overloaded."""
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
    """return the peak SNR of each port of a spectrogram `result`.

    `result` is the ``(spectrogram, attrs)`` pair that the spectrogram cache stores:
    linear power with the port on axis 0, and ``attrs['noise_bandwidth']`` (Hz). The
    noise power of each port is `cal_lookups.lookup_system_noise_power` from
    `sweep.source.calibration` for `capture`.

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
    """return the SNR line for `entry`: the rounded SNR and the fields of the capture.

    The fields are `center_frequency`, `port`, and `gain` for a `SoapyCapture`, and
    `port` otherwise, described through `describe_capture` with `source_id` and
    `adjust_spec`.
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
