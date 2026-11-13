"""per-port summaries of one analyzed capture.

Each summary is a frozen dataclass with one entry per port of the capture, in
`split_capture_ports` order.
"""

from __future__ import annotations as __

import dataclasses
import math
import numbers
from collections.abc import Sequence
from typing import Any, cast

import striqt.analysis as sa
import striqt.waveform as sw
from striqt.analysis.lib.util import np

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
    """sorted center frequencies (Hz) of the sweep captures at the port that the
    third-order products of the capture overload, empty when `extra_data` lacks
    `if_headroom`"""


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
    determine `adc` and `im3_victims`. `adc` is False when `adc_headroom` is absent.
    The candidate IM3 victims are the center frequencies of `dd.config.sweep_spec`.

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
    """return the sorted victim frequencies (Hz) of the IM3 products of `capture`.

    A sweep capture at the port of `capture` is a victim when `if_headroom` (dB) plus
    2/3 of the gain (dB) of `capture` above that capture is at or below 0.
    """
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
        the description, or None when no port is overloaded
    """
    adc = []
    im3 = []
    for p in ports:
        assert not isinstance(p.capture.center_frequency, tuple)
        if p.adc:
            fc = sw.units.format_quantity(
                p.capture.center_frequency, 'Hz', prefix='M', places=0
            )
            adc.append(f'port {p.capture.port} ({fc})')
        if p.im3_victims:
            freqs = ', '.join(
                sw.units.format_quantity(f, None, prefix='M', places=0)
                for f in p.im3_victims
            )
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

    `sweep.source.calibration` selects the noise figure and
    `sweep.source.master_clock_rate` the calibration row.

    Args:
        sweep: the sweep with the calibration
        capture: the capture of the spectrogram
        result: the cached ``(spectrogram, attrs)`` pair, with the spectrogram in
            linear power (mW) with the port on axis 0 and ``attrs['noise_bandwidth']``
            in Hz
        format_path: the formatter of the ``{field}`` templates in the calibration
            path, or None

    Returns:
        one entry per port with a finite SNR, in `split_capture_ports` order, or None
        when `sweep.source.calibration` is None

    Raises:
        KeyError: if the radio settings of a port are not in the calibration grid
        ValueError: if the center frequency of a port is outside the calibrated range
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
    selects the adjustment.

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


@dataclasses.dataclass(frozen=True)
class FieldValue:
    """the text of one spec field across the ports of a capture."""

    name: str
    """the field name"""
    values: tuple[str, ...]
    """the formatted value at each port, in `split_capture_ports` order"""
    units: str
    """the units of `values` with the shared SI prefix, or ``''`` without units"""


@dataclasses.dataclass(frozen=True)
class FieldGroup:
    """the field values that one struct class declares."""

    owner: str
    """the qualified name ``module.Class`` of the struct class that first declares
    the fields in the MRO"""
    fields: tuple[FieldValue, ...]
    """the fields in declaration order"""


def capture_fields(dd: datasets.DelayedDataset) -> tuple[FieldGroup, ...]:
    """return the text of the fields of `dd.capture` at each port.

    A field is included when its name is a coordinate of the capture. Finite numbers
    with units share one SI prefix per field.

    Args:
        dd: the analyzed capture

    Returns:
        one group per declaring class in base-to-derived order, or an empty tuple
        when `dd.capture` has no ports
    """
    coords = datasets.build_capture_coords(
        dd.capture, dd.extra_coords, dd.config.sweep_spec.loops
    )
    if coords is None:
        return ()

    groups: dict[type, list[FieldValue]] = {}
    for name, owner in sa.specs.helpers.field_owners(type(dd.capture)).items():
        if name not in coords:
            continue
        coord = coords[name]
        units = str(coord.attrs.get('units', ''))
        items = _coordinate_items(coord.values)
        groups.setdefault(owner, []).append(_field_value(name, items, units))
    return _as_groups(groups)


def varying_fields(
    sweep: specs.Sweep[Any, Any, Any, Any],
    source_id: specs.types.SourceID | None,
) -> frozenset[str]:
    """return the names of the capture fields that vary across the plan of `sweep`.

    The plan is `loop_captures` of `sweep` for `source_id`, with each planned
    capture split into its single-port captures. A field varies when its value
    differs between any 2 single-port captures of the plan, so a field set to a
    per-port tuple with unequal entries varies within one capture. Values are
    compared by the rule of `drop_conflicts`: a tuple by position, a dict field
    (`adjust_analysis`) by its items, and None is unequal to every set value. `port`
    is compared between the planned captures as a whole, not between the ports of
    one capture.

    Args:
        sweep: the sweep to expand
        source_id: hex id of the open source, selecting the `adjust_captures` block
            applied on top of ``'defaults'``; None applies ``'defaults'`` alone

    Returns:
        the field names, empty when the sweep plans no captures

    Raises:
        msgspec.ValidationError: if `loop_captures` rejects the loops or the
            adjustments of `sweep`
    """
    captures = specs.sequencing.loop_captures(sweep, source_id)
    if not captures:
        return frozenset()

    per_port = [
        {k: v for k, v in c.to_dict().items() if k != 'port'}
        for capture in captures
        for c in specs.captures.split_capture_ports(capture)
    ]
    names = set().union(*per_port)
    varying = names - set(datasets.drop_conflicts(per_port))

    ports = [{'port': c.port} for c in captures]
    if 'port' not in datasets.drop_conflicts(ports):
        varying.add('port')
    return frozenset(varying)


def acquisition_fields(dd: datasets.DelayedDataset) -> tuple[FieldGroup, ...]:
    """return the text of the fields of `dd.extra_coords` that are set.

    `signal_trigger` and fields equal to None are omitted. A finite number with units
    is formatted with an SI prefix.

    Args:
        dd: the analyzed capture

    Returns:
        one group per declaring class in base-to-derived order, each field with a
        single-element `values`
    """
    info = dd.extra_coords
    attrs = sa.specs.helpers.get_capture_type_attrs(type(info))

    groups: dict[type, list[FieldValue]] = {}
    for name, owner in sa.specs.helpers.field_owners(type(info)).items():
        if name == 'signal_trigger':
            continue
        value = getattr(info, name)
        if value is None:
            continue
        units = str(attrs.get(name, {}).get('units', ''))
        groups.setdefault(owner, []).append(_field_value(name, (value,), units))
    return _as_groups(groups)


def analysis_labels(
    analysis: sa.specs.AnalysisGroup,
    registry: sa.register.AnalysisRegistry = sa.registry,
) -> dict[str, str]:
    """return a title for each enabled field of `analysis`, keyed by field name.

    Args:
        analysis: the analysis group, with a field equal to None omitted
        registry: the registry that names the measurements

    Returns:
        the `long_name` of the `registry` entry named by the field, its
        `standard_name` when the entry has no `long_name`, or the field name in
        title case with spaces for underscores when no entry has that name
    """
    titles = {
        info.name: str(info.attrs.get('long_name', info.attrs['standard_name']))
        for info in registry.values()
    }
    labels: dict[str, str] = {}
    for name in analysis.__struct_fields__:
        if getattr(analysis, name) is None:
            continue
        fallback = name.replace('_', ' ').title()
        labels[name] = titles.get(name, fallback)
    return labels


def _as_groups(groups: dict[type, list[FieldValue]]) -> tuple[FieldGroup, ...]:
    return tuple(
        FieldGroup(f'{cls.__module__}.{cls.__qualname__}', tuple(fields))
        for cls, fields in groups.items()
    )


def _field_value(name: str, values: Sequence[Any], units: str) -> FieldValue:
    """return the `FieldValue` of `values` in `units`.

    One SI prefix is applied to `units` when `units` is set and every value is a
    finite number.
    """
    if units and all(_is_finite_number(v) for v in values):
        prefix = sw.units.si_prefix([float(v) for v in values], units)
        text = tuple(sw.units.format_quantity(v, None, prefix=prefix) for v in values)
        return FieldValue(name, text, f'{prefix.symbol}{units}')
    return FieldValue(name, tuple(_describe(v, units) for v in values), units)


def _coordinate_items(values: np.ndarray) -> list[Any]:
    """return the entries of a per-port coordinate as Python values.

    Datetime and timedelta entries are text.
    """
    if values.dtype.kind in 'mM':
        return [str(v) for v in values.flat]
    return np.ravel(values).tolist()


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _describe(value: Any, units: str) -> str:
    """return `value` as text, via `describe_value` for None, numbers, and tuples."""
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace')
    if value is None or isinstance(value, (bool, numbers.Number, tuple)):
        return sa.labels.describe_value(value, units or None)
    return str(value)
