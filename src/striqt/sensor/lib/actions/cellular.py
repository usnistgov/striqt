"""the action that runs a `Cellular5GNRSweep` as a `Sweep`.

The `Sweep` is built once per spec and memoized by `build_cellular_sweep`.
"""

from __future__ import annotations as __

import typing
from collections.abc import Callable, Collection, Generator, Mapping
from fractions import Fraction
from typing import Any, Generic

import msgspec

import striqt.analysis as sa
import striqt.waveform as sw
from striqt.analysis.lib.util import xr
from striqt.analysis.specs.helpers import SpecValidationError, spec_subset

from ... import specs
from .. import compute
from ..resources import Resources
from ..sinks import NoSink, SinkBase, ZarrCaptureSink
from ..typing import PC, PS, SA, SC, SP, SS
from . import core
from .registry import registry

# the OFDM symbol grid of a normal cyclic prefix slot: 14 symbols per slot; at SCS/2
# resolution the STFT windows overlap by 13/28 and fill 15/28 of each hop
SYMBOL_FRACTIONAL_OVERLAP = Fraction(13, 28)
SYMBOL_WINDOW_FILL = Fraction(15, 28)
FRAME_PERIOD = 10e-3
SYMBOLS_PER_SLOT = 14
PSS_SYNC_TRIGGER = 'cellular_5g_pss_sync'


def symbol_period(subcarrier_spacing: float) -> Fraction:
    """return the duration (in s) of one OFDM symbol at `subcarrier_spacing` (Hz).

    The symbol has the normal cyclic prefix, 14 symbols per slot (3GPP TS 38.211
    §4.3.2).
    """
    slots_per_frame = round(10 * subcarrier_spacing / 15e3)
    return Fraction(1, 100 * slots_per_frame * SYMBOLS_PER_SLOT)


def ssb_symbol_indexes(channel: specs.CellChannel) -> Any:
    """return the SSB case of `channel` for the `symbol_indexes` analysis parameter.

    `ssb_case` is returned when set. Otherwise the case is ``'a'`` at 15 kHz and
    ``'c'`` at 30 kHz (3GPP TS 38.213 §4.1), and ``'auto'`` at any other spacing.
    """
    if channel.ssb_case is not None:
        return channel.ssb_case
    return {15e3: 'a', 30e3: 'c'}.get(channel.subcarrier_spacing, 'auto')


def match_channel(
    capture: specs.SensorCapture,
    cell_channels: Mapping[str, specs.CellChannel],
    index: int,
) -> specs.CellChannel:
    """return the channel of `cell_channels` at the `center_frequency` of `capture`.

    Args:
        capture: a capture with a `center_frequency` field, scalar or one value per
            port
        cell_channels: the `cell_channels` table of the spec
        index: the position of `capture` in `spec.captures`, for the error location

    Raises:
        SpecValidationError: if `capture` has no `center_frequency`, if the ports
            differ in center frequency, or if no channel is at that frequency
    """
    at = (f'.captures[{index}]', '.center_frequency')
    center_frequency = getattr(capture, 'center_frequency', None)
    if center_frequency is None:
        raise SpecValidationError('Expected a center frequency', at)
    frequencies = set(specs.captures.ensure_tuple(center_frequency))
    if len(frequencies) != 1:
        raise SpecValidationError(
            'Expected one center frequency across the ports of a cellular capture', at
        )
    (frequency,) = frequencies
    for channel in cell_channels.values():
        if channel.center_frequency == frequency:
            return channel
    raise SpecValidationError(
        f'Expected the center frequency of a channel in `cell_channels`, got '
        f'{frequency!r}',
        at,
    )


def _field_defaults(cls: type[msgspec.Struct]) -> dict[str, Any]:
    return {f.name: f.default for f in msgspec.structs.fields(cls)}


def resolve_capture(
    capture: SC,
    channel: specs.CellChannel,
    index: int,
    consumed_keys: Collection[str] | None = None,
) -> SC:
    """return `capture` with the radio and analysis settings of `channel`.

    `sample_rate` is `nr_sample_rate` of the channel, and `analysis_bandwidth` is the
    channel bandwidth. The `frequency_offset`, `guard_bandwidths`, `frame_slots`,
    `special_symbols`, `subcarrier_spacing`, and `symbol_indexes` of the channel are
    written into `adjust_analysis`. A key that the capture already sets in
    `adjust_analysis` keeps the value of the capture.

    Args:
        capture: a capture of the spec, with `sample_rate` and `analysis_bandwidth`
            at the class defaults
        channel: the matched `CellChannel`
        index: the position of `capture` in `spec.captures`, for the error location
        consumed_keys: the parameter names of the enabled measurements; a generated
            key outside the set is dropped. None keeps every key.

    Raises:
        SpecValidationError: if the capture sets a `sample_rate` or
            `analysis_bandwidth` other than the class default or the derived value
    """
    sample_rate = sw.comms_5gnr.nr_sample_rate(
        channel.channel_bandwidth,
        channel.subcarrier_spacing,
        min_sample_rate=channel.min_sample_rate,
    )
    derived = {
        'sample_rate': sample_rate,
        'analysis_bandwidth': channel.channel_bandwidth,
    }
    defaults = _field_defaults(type(capture))
    for name, value in derived.items():
        given = getattr(capture, name)
        if given not in (defaults[name], value):
            raise SpecValidationError(
                f'Expected the default or the derived value {value!r}, since '
                f'`{name}` follows from the channel (set `min_sample_rate` in the '
                'channel to raise the sample rate)',
                (f'.captures[{index}]', f'.{name}'),
            )

    generated = {
        'frequency_offset': channel.ssb_frequency_offset,
        'guard_bandwidths': channel.channel_guard_bandwidths,
        'frame_slots': channel.tdd_frame_slots,
        'special_symbols': channel.tdd_special_symbols,
        'subcarrier_spacing': channel.subcarrier_spacing,
        'symbol_indexes': ssb_symbol_indexes(channel),
    }
    if consumed_keys is not None:
        generated = {k: v for k, v in generated.items() if k in consumed_keys}
    adjust_analysis = {**generated, **dict(capture.adjust_analysis)}
    return capture.replace(adjust_analysis=adjust_analysis, **derived)


def analysis_parameter_names(analysis: specs.AnalysisGroup) -> frozenset[str]:
    """return the field names of every enabled measurement in `analysis`."""
    names: set[str] = set()
    for meas in analysis.to_dict().values():
        if meas is not None:
            names.update(meas)
    return frozenset(names)


def channel_subcarrier_spacing(channels: Mapping[int, specs.CellChannel]) -> float:
    """return the one subcarrier spacing of the channels in `channels`.

    Args:
        channels: the channel matched to each capture, keyed by capture index

    Raises:
        SpecValidationError: if the channels differ in spacing, or there are none
    """
    spacings = {c.subcarrier_spacing for c in channels.values()}
    if len(spacings) != 1:
        raise SpecValidationError(
            'Expected the channels of the captures to share one subcarrier spacing, '
            f'got {sorted(spacings)!r}',
            ('.cell_channels',),
        )
    (spacing,) = spacings
    return spacing


def _resource_block(scs: float) -> float:
    return 12 * scs


def _same(scs: float) -> float:
    return scs


def _half(scs: float) -> float:
    return scs / 2


def _frame(scs: float) -> float:
    return FRAME_PERIOD


def _spacings(scs: float) -> tuple[float, ...]:
    return (scs,)


# the spectrogram parameters that place the bins on the OFDM symbol grid
_GRID: dict[str, Callable[[float], Any]] = {
    'frequency_resolution': _half,
    'fractional_overlap': lambda scs: SYMBOL_FRACTIONAL_OVERLAP,
    'window_fill': lambda scs: SYMBOL_WINDOW_FILL,
}

# measurement parameters that the subcarrier spacing fixes, by measurement name; the
# block struct of the same name omits these fields. A resource block is 12
# subcarriers (3GPP TS 38.211 §4.4.4.1).
FIXED: dict[str, dict[str, Callable[[float], Any]]] = {
    'spectrogram': {**_GRID, 'integration_bandwidth': _resource_block},
    'power_spectral_density': {**_GRID, 'integration_bandwidth': _same},
    'cellular_resource_power_histogram': {'subcarrier_spacing': _same},
    'cellular_5g_pss_correlation': {'subcarrier_spacing': _same},
    PSS_SYNC_TRIGGER: {'subcarrier_spacing': _same},
    'cellular_5g_ssb_spectrogram': {'subcarrier_spacing': _same},
    'cyclic_channel_power': {'cyclic_period': _frame, 'detector_period': symbol_period},
    'cellular_cyclic_autocorrelation': {'subcarrier_spacings': _spacings},
}

# block parameters that stay exposed as Optional and take the derived value when None
DEFAULTED: dict[str, dict[str, Callable[[float], Any]]] = {
    'channel_power_time_series': {'detector_period': symbol_period},
    'channel_power_histogram': {'detector_period': symbol_period},
}

# measurement parameters that `resolve_capture` writes into the `adjust_analysis` of
# each capture from the matched channel
PER_CAPTURE = frozenset({
    'frequency_offset',
    'guard_bandwidths',
    'frame_slots',
    'special_symbols',
    'subcarrier_spacing',
    'symbol_indexes',
})


def _optional_inner(field: msgspec.structs.FieldInfo) -> type[msgspec.Struct]:
    """return the struct type `X` of an ``Optional[X]`` field."""
    args = set(typing.get_args(field.type)) - {type(None)}
    (inner,) = args
    return typing.cast('type[msgspec.Struct]', inner)


def check_block_fields(
    block_type: type[msgspec.Struct], spec_type: type[msgspec.Struct], name: str
) -> None:
    """raise unless `block_type` matches the `FIXED`, `PER_CAPTURE`, and `DEFAULTED`
    tables for `name`.

    `block_type` must be declared with `subset_of` on `spec_type`. Every key of
    ``FIXED[name]``, and every `PER_CAPTURE` name that is a field of `spec_type`,
    must be among the omitted fields. The Optional fields must be exactly the keys of
    ``DEFAULTED[name]``. An omitted field in neither table keeps the measurement
    default.

    Raises:
        TypeError: if `block_type` is not a `subset_of` `spec_type`, if a fixed or
            per-capture field is exposed, or if the Optional fields differ from
            ``DEFAULTED[name]``
    """
    subset = spec_subset(block_type)
    if subset is None:
        raise TypeError(f'{block_type.__name__} is not declared with subset_of')
    if subset.base is not spec_type:
        raise TypeError(
            f'{block_type.__name__} is a subset of {subset.base.__name__}, not of '
            f'the {name!r} measurement spec {spec_type.__name__}'
        )

    spec_fields = set(spec_type.__struct_fields__)
    fixed = set(FIXED.get(name, {}))
    per_capture = PER_CAPTURE & spec_fields
    defaulted = set(DEFAULTED.get(name, {}))

    problems: list[str] = []
    if not fixed <= subset.without:
        problems.append(f'fixed but exposed {sorted(fixed - subset.without)!r}')
    if not per_capture <= subset.without:
        problems.append(
            f'per capture but exposed {sorted(per_capture - subset.without)!r}'
        )
    if defaulted != subset.optional:
        problems.append(
            f'defaulted {sorted(defaulted)!r} differs from optional '
            f'{sorted(subset.optional)!r}'
        )
    if problems:
        raise TypeError(f'{block_type.__name__} vs {name!r}: ' + '; '.join(problems))


def check_all_blocks() -> None:
    """run `check_block_fields` for every block of `Cellular5GNRAnalysis`.

    Raises:
        TypeError: if a field of `Cellular5GNRAnalysis` is not a measurement name, or
            from `check_block_fields`
    """
    measurement_types = {
        f.name: _optional_inner(f)
        for f in msgspec.structs.fields(specs.BuiltinAnalyses)
    }
    for field in msgspec.structs.fields(specs.Cellular5GNRAnalysis):
        if field.name not in measurement_types:
            raise TypeError(f'{field.name!r} is not a registered measurement name')
        check_block_fields(
            _optional_inner(field), measurement_types[field.name], field.name
        )


check_all_blocks()


def cellular_analysis(
    spec: specs.Cellular5GNRSweep[Any, Any, Any, Any], subcarrier_spacing: float
) -> specs.AnalysisGroup:
    """return the `Sweep` analysis that `spec.analysis` and `subcarrier_spacing` select.

    The result is an instance of the analysis group of the binding of `spec`. Each
    enabled block becomes the measurement of the same name, with the exposed
    parameters copied, the `FIXED` parameters computed from the spacing, and a None
    `DEFAULTED` parameter replaced by the derived value. A struct-form
    `signal_trigger` is accepted with or without the block of the same name.

    Args:
        subcarrier_spacing: the one subcarrier spacing (Hz) of the channels

    Raises:
        SpecValidationError: if `spec.source.signal_trigger` is a string other than
            `PSS_SYNC_TRIGGER`, or is `PSS_SYNC_TRIGGER` while the block of that
            name is None
    """
    scs = subcarrier_spacing
    result: dict[str, dict[str, Any]] = {}
    for name in specs.Cellular5GNRAnalysis.__struct_fields__:
        block = getattr(spec.analysis, name)
        if block is None:
            continue
        kws = block.to_dict()
        for key, fn in DEFAULTED.get(name, {}).items():
            if kws[key] is None:
                kws[key] = fn(scs)
        kws.update({key: fn(scs) for key, fn in FIXED.get(name, {}).items()})
        result[name] = kws

    # a string trigger reads its measurement spec from the sweep analysis, which must
    # enable that measurement; a struct-form signal_trigger includes the spec
    trigger = spec.source.signal_trigger
    if isinstance(trigger, str) and trigger not in result:
        if trigger == PSS_SYNC_TRIGGER:
            raise SpecValidationError(
                f'Expected a `{PSS_SYNC_TRIGGER}` block, since the source triggers '
                f'on {PSS_SYNC_TRIGGER!r}',
                ('.analysis', f'.{PSS_SYNC_TRIGGER}'),
            )
        raise SpecValidationError(
            f'Expected {PSS_SYNC_TRIGGER!r}, the trigger the cellular analysis '
            f'generates, or the struct form of signal_trigger, got {trigger!r}',
            ('.source', '.signal_trigger'),
        )

    return specs.action.analysis_group_type(spec).from_dict(result)


@sa.util.lru_cache()
def build_cellular_sweep(
    spec: specs.Cellular5GNRSweep[SS, SP, SC, SA],
) -> specs.Sweep[SS, SP, SC, SA]:
    """return the `Sweep` derived from a `Cellular5GNRSweep`.

    The captures of `spec` are resolved against `cell_channels` by `resolve_capture`.
    The loops and `adjust_captures` are copied, and the analysis is
    `cellular_analysis`. The result is an instance of the `sweep_cls` of the spec
    class and is memoized by `spec`.

    Raises:
        TypeError: if the class of `spec` is not bound to a sensor
        SpecValidationError: for a loop over `center_frequency`, a capture without a
            channel, channels of mixed subcarrier spacing, or a capture that sets a
            derived field
    """
    sweep_cls = type(spec).sweep_cls
    if sweep_cls is None:
        raise TypeError('the cellular sweep spec is not bound to a sensor')

    for i, loop in enumerate(spec.loops):
        if getattr(loop, 'field', None) == 'center_frequency':
            raise SpecValidationError(
                'Expected no loop over center_frequency: list the channels as captures',
                (f'.loops[{i}]', '.field'),
            )

    channels = {
        i: match_channel(c, spec.cell_channels, i) for i, c in enumerate(spec.captures)
    }
    scs = channel_subcarrier_spacing(channels)
    analysis = cellular_analysis(spec, scs)
    consumed = analysis_parameter_names(analysis)
    captures = tuple(
        resolve_capture(c, channels[i], i, consumed)
        for i, c in enumerate(spec.captures)
    )

    return sweep_cls(
        description=spec.description,
        source=spec.source,
        peripherals=spec.peripherals,
        extensions=spec.extensions,
        sink=spec.sink,
        captures=captures,
        loops=spec.loops,
        adjust_captures=spec.adjust_captures,
        plot_hint=spec.plot_hint,
        analysis=analysis,
        options=spec.options,
    )


@registry.runner(spec_type=specs.Cellular5GNRSweep)
class Cellular5GNRSweepRunner(Generic[SS, SP, SC, SA, PS, PC]):
    """the runner of a `Cellular5GNRSweep`.

    `sweep` is `build_cellular_sweep` of `spec`. The sink is a `ZarrCaptureSink` at
    ``sweep.sink.path``, or a `NoSink` when the path is None.
    """

    spec: specs.Cellular5GNRSweep[SS, SP, SC, SA]

    def __init__(self, spec: specs.Cellular5GNRSweep[SS, SP, SC, SA]) -> None:
        self.spec = spec

    @property
    def sweep(self) -> specs.Sweep[SS, SP, SC, SA]:
        """the `Sweep` built by `build_cellular_sweep` from `spec`.

        Raises:
            TypeError: if the class of `spec` is not bound to a sensor
            SpecValidationError: if the captures, loops, or channels of `spec` fail
                the checks of `build_cellular_sweep`
        """
        return build_cellular_sweep(self.spec)

    def open_sink(self, format_path: specs.paths.PathFormatter | None) -> SinkBase[SC]:
        if self.sweep.sink.path is None:
            return NoSink(self.sweep)
        return ZarrCaptureSink(self.sweep, format_path)

    def iterate(
        self,
        resources: Resources[SS, SP, SC, SA, PS, PC],
        *,
        always_yield: bool = False,
        yield_values: bool = True,
        loop: bool = False,
    ) -> Generator[xr.Dataset | compute.DelayedDataset | None]:
        return core.iterate_captures(
            resources,
            always_yield=always_yield,
            yield_values=yield_values,
            loop=loop,
        )
