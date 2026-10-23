"""the action that runs a `Cellular5GNRSweep` as a `Sweep`.

The `Sweep` is built once per spec, when the runner is constructed, and cached on the
spec.
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

# the OFDM symbol grid of a normal cyclic prefix slot: 14 symbols whose STFT windows
# overlap by 13/28 and fill 15/28 of each hop at SCS/2 resolution
SYMBOL_FRACTIONAL_OVERLAP = Fraction(13, 28)
SYMBOL_WINDOW_FILL = Fraction(15, 28)
FRAME_PERIOD = 10e-3
SYMBOLS_PER_SLOT = 14
PSS_SYNC_TRIGGER = 'cellular_5g_pss_sync'


def symbol_period(subcarrier_spacing: float) -> Fraction:
    """return the duration (in s) of one OFDM symbol with a normal cyclic prefix."""
    slots_per_frame = round(10 * subcarrier_spacing / 15e3)
    return Fraction(1, 100 * slots_per_frame * SYMBOLS_PER_SLOT)


def ssb_symbol_indexes(channel: specs.CellChannel) -> Any:
    """return the `symbol_indexes` of `channel`: `ssb_case`, or the case of the spacing.

    The default is the TS 38.213 §4.1 case ``'a'`` at 15 kHz and ``'c'`` at 30 kHz.
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
        SpecValidationError: if the ports name different center frequencies, or no
            channel has the capture's center frequency
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
        consumed_keys: the analysis parameter names some enabled measurement reads;
            generated keys outside that set are omitted. None generates every key.

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


# the spectrogram parameters of the OFDM symbol grid defined above
_GRID: dict[str, Callable[[float], Any]] = {
    'frequency_resolution': _half,
    'fractional_overlap': lambda scs: SYMBOL_FRACTIONAL_OVERLAP,
    'window_fill': lambda scs: SYMBOL_WINDOW_FILL,
}

# measurement parameters that the subcarrier spacing fixes, by measurement name; they
# are absent from the block struct of the same name. A resource block is 12
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

# measurement parameters that `resolve_capture` writes into each capture's
# `adjust_analysis` from its channel
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
    """raise unless the tables for `name` agree with the declaration of `block_type`.

    `block_type` is declared with `subset_of` on `spec_type`. The `FIXED[name]` keys
    and the `PER_CAPTURE` names that `spec_type` has are among the fields the block
    omits, and `DEFAULTED[name]` names exactly the fields the block widens to Optional.
    An omitted field in neither table stays at the measurement default.

    Raises:
        TypeError: naming the undeclared block, the mismatched base, or the table
            entries that the block declaration contradicts
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
        TypeError: for a block whose name is not a measurement, or from
            `check_block_fields`
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
    `DEFAULTED` parameter replaced by the derived value.

    Raises:
        SpecValidationError: if `spec.source.signal_trigger` is the PSS sync name and
            the block of that name is None, or any other trigger name. The struct
            form of `signal_trigger` is accepted without a block
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

    # a string trigger takes its spec from the sweep analysis, so the analysis must
    # enable that measurement; the struct form of signal_trigger brings its own spec
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
    """return the `Sweep` that `open_resources` opens for a `Cellular5GNRSweep`.

    The sweep has the spec's captures resolved against `cell_channels`
    (`resolve_capture`), the spec's loops and `adjust_captures` unchanged, and the
    analysis of `cellular_analysis`. The sweep is an instance of the `sweep_cls` of
    the spec class, and the result is cached on `spec`.

    Raises:
        TypeError: if `spec` is not an instance of a bound spec class
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
    """run a `Cellular5GNRSweep`.

    `sweep` is `build_cellular_sweep` of the spec. The output is a `ZarrCaptureSink`
    at `sink.path`, or nothing when the path is None.
    """

    spec: specs.Cellular5GNRSweep[SS, SP, SC, SA]
    sweep: specs.Sweep[SS, SP, SC, SA]

    def __init__(self, spec: specs.Cellular5GNRSweep[SS, SP, SC, SA]) -> None:
        self.spec = spec
        self.sweep = build_cellular_sweep(spec)

    @classmethod
    def validate(cls, spec: specs.Cellular5GNRSweep[SS, SP, SC, SA]) -> None:
        """build the sweep of `spec`, raising what `build_cellular_sweep` raises."""
        build_cellular_sweep(spec)

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
