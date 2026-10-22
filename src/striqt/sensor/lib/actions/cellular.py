"""the action that runs a `Cellular5GNRSweep` as a plain `Sweep`.

The transform from channel table to radio settings is explicit and runs once, when the
runner builds its sweep: each capture is matched to the channel with its
`center_frequency`, its `sample_rate` and `analysis_bandwidth` are derived from that
channel, the channel's cell parameters are written into the capture's
`adjust_analysis`, and the analysis block is expanded into an `AnyAnalysis` with the
parameters that follow from the subcarrier spacing filled in. The sweep's captures
are the binding's own type, and loops over their fields pass through unchanged.
"""

from __future__ import annotations as __

from collections.abc import Collection, Generator, Mapping
from fractions import Fraction
from typing import Any, Generic

import msgspec

import striqt.analysis as sa
import striqt.waveform as sw
from striqt.analysis.lib.util import xr
from striqt.analysis.specs.helpers import SpecValidationError

from ... import specs
from .. import compute
from ..resources import Resources
from ..sinks import NoSink, SinkBase, ZarrCaptureSink
from ..typing import PC, PS, SC, SP, SS
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
    """return the SSB `symbol_indexes` of `channel`: its `ssb_case`, or the case of its spacing.

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

    `sample_rate` is `nr_sample_rate` of the channel, `analysis_bandwidth` is the
    channel bandwidth, and `adjust_analysis` gains `frequency_offset`,
    `guard_bandwidths`, `frame_slots`, `special_symbols`, `subcarrier_spacing`, and
    `symbol_indexes` from the channel, under any key the capture already sets.

    Args:
        capture: a capture of the spec, at its class defaults for `sample_rate` and
            `analysis_bandwidth`
        channel: the matched `CellChannel`
        index: the position of `capture` in `spec.captures`, for the error location
        consumed_keys: the analysis parameter names some enabled measurement reads;
            generated keys outside it are omitted. None generates every key.

    Raises:
        SpecValidationError: if the capture sets a `sample_rate` or
            `analysis_bandwidth` other than its class default or the derived value
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


def analysis_parameter_names(analysis: specs.AnyAnalysis) -> frozenset[str]:
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


def cellular_analysis(
    spec: specs.Cellular5GNRSweep[Any, Any, Any], subcarrier_spacing: float
) -> specs.AnyAnalysis:
    """return the `Sweep` analysis that `spec.analysis` and `subcarrier_spacing` select.

    Each enabled block of `spec.analysis` becomes the measurement of the same role,
    with its free parameters copied and the spacing-dependent parameters filled.

    Raises:
        SpecValidationError: if `spec.source.signal_trigger` is the PSS sync name and
            `spec.analysis.pss_sync` is None, or any other trigger name; the struct
            form of `signal_trigger` carries its own spec and needs no block
    """
    scs = subcarrier_spacing
    blocks = spec.analysis
    symbol = symbol_period(scs)
    grid = {
        'frequency_resolution': scs / 2,
        'fractional_overlap': SYMBOL_FRACTIONAL_OVERLAP,
        'window_fill': SYMBOL_WINDOW_FILL,
    }
    result: dict[str, dict[str, Any]] = {}

    if blocks.spectrogram is not None:
        result['spectrogram'] = {
            **blocks.spectrogram.to_dict(),
            **grid,
            'integration_bandwidth': 12 * scs,
        }
    if blocks.power_spectral_density is not None:
        result['power_spectral_density'] = {
            **blocks.power_spectral_density.to_dict(),
            **grid,
            'integration_bandwidth': scs,
        }
    if blocks.resource_power_histogram is not None:
        result['cellular_resource_power_histogram'] = {
            **blocks.resource_power_histogram.to_dict(),
            'subcarrier_spacing': scs,
        }
    if blocks.pss_correlation is not None:
        result['cellular_5g_pss_correlation'] = {
            **blocks.pss_correlation.to_dict(),
            'subcarrier_spacing': scs,
        }
    if blocks.pss_sync is not None:
        result[PSS_SYNC_TRIGGER] = {
            **blocks.pss_sync.to_dict(),
            'subcarrier_spacing': scs,
        }
    if blocks.ssb_spectrogram is not None:
        result['cellular_5g_ssb_spectrogram'] = {
            **blocks.ssb_spectrogram.to_dict(),
            'subcarrier_spacing': scs,
        }
    if blocks.cyclic_channel_power is not None:
        result['cyclic_channel_power'] = {
            **blocks.cyclic_channel_power.to_dict(),
            'cyclic_period': FRAME_PERIOD,
            'detector_period': symbol,
        }
    if blocks.channel_power_time_series is not None:
        kws = blocks.channel_power_time_series.to_dict()
        if kws['detector_period'] is None:
            kws['detector_period'] = symbol
        result['channel_power_time_series'] = kws
    if blocks.channel_power_histogram is not None:
        kws = blocks.channel_power_histogram.to_dict()
        if kws['detector_period'] is None:
            kws['detector_period'] = symbol
        result['channel_power_histogram'] = kws
    if blocks.cyclic_autocorrelation is not None:
        result['cellular_cyclic_autocorrelation'] = {
            **blocks.cyclic_autocorrelation.to_dict(),
            'subcarrier_spacings': (scs,),
        }
    if blocks.iq_waveform is not None:
        result['iq_waveform'] = blocks.iq_waveform.to_dict()

    # a string trigger takes its spec from the sweep analysis, so the analysis must
    # enable that measurement; the struct form of signal_trigger brings its own spec
    trigger = spec.source.signal_trigger
    if isinstance(trigger, str) and trigger not in result:
        if trigger == PSS_SYNC_TRIGGER:
            raise SpecValidationError(
                f'Expected a `pss_sync` block, since the source triggers on '
                f'{PSS_SYNC_TRIGGER!r}',
                ('.analysis', '.pss_sync'),
            )
        raise SpecValidationError(
            f'Expected {PSS_SYNC_TRIGGER!r}, the trigger the cellular analysis '
            f'generates, or the struct form of signal_trigger, got {trigger!r}',
            ('.source', '.signal_trigger'),
        )

    return specs.AnyAnalysis.from_dict(result)


@sa.util.lru_cache()
def build_cellular_sweep(
    spec: specs.Cellular5GNRSweep[SS, SP, SC],
) -> specs.Sweep[SS, SP, SC]:
    """return the `Sweep` a cellular sweep opens its resources for.

    The sweep has the spec's captures resolved against `cell_channels`
    (`resolve_capture`), the spec's loops and `adjust_captures` unchanged, and the
    analysis of `cellular_analysis`. It is an instance of the spec class's
    `sweep_cls`, and the result is cached on `spec`.

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
class Cellular5GNRSweepRunner(Generic[SS, SP, SC, PS, PC]):
    """run a `Cellular5GNRSweep`.

    `sweep` is `build_cellular_sweep` of the spec. The output is a `ZarrCaptureSink`
    at `sink.path`, or nothing when the path is None, as for a plain `Sweep`.
    """

    spec: specs.Cellular5GNRSweep[SS, SP, SC]
    sweep: specs.Sweep[SS, SP, SC]

    def __init__(self, spec: specs.Cellular5GNRSweep[SS, SP, SC]) -> None:
        self.spec = spec
        self.sweep = build_cellular_sweep(spec)

    @classmethod
    def validate(cls, spec: specs.Cellular5GNRSweep[SS, SP, SC]) -> None:
        """build the sweep of `spec`, raising what `build_cellular_sweep` raises."""
        build_cellular_sweep(spec)

    def open_sink(self, format_path: specs.paths.PathFormatter | None) -> SinkBase[SC]:
        if self.sweep.sink.path is None:
            return NoSink(self.sweep)
        return ZarrCaptureSink(self.sweep, format_path)

    def iterate(
        self,
        resources: Resources[SS, SP, SC, PS, PC],
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
