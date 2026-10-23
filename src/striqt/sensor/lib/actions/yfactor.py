"""the manual Y-factor calibration action.

The operator switches the noise diode by hand. The runner acquires the capture grid
of each port once with the diode off and once with the diode on, and prompts the
operator between the passes. The diode state is not a capture field. The state
enters each acquisition through the `extra_data` of the pipeline and is a dimension
of the saved corrections beside the capture fields.
"""

from __future__ import annotations as __

from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any, Generic

import striqt.analysis as sa
from striqt.analysis.lib.util import np, pd, xr
from striqt.analysis.specs.helpers import SpecValidationError

from ... import specs
from .. import compute, io
from ..compute import cal_lookups
from ..resources import Resources
from ..sinks import MemoryCacheSink, SinkBase
from ..typing import PC, PS, SA, SC, SP, SS
from . import core
from .registry import registry

DIODE_FIELD = 'noise_diode_enabled'


def calibration_analysis(
    spec: specs.ManualYFactorCalibration[Any, Any, Any, Any],
) -> specs.BuiltinAnalyses:
    """return the analysis a Y-factor calibration runs on each capture.

    The analysis is `channel_power_time_series` with the ``rms`` and ``peak``
    detectors. The detector period is `spec.calibration.detector_period`, or the one
    `duration` that the captures and loops share when that is None.

    Raises:
        SpecValidationError: at ``$.calibration.detector_period`` when
            `detector_period` is None and the captures and loops have more than one
            `duration`
    """
    period = spec.calibration.detector_period

    if period is None:
        # a looped field overrides the captures' own values
        duration_loops = [
            loop
            for loop in spec.loops
            if loop.isin == 'capture' and loop.field == 'duration'
        ]
        if duration_loops:
            durations = set(duration_loops[0].get_points())
        else:
            durations = {c.duration for c in spec.captures}
        if len(durations) != 1:
            raise SpecValidationError(
                'Expected a `detector_period`, since the captures and loops do not '
                'share one `duration`',
                ('.calibration', '.detector_period'),
            )
        (period,) = durations

    return specs.BuiltinAnalyses.from_dict({
        'channel_power_time_series': {
            'detector_period': period,
            'power_detectors': ('rms', 'peak'),
        }
    })


def _limit_nyquist_bandwidth(data: xr.DataArray) -> xr.DataArray:
    """replace float('inf') analysis bandwidth with the Nyquist bandwidth"""

    # return bandwidth with same shape as dataset.channel_power_time_series
    bw = data.analysis_bandwidth.broadcast_like(data).copy().squeeze()
    sample_rate = data.backend_sample_rate.broadcast_like(data).squeeze()
    where = ~np.isfinite(bw.values)
    bw.values[where] = sample_rate.values[where]
    return bw


def compute_y_factor_corrections(
    dataset: xr.Dataset, enr_dB: float, Tref: float = 290.0
) -> xr.Dataset:
    """return the Y-factor noise figure and power correction of each grid point.

    The dimensions of `dataset` are the calibration fields and
    ``noise_diode_enabled``. The variables are `channel_power_time_series` (dB,
    ``rms`` and ``peak`` detectors over ``time_elapsed``), `analysis_bandwidth`
    (Hz), and `backend_sample_rate` (S/s). The rms detector is averaged in linear
    power over ``time_elapsed`` before the Y-factor is formed. An infinite
    `analysis_bandwidth` counts as the backend sample rate.

    Args:
        dataset: the collected captures as `save_yfactor_corrections` indexes them
        enr_dB: the excess noise ratio (dB) of the noise diode
        Tref: the reference temperature (K) of the hot source and the noise figure

    Returns:
        a dataset over the calibration fields with `noise_figure` (dB),
        `temperature` (K), `power_correction` (mW per full-scale power unit), the
        linear powers `p_on` and `p_off`, and the diagnostics `papr_on` and `papr_off`
        (dB)
    """
    from scipy.constants import Boltzmann

    k = Boltzmann * 1000  # W/K -> mW/K
    enr = 10 ** (enr_dB / 10.0)

    pvt = dataset.channel_power_time_series.drop_vars('capture_index')
    power = (
        pvt
        .sel(power_detector='rms', drop=True)
        .pipe(sa.dBtopow)
        .mean(dim='time_elapsed')
    )
    power.name = 'RMS power'

    peak = pvt.sel(power_detector='peak', drop=True).max('time_elapsed')
    avg = pvt.sel(power_detector='rms', drop=True).pipe(sa.dBlinmean, 'time_elapsed')
    papr = (peak - avg).assign_attrs(units='dB')

    Pon = power.sel(noise_diode_enabled=True, drop=True)
    Poff = power.sel(noise_diode_enabled=False, drop=True)
    Y = Pon / Poff

    noise_figure = enr_dB - 10 * np.log10(Y - 1)
    noise_figure.name = 'Noise figure'
    noise_figure.attrs = {'units': 'dB'}

    Te = Tref * (10 ** (noise_figure / 10) - 1)
    Te.name = 'Effective noise temperature'
    Te.attrs = {'units': 'K'}

    B = _limit_nyquist_bandwidth(Te)

    power_correction = (k * (enr * Tref) * B) / (Pon - Poff)
    power_correction.name = 'Power scaling correction'
    # FS: the full-scale power unit of the uncalibrated detector, 0 dBFS
    power_correction.attrs = {'units': 'mW/FS'}

    return xr.Dataset({
        'temperature': Te,
        'noise_figure': noise_figure,
        'power_correction': power_correction,
        'p_on': Pon,
        'p_off': Poff,
        'papr_on': papr.sel(noise_diode_enabled=True, drop=True),
        'papr_off': papr.sel(noise_diode_enabled=False, drop=True),
    })


_DROP_FIELDS = (
    'sweep_start_time',
    'start_time',
    'delay',
    'host_resample',
    'duration',
    'sample_rate',
)


def save_yfactor_corrections(
    data: Sequence[xr.Dataset],
    calibration_fields: Sequence[str],
    path: Path,
    *,
    enr_dB: float,
) -> xr.Dataset:
    """compute the Y-factor corrections from the collected captures and save them.

    `data` is one dataset per capture, as the sink collected them. Each of
    `calibration_fields` becomes a dimension of the concatenated data. The
    corrections of `compute_y_factor_corrections` at `enr_dB` are written to `path`
    as netCDF with a ``calibration_fields`` attr and summarized per port on stdout.

    Returns:
        the saved corrections

    Raises:
        KeyError: if a calibration field is neither a capture coordinate nor a data
            variable of the collected datasets
    """
    fields = list(calibration_fields)

    # re-index by radio setting rather than capture; the diode state arrives as a
    # data variable from the runner, the rest as capture coordinates
    capture_data = xr.concat(data, compute.CAPTURE_DIM).drop_vars(
        _DROP_FIELDS, errors='ignore'
    )
    missing = [f for f in fields if f not in capture_data.variables]
    if missing:
        raise KeyError(f'calibration fields {missing!r} are not in the captures')
    to_coords = [f for f in fields if f in capture_data.data_vars]
    capture_data = capture_data.set_coords(to_coords).set_xindex(fields)

    by_field = capture_data.unstack(compute.CAPTURE_DIM)
    by_field[DIODE_FIELD] = by_field[DIODE_FIELD].astype('bool')

    corrections = compute_y_factor_corrections(by_field, enr_dB)
    corrections = corrections.assign_attrs(calibration_fields=fields)

    io.save_calibration(path, corrections)
    print(f'saved to {str(path)!r}')
    for port, _ in corrections.groupby('port', squeeze=False):
        print(f'{30 * "▀"} Port {port} at max gain {30 * "▀"}')
        summary = cal_lookups.summarize_calibration(corrections, port=port)
        with pd.option_context('display.max_rows', None, 'display.precision', 2):
            print(summary.sort_index(axis=1).sort_index(axis=0))

    return corrections


def _is_port_loop(loop: specs.LoopSpec) -> bool:
    return loop.isin == 'capture' and loop.field == 'port'


@sa.util.lru_cache()
def build_calibration_sweep(
    spec: specs.ManualYFactorCalibration[SS, SP, SC, SA],
) -> specs.Sweep[SS, SP, SC, SA]:
    """return the `Sweep` that `open_resources` opens for a `ManualYFactorCalibration`.

    The sweep is the spec's capture plan over every port, with the analysis of
    `calibration_analysis`, IQ reuse across the `analysis_bandwidth` points of one
    radio setting, `loop_only_nyquist`, and no warmup. The `sink` is the spec's
    `Sink` converted to a `ZarrSink` with default store fields. The store fields
    of `sink` are unused. The sweep is an instance of the
    `sweep_cls` of the spec class, and the result is cached on `spec`.

    Raises:
        TypeError: if `spec` is not an instance of a bound action spec class
        msgspec.ValidationError: if the captures and loops do not share one
            `duration` and `calibration.detector_period` is None, or if the sweep
            fails `validate_sweep`
    """
    sweep_cls = type(spec).sweep_cls
    if sweep_cls is None:
        raise TypeError('the calibration spec is not bound to a sensor')

    return sweep_cls(
        description=spec.description,
        source=spec.source,
        peripherals=spec.peripherals,
        extensions=spec.extensions,
        sink=specs.ZarrSink.from_spec(spec.sink),
        captures=spec.captures,
        loops=spec.loops,
        adjust_captures=spec.adjust_captures,
        analysis=calibration_analysis(spec),
        # reuse_iq lets the analysis_bandwidth points of one radio setting share
        # an acquisition. The runner discards it at each diode switch, which the
        # capture spec does not describe (Controller.discard_iq).
        options=specs.SweepOptions(
            reuse_iq=True, loop_only_nyquist=True, skip_warmup=True
        ),
    )


def calibration_ports(
    spec: specs.ManualYFactorCalibration[Any, Any, Any, Any],
) -> tuple[Any, ...]:
    """return the ports to calibrate in the order of the `port` loop or the captures."""
    for loop in spec.loops:
        if _is_port_loop(loop):
            return tuple(loop.get_points())

    ports: list[Any] = []
    for capture in spec.captures:
        if capture.port not in ports:
            ports.append(capture.port)
    return tuple(ports)


def calibration_passes(
    spec: specs.ManualYFactorCalibration[Any, Any, Any, Any],
) -> list[tuple[Any, bool]]:
    """return the ``(port, diode_on)`` of each pass: port-major, diode off then on."""
    return [(port, on) for port in calibration_ports(spec) for on in (False, True)]


def pass_sweep(
    sweep: specs.Sweep[SS, SP, SC, SA], port: Any
) -> specs.Sweep[SS, SP, SC, SA]:
    """return `sweep` with the captures fixed to `port` and without a `port` loop."""
    loops = tuple(loop for loop in sweep.loops if not _is_port_loop(loop))
    if len(loops) < len(sweep.loops):
        captures = tuple(c.replace(port=port) for c in sweep.captures)
    else:
        captures = tuple(c for c in sweep.captures if c.port == port)
    return sweep.replace(captures=captures, loops=loops)


def calibration_fields(
    spec: specs.ManualYFactorCalibration[Any, Any, Any, Any],
    sweep: specs.Sweep[Any, Any, Any, Any],
) -> list[str]:
    """return the capture fields that index the saved corrections.

    The implied loops of `spec.calibration` come first, then the loop fields of
    `sweep` in order. The diode state follows `port`, or comes first when there is no
    `port` loop. `sample_rate` is renamed to the acquisition's `backend_sample_rate`.
    """
    fields = list(spec.calibration.implied_loops)
    fields += [loop.field for loop in sweep.loops if loop.field is not None]
    if 'sample_rate' in fields:
        fields[fields.index('sample_rate')] = 'backend_sample_rate'
    at = fields.index('port') + 1 if 'port' in fields else 0
    fields.insert(at, DIODE_FIELD)
    return fields


def corrections_path(
    sweep: specs.Sweep[Any, Any, Any, Any],
    format_path: specs.paths.PathFormatter | None,
) -> Path:
    """return `sweep.sink.path` with the ``{fields}`` templates formatted.

    Raises:
        ValueError: if `sweep.sink.path` is None
    """
    path = sweep.sink.path
    if path is None:
        raise ValueError('sink.path is null, so the calibration has no output path')
    if format_path is not None:
        path = format_path(path)
    return Path(path)


def confirm_enr(enr: float) -> None:
    """ask the operator to confirm the noise diode ENR `enr` (dB).

    Raises:
        RuntimeError: if the operator answers no
    """
    prompt = f'Confirm that the noise diode ENR is {enr} dB (y/n): '
    while True:
        response = sa.util.blocking_input(prompt)
        if response.lower() == 'y':
            return
        elif response.lower() == 'n':
            raise RuntimeError('Set the proper noise diode ENR and run again')


def switch_diode(port: Any, diode_on: bool) -> None:
    """prompt the operator to set the noise diode at `port` and wait for enter."""
    what = f'noise diode at port {port}'
    if diode_on:
        sa.util.blocking_input(f'enable {what} and press enter: ')
    else:
        sa.util.blocking_input(f'disable {what} and press enter: ')


@registry.runner(spec_type=specs.ManualYFactorCalibration)
class ManualYFactorRunner(Generic[SS, SP, SC, SA, PS, PC]):
    """run a `ManualYFactorCalibration`.

    `sweep` is `build_calibration_sweep` of the spec, and the resources open for
    that sweep. The captures of every pass collect in one `MemoryCacheSink`. After
    the last pass, the corrections are computed from them and written to the spec's
    `sink.path` as netCDF. A path of None writes nothing.
    """

    spec: specs.ManualYFactorCalibration[SS, SP, SC, SA]
    sweep: specs.Sweep[SS, SP, SC, SA]

    def __init__(self, spec: specs.ManualYFactorCalibration[SS, SP, SC, SA]) -> None:
        self.spec = spec
        self.sweep = build_calibration_sweep(spec)

    @classmethod
    def validate(cls, spec: specs.ManualYFactorCalibration[SS, SP, SC, SA]) -> None:
        """build the calibration sweep of `spec`.

        Raises:
            msgspec.ValidationError: what `build_calibration_sweep` raises
            TypeError: if `spec` is not an instance of a bound action spec class
        """
        build_calibration_sweep(spec)

    def open_sink(self, format_path: specs.paths.PathFormatter | None) -> SinkBase[SC]:
        return MemoryCacheSink(self.sweep)

    def iterate(
        self,
        resources: Resources[SS, SP, SC, SA, PS, PC],
        *,
        always_yield: bool = False,
        yield_values: bool = True,
        loop: bool = False,
    ) -> Generator[xr.Dataset | compute.DelayedDataset | None]:
        """run every pass, then compute and save the corrections.

        The yields are those of `iterate_captures` over each pass. Each pass begins
        with a new acquisition (`Controller.discard_iq`). Within a pass, `reuse_iq`
        applies. The corrections are saved only when the sink is a `MemoryCacheSink`
        and the spec's `sink.path` is not None.

        Raises:
            ValueError: if `loop` is True
            RuntimeError: if the operator rejects the ENR confirmation
        """
        if loop:
            raise ValueError('a calibration does not loop')

        confirm_enr(self.spec.calibration.enr)

        source_id = resources['source'].source_id
        sweeps = {
            port: pass_sweep(self.sweep, port) for port in calibration_ports(self.spec)
        }
        count = 2 * sum(
            len(specs.sequencing.loop_captures(sweep, source_id))
            for sweep in sweeps.values()
        )
        indexer = core.AcquisitionIndexer(count)

        for port, diode_on in calibration_passes(self.spec):
            switch_diode(port, diode_on)
            # the diode state is not a capture field, so the controller cannot see
            # that the acquisition it would reuse is now stale
            resources['source'].discard_iq()
            inner = Resources(resources, sweep_spec=sweeps[port])
            yield from core.iterate_captures(
                inner,
                extra_data={DIODE_FIELD: diode_on},
                indexer=indexer,
                always_yield=always_yield,
                yield_values=yield_values,
            )

        sink = resources['sink']
        if not isinstance(sink, MemoryCacheSink):
            return
        if self.spec.sink.path is None:
            sa.util.get_logger('sink').info(
                'sink.path is null: the corrections were not written'
            )
            return
        save_yfactor_corrections(
            sink.pop(),
            calibration_fields(self.spec, self.sweep),
            corrections_path(self.sweep, resources['format_path']),
            enr_dB=self.spec.calibration.enr,
        )
