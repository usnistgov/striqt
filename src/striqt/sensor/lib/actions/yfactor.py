"""the manual Y-factor calibration action.

The runner acquires the calibration grid once with the noise diode off and once
with it on at each port, prompting the operator to switch the diode between those
passes, and generates the `channel_power_time_series` analysis that the Y-factor
math in `striqt.sensor.lib.calibration` consumes. The diode state reaches the saved
data through the per-capture `extra_data` hook of the pipeline, the captures collect
in a `MemoryCacheSink`, and `save_yfactor_corrections` computes and writes the
corrections after the last pass.
"""

from __future__ import annotations as __

from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any

import striqt.analysis as sa
from striqt.analysis.lib.util import pd, xr
from striqt.analysis.specs.helpers import SpecValidationError

from ... import specs
from .. import calibration, compute, io
from ..resources import Resources, resolve_sink_cls
from ..sinks import MemoryCacheSink, SinkBase
from ..typing import PC, PS, SC, SP, SS
from . import core
from .registry import registry

DIODE_FIELD = 'noise_diode_enabled'


def calibration_analysis(
    spec: specs.ManualYFactorCalibration[Any, Any, Any],
) -> specs.BundledAnalysis:
    """return the analysis a Y-factor calibration runs on each capture.

    The analysis is `channel_power_time_series` with the ``rms`` and ``peak``
    detectors. Its detector period is `spec.calibration.detector_period`, or the one
    `duration` that the captures and loops share when that is None.

    Raises:
        SpecValidationError: at ``$.calibration.detector_period`` when it is None and
            the captures and loops carry more than one `duration`
    """
    period = spec.calibration.detector_period

    if period is None:
        durations = {c.duration for c in spec.captures}
        for loop in spec.loops:
            if loop.isin == 'capture' and loop.field == 'duration':
                durations.update(loop.get_points())
        if len(durations) != 1:
            raise SpecValidationError(
                'Expected a `detector_period`, since the captures and loops do not '
                'share one `duration`',
                ('.calibration', '.detector_period'),
            )
        (period,) = durations

    return specs.BundledAnalysis.from_dict({
        'channel_power_time_series': {
            'detector_period': period,
            'power_detectors': ('rms', 'peak'),
        }
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
    data: Sequence[xr.Dataset], calibration_fields: Sequence[str], path: Path
) -> xr.Dataset:
    """compute the Y-factor corrections from the collected captures and save them.

    `data` holds one dataset per capture as the sink collected them. They are
    indexed by `calibration_fields` so that each field becomes a dimension, the
    corrections are computed with `compute_y_factor_corrections`, written to `path`
    as netCDF with a ``calibration_fields`` attr, and summarized per port on stdout.

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

    # break out each remaining capture coordinate into its own dimension
    by_field = capture_data.unstack(compute.CAPTURE_DIM)
    by_field[DIODE_FIELD] = by_field[DIODE_FIELD].astype('bool')

    corrections = calibration.compute_y_factor_corrections(by_field)
    corrections = corrections.assign_attrs(calibration_fields=fields)

    io.save_calibration(path, corrections)
    print(f'saved to {str(path)!r}')
    for port, _ in corrections.groupby('port', squeeze=False):
        print(f'{30 * "▀"} Port {port} at max gain {30 * "▀"}')
        summary = calibration.summarize_calibration(corrections, port=port)
        with pd.option_context('display.max_rows', None, 'display.precision', 2):
            print(summary.sort_index(axis=1).sort_index(axis=0))

    return corrections


def _is_port_loop(loop: specs.LoopSpec) -> bool:
    return loop.isin == 'capture' and loop.field == 'port'


@registry.runner(spec_type=specs.ManualYFactorCalibration)
class ManualYFactorRunner(
    core.ActionRunner[specs.ManualYFactorCalibration[SS, SP, SC], SS, SP, SC, PS, PC]
):
    """run a `ManualYFactorCalibration`.

    `sweep` is the document's capture plan over every port with the generated
    analysis; the resources open for it. `iterate` confirms the ENR with the
    operator, then for each port and each diode state prompts the operator to
    switch the diode and runs the captures of that port through `iterate_captures`.
    `extra_data` adds the diode state and the `calibration:` block to every
    acquisition, `build_sink` returns a `MemoryCacheSink` that collects the
    captures, and `iterate` ends by computing and saving the corrections with
    `save_yfactor_corrections`.
    """

    _diode_on: bool = False

    @classmethod
    def build_sweep(
        cls, spec: specs.ManualYFactorCalibration[SS, SP, SC]
    ) -> specs.Sweep[SS, SP, SC]:
        sweep_cls = type(spec).sweep_cls
        if sweep_cls is None:
            raise TypeError('the calibration document is not bound to a sensor')

        return sweep_cls(
            description=spec.description,
            source=spec.source,
            peripherals=spec.peripherals,
            extensions=spec.extensions,
            sink=spec.sink,
            captures=spec.captures,
            loops=spec.loops,
            adjust_captures=spec.adjust_captures,
            analysis=calibration_analysis(spec),
            # consecutive passes may start with the capture the previous one ended
            # with, which the controller would otherwise answer with the reused IQ
            options=specs.SweepOptions(
                reuse_iq=False, loop_only_nyquist=True, skip_warmup=True
            ),
        )

    def ports(self) -> tuple[Any, ...]:
        """return the port values of the passes, in the order of a `port` loop or of the captures."""
        for loop in self.spec.loops:
            if _is_port_loop(loop):
                return tuple(loop.get_points())

        ports: list[Any] = []
        for capture in self.spec.captures:
            if capture.port not in ports:
                ports.append(capture.port)
        return tuple(ports)

    def passes(self) -> list[tuple[Any, bool]]:
        """return the ``(port, diode_on)`` of each pass: port-major, diode off then on."""
        return [(port, on) for port in self.ports() for on in (False, True)]

    def pass_sweep(self, port: Any) -> specs.Sweep[SS, SP, SC]:
        """return `sweep` restricted to `port`: its captures fixed to `port`, without a `port` loop."""
        loops = tuple(loop for loop in self.sweep.loops if not _is_port_loop(loop))
        if len(loops) < len(self.sweep.loops):
            captures = tuple(c.replace(port=port) for c in self.sweep.captures)
        else:
            captures = tuple(c for c in self.sweep.captures if c.port == port)
        return self.sweep.replace(captures=captures, loops=loops)

    def calibration_fields(self) -> list[str]:
        """return the capture fields that index the saved corrections.

        The implied loops come first, then the loop fields in order with the diode
        state after `port` (or first), and `sample_rate` reads as the acquisition's
        `backend_sample_rate`.
        """
        fields = list(self.spec.calibration.implied_loops)
        fields += [loop.field for loop in self.sweep.loops if loop.field is not None]
        if 'sample_rate' in fields:
            fields[fields.index('sample_rate')] = 'backend_sample_rate'
        at = fields.index('port') + 1 if 'port' in fields else 0
        fields.insert(at, DIODE_FIELD)
        return fields

    def build_sink(self, format_path: specs.paths.PathFormatter | None) -> SinkBase[SC]:
        sink_cls = resolve_sink_cls(self.sweep, MemoryCacheSink)
        return sink_cls(self.sweep, format_path)

    def corrections_path(self, format_path: specs.paths.PathFormatter | None) -> Path:
        """return `sweep.sink.path` with its ``{fields}`` formatted."""
        path = self.sweep.sink.path
        if format_path is not None:
            path = format_path(path)
        return Path(path)

    def extra_data(self, capture: SC) -> dict[str, Any]:
        cal = self.spec.calibration
        return {
            DIODE_FIELD: self._diode_on,
            'enr': cal.enr,
            'ambient_temperature': cal.ambient_temperature,
        }

    def iterate(
        self,
        resources: Resources[SS, SP, SC, PS, PC],
        *,
        always_yield: bool = False,
        yield_values: bool = True,
        loop: bool = False,
    ) -> Generator[xr.Dataset | compute.DelayedDataset | None]:
        """run every pass, then compute and save the corrections.

        The yields are those of `iterate_captures` over each pass. The corrections
        are saved only when the sink is the `MemoryCacheSink` that `build_sink`
        returns; an `extensions.sink` override collects nothing to save.

        Raises:
            ValueError: if `loop` is set, since a calibration runs its grid once
            RuntimeError: if the operator rejects the ENR confirmation
        """
        if loop:
            raise ValueError('a calibration does not loop')

        self.confirm_enr()

        source_id = resources['source'].source_id
        pass_sweeps = {port: self.pass_sweep(port) for port in self.ports()}
        count = 2 * sum(
            len(specs.sequencing.loop_captures(sweep, source_id))
            for sweep in pass_sweeps.values()
        )
        indexer = core._AcquisitionIndexer(count)

        for port, diode_on in self.passes():
            self.switch_diode(port, diode_on)
            inner = Resources(resources, sweep_spec=pass_sweeps[port])
            yield from core.iterate_captures(
                inner,
                extra_data=self.extra_data,
                indexer=indexer,
                always_yield=always_yield,
                yield_values=yield_values,
            )

        sink = resources['sink']
        if isinstance(sink, MemoryCacheSink):
            save_yfactor_corrections(
                sink.pop(),
                self.calibration_fields(),
                self.corrections_path(resources['format_path']),
            )

    def confirm_enr(self) -> None:
        """ask the operator to confirm the ENR of `spec.calibration`.

        Raises:
            RuntimeError: if the operator answers no
        """
        enr = self.spec.calibration.enr
        prompt = f'Confirm that the noise diode ENR is {enr} dB (y/n): '
        while True:
            response = sa.util.blocking_input(prompt)
            if response.lower() == 'y':
                return
            elif response.lower() == 'n':
                raise RuntimeError('Set the proper noise diode ENR and run again')

    def switch_diode(self, port: Any, diode_on: bool) -> None:
        """prompt the operator to set the noise diode at `port` and record the state."""
        what = f'noise diode at port {port}'
        if diode_on:
            sa.util.blocking_input(f'enable {what} and press enter: ')
        else:
            sa.util.blocking_input(f'disable {what} and press enter: ')
        self._diode_on = diode_on
