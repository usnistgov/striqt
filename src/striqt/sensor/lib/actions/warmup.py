"""the warmup: a one-capture `NoSource` sweep run before a GPU sweep.

The warmup runs the python imports and JIT compilation of the analysis before the
first hardware capture. The warmup sweep is derived from the sweep to run, and the
resources of the warmup are opened without hardware or files.
"""

from __future__ import annotations as __

import functools
from typing import TYPE_CHECKING, Any, cast

from ... import specs
from .. import sinks
from ..compute import gpu
from ..resources import Resources
from ..typing import PC, PS, SA, SC

if TYPE_CHECKING:
    from ..controller import Controller
    from .core import ActionRunner


# functools rather than sw.util.lru_cache, which clear_caches() empties. A cleared
# cache would bind a second controller and sweep class for the same origin.
@functools.lru_cache
def warmup_binding(
    origin: type[Controller[Any, Any, Any, Any, Any, Any]],
) -> type[Controller[specs.NoSource, specs.NoPeripherals, Any, Any, Any, Any]]:
    """return the unregistered ``warmup`` binding for the capture type of `origin`.

    The binding combines the implementation classes and source spec of the
    ``warmup`` sensor with the capture, peripherals, and analysis spec types of
    `origin`. The result is memoized by `origin`.
    """
    from ...bindings import warmup
    from .. import bindings

    schema = specs.Schema(
        source=warmup.schema.source,
        init_like=warmup.schema.init_like,
        capture=origin.schema.capture,
        arm_like=origin.schema.arm_like,
        peripherals=origin.schema.peripherals,
        analysis=origin.schema.analysis,
    )
    return bindings.bind_sensor(
        f'warmup_{origin.sensor.sweep_spec_cls.__name__}',
        warmup.sensor,
        schema,
        register=False,
        module=__name__,
    )


def derive_warmup_sweep(
    sweep_spec: specs.Sweep[Any, Any, SC, SA], count: int = 1
) -> specs.Sweep[specs.NoSource, specs.NoPeripherals, SC, SA]:
    """return the warmup `Sweep` for `sweep_spec`.

    The source is a `NoSource` with ``max(port) + 1`` ports over every port that a
    capture or a `port` loop names, and the `master_clock_rate` and `signal_trigger`
    of `sweep_spec.source`. The captures are the first `count` captures of the loop
    expansion of `sweep_spec`, with no loops. `analysis` is copied. `sink` is copied
    with `path` None. The result is an instance of the sweep class of the
    `warmup_binding` of the binding of `sweep_spec`.

    Args:
        sweep_spec: the sweep to run after the warmup
        count: the number of expanded captures to keep
    """
    from .. import bindings

    binding = warmup_binding(bindings.get_controller(sweep_spec))

    # ports are numbered from 0, so the source needs max(port) + 1 ports
    ports: list[Any] = [c.port for c in sweep_spec.captures]
    max_rx_ports = 0
    for loop in sweep_spec.loops:
        if loop.isin == 'capture' and loop.field == 'port':
            ports.extend(loop.get_points())
    for port in ports:
        if port is None:
            continue
        if isinstance(port, tuple):
            n = max(port) + 1
        else:
            n = port + 1
        max_rx_ports = max(max_rx_ports, n)

    source = binding.schema.source(
        num_rx_ports=max_rx_ports,
        master_clock_rate=sweep_spec.source.master_clock_rate,
        trigger_strobe=None,
        signal_trigger=sweep_spec.source.signal_trigger,
    )

    sweep_cls = cast(
        'type[specs.Sweep[specs.NoSource, specs.NoPeripherals, SC, SA]]',
        binding.sensor.sweep_spec_cls,
    )
    looped = sweep_cls(
        source=source,
        captures=sweep_spec.captures,
        loops=sweep_spec.loops,
        analysis=sweep_spec.analysis,
        sink=sweep_spec.sink.replace(path=None),
    )

    captures = specs.sequencing.loop_captures(looped, limit=count)

    return looped.replace(captures=captures, loops=())


def warmup_sweep_for(
    sweep: specs.Sweep[Any, Any, SC, SA],
) -> specs.Sweep[Any, Any, SC, SA] | None:
    """return the warmup `Sweep` for `sweep`, or None.

    None is returned when `sweep.options.skip_warmup` is set or `sweep_touches_gpu`
    is false for `sweep`.
    """
    if sweep.options.skip_warmup or not gpu.sweep_touches_gpu(sweep):
        return None
    return derive_warmup_sweep(sweep)


def open_warmup_resources(
    runner: ActionRunner[Any, specs.NoSource, specs.NoPeripherals, SC, SA, PS, PC],
) -> Resources[specs.NoSource, specs.NoPeripherals, SC, SA, PS, PC]:
    """return the resources of a warmup `runner`, opened without hardware or files.

    The source is a `NoSource` controller for `runner.sweep.source`, the peripherals
    are `NoPeripherals`, and the sink is a `NoSink`. The ``source`` entry is returned
    unentered.
    """
    from ...bindings import warmup

    sweep = runner.sweep
    return Resources(
        action_spec=runner.spec,
        sweep_spec=sweep,
        runner=runner,
        source=warmup.from_source_spec(sweep.source),
        peripherals=warmup.sensor.peripherals_cls(sweep),
        sink=sinks.NoSink(sweep),
        calibration=None,
        format_path=None,
    )
