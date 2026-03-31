"""evaluate xarray datasets from sensor (meta)data and calibrations"""

from __future__ import annotations as __

from typing import Any, TYPE_CHECKING
from ... import specs
import striqt.analysis as sa

if TYPE_CHECKING:
    from ..typing import SC, SS, SP, WarmupSweep


def sweep_touches_gpu(sweep: specs.Sweep) -> bool:
    """returns True if the specified sweep uses the GPU"""

    if sweep.source.array_backend == 'numpy':
        return False

    if sweep.source.calibration is not None:
        return True

    analysis_dict = sweep.analysis.to_dict()
    if tuple(analysis_dict.keys()) != (sa.measurements.iq_waveform.__name__,):
        # everything except iq_clipping requires a warmup
        return True

    # check the inner loop (explicit) values
    for capture in sweep.captures:
        if capture.host_resample or capture.analysis_bandwidth is not None:
            return True

    # check any values specified in outer loops
    for loop in sweep.loops:
        if loop.isin == 'analysis':
            continue
        if loop.field == 'host_resample' and True in loop.get_points():
            return True
        elif loop.field == 'analysis_bandwidth' and not all(loop.get_points()):
            return True

    return False


def build_warmup_sweep(sweep: specs.Sweep[SS, SP, SC], count: int = 1) -> WarmupSweep:
    """derive a warmup sweep specification derived from sweep.

    This is meant to trigger expensive python imports and warm up JIT caches. The goal
    is to avoid analysis slowdowns later during the execution of the first captures.

    The derived sweep has the following characteristics:
        - It is bound to NoPeripheral and NoSink
        - It contains only one capture, with no loops
    """

    from ..bindings import mock_binding
    from ...bindings import warmup

    # introspect the maximum number of ports used in the sweep
    ports: list[Any] = [c.port for c in sweep.captures]
    max_rx_ports = 0
    for loop in sweep.loops:
        if loop.isin == 'capture' and loop.field == 'port':
            ports.extend(loop.get_points())
    for port in ports:
        if port is None:
            continue
        if isinstance(port, tuple):
            n = max(port)
        else:
            n = port
        if n > max_rx_ports:
            max_rx_ports = n

    # then build up the warmup sweep
    b = mock_binding(sweep._bindings__, 'warmup', register=False)

    source = warmup.schema.source(
        num_rx_ports=max_rx_ports,
        master_clock_rate=sweep.source.master_clock_rate,
        trigger_strobe=None,
        signal_trigger=sweep.source.signal_trigger,
    )

    sweep_spec = b.sweep_spec(
        source=source,
        captures=sweep.captures,
        loops=sweep.loops,
        analysis=sweep.analysis,
        sink=sweep.sink,
    )

    captures = specs.helpers.loop_captures(sweep_spec, limit=count)

    return sweep_spec.replace(captures=captures, loops=())
