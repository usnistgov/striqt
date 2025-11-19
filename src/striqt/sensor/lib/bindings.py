import typing
import dataclasses

import msgspec

from .peripherals import NoPeripherals, PeripheralsBase
from .sources import SourceBase
from . import specs, sinks
from .specs import _TC, _TS, _TP


def tagged_subclass(name: str, cls: type[specs.Sweep], tag_field: str) -> type[specs.Sweep]:
    """build a subclass of binding.sweep_spec for use in a tagged union"""
    kls = msgspec.defstruct(
        name,
        (),
        bases=(cls,),
        frozen=True,
        forbid_unknown_fields=True,
        cache_hash=True,
        tag_field=tag_field,
        kw_only=True,
    )

    return typing.cast(type[specs.Sweep], kls)


@dataclasses.dataclass(kw_only=True, frozen=True)
class SensorBinding(typing.Generic[_TS, _TP, _TC]):
    source_spec: type[_TS]
    capture_spec: type[_TC]
    peripherals_spec: type[_TP]
    source: type[SourceBase[_TS, _TC]]
    peripherals: type[PeripheralsBase[_TP, _TC]] = NoPeripherals
    sweep_spec: type[specs.Sweep[_TS, _TP, _TC]] = specs.Sweep[_TS, _TP, _TC]
    sink: type[sinks.SinkBase[_TC]] | None = None

    def __post_init__(self):
        assert issubclass(self.source_spec, specs.Source)
        assert issubclass(self.capture_spec, specs.ResampledCapture)
        assert issubclass(self.source, SourceBase)
        assert issubclass(self.sweep_spec, specs.Sweep)
        assert issubclass(self.peripherals, PeripheralsBase)
        assert self.sink is None or issubclass(self.sink, sinks.SinkBase)


registry: dict[str, SensorBinding[typing.Any, typing.Any, typing.Any]] = {}
tagged_sweep_spec_type = tagged_subclass('SweepSpec', specs.Sweep, 'sensor_binding')


def bind_sensor(
    key: str, sensor: SensorBinding[_TS, _TP, _TC]
) -> SensorBinding[_TS, _TP, _TC]:
    """register a binding between specifications and controller classes.

    Args:
        key: the key used when instantiating from yaml/json
        sensor: the binding classes
    """
    if not isinstance(sensor, SensorBinding):
        raise TypeError('sensor argument must be a SensorBindings instance')

    class BoundSweep(sensor.sweep_spec, frozen=True, kw_only=True, **specs.kws):
        __bindings__ = sensor

    BoundSweep.__qualname__ = sensor.sweep_spec.__qualname__
    BoundSweep.__name__ = sensor.sweep_spec.__name__
    BoundSweep.__module__ = sensor.sweep_spec.__module__

    sensor = dataclasses.replace(sensor, sweep_spec=BoundSweep)
    registry[key] = sensor

    global tagged_sweep_spec_type
    tagged_cls = tagged_subclass(key, sensor.sweep_spec, 'sensor_binding')
    tagged_sweep_spec_type = typing.Union[tagged_sweep_spec_type, tagged_cls]

    return sensor


def get_registry() -> dict[str, SensorBinding]:
    return dict(registry)


def get_binding(key: str | specs.Sweep) -> SensorBinding:
    if isinstance(key, str):
        return registry[key]
    elif not isinstance(key, specs.Sweep):
        raise TypeError('key must be a string or a SweepSpec')
    elif key is specs.Sweep:
        raise TypeError('must provide a tagged SweepSpec')
    else:
        return registry[type(key).__name__]


def get_tagged_sweep_spec() -> type[msgspec.Struct]:
    """return a tagged union type that msgspec can decode"""
    return tagged_sweep_spec_type
