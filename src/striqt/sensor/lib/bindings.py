import typing
import dataclasses

import msgspec

from .peripherals import NoPeripherals, PeripheralsBase
from .sources import SourceBase
from . import specs, sinks
from .specs import _TC, _TS, _TP

_TC2 = typing.TypeVar('_TC2', bound=specs.ResampledCapture)
_TP2 = typing.TypeVar('_TP2', bound=specs.Peripherals)
_TS2 = typing.TypeVar('_TS2', bound=specs.Source)


def tagged_subclass(
    name: str, cls: type[specs.Sweep], tag_field: str
) -> type[specs.Sweep]:
    """build a subclass of binding.sweep_spec for use in a tagged union"""
    kls = msgspec.defstruct(
        name,
        (),
        bases=(cls,),
        frozen=True,
        forbid_unknown_fields=True,
        cache_hash=True,
        tag=str,
        tag_field=tag_field,
        kw_only=True,
    )

    return typing.cast(type[specs.Sweep], kls)


@dataclasses.dataclass(kw_only=True, frozen=True)
class Sensor(typing.Generic[_TS, _TP, _TC]):
    source: type[SourceBase[_TS, _TC]]
    sweep_spec: type[specs.Sweep[_TS, _TP, _TC]] = specs.Sweep
    peripherals: type[PeripheralsBase[_TP, _TC]] = NoPeripherals
    sink: type[sinks.SinkBase[_TC]] = sinks.ZarrCaptureSink

    def __post_init__(self):
        assert issubclass(self.source, SourceBase)
        assert issubclass(self.sweep_spec, specs.Sweep)
        assert issubclass(self.peripherals, PeripheralsBase)
        assert issubclass(self.sink, sinks.SinkBase)


@dataclasses.dataclass(kw_only=True, frozen=True)
class Schema(typing.Generic[_TS, _TP, _TC]):
    source: type[_TS]
    capture: type[_TC]
    peripherals: type[_TP]

    def __post_init__(self):
        assert issubclass(self.source, specs.Source)
        assert issubclass(self.capture, specs.ResampledCapture)
        assert issubclass(self.peripherals, specs.Peripherals)


@dataclasses.dataclass(kw_only=True, frozen=True)
class SensorBinding(typing.Generic[_TS, _TP, _TC]):
    source: type[SourceBase[_TS, _TC]]
    sweep_spec: type[specs.Sweep[_TS, _TP, _TC]] = specs.Sweep[_TS, _TP, _TC]
    peripherals: type[PeripheralsBase[_TP, _TC]] = NoPeripherals[_TP, _TC]
    sink: type[sinks.SinkBase[_TC]] | None = None
    schema: Schema[_TS, _TP, _TC]

    def __post_init__(self):
        assert issubclass(self.source, SourceBase)
        assert issubclass(self.sweep_spec, specs.Sweep)
        assert issubclass(self.peripherals, PeripheralsBase)
        assert isinstance(self.schema, Schema)
        assert self.sink is None or issubclass(self.sink, sinks.SinkBase)


registry: dict[str, SensorBinding[typing.Any, typing.Any, typing.Any]] = {}
tagged_sweep_spec_type = tagged_subclass('SweepSpec', specs.Sweep, 'sensor_binding')


def bind_sensor(
    key: str, sensor: Sensor[_TS2, _TP2, _TC2], schema: Schema[_TS, _TP, _TC]
) -> SensorBinding[_TS, _TP, _TC]:
    """register a binding between specifications and controller classes.

    Args:
        key: the key used when instantiating from yaml/json
        sensor: the binding classes
    """
    if not isinstance(sensor, Sensor):
        raise TypeError('sensor argument must be a Sensor instance')

    if not isinstance(sensor, Sensor):
        raise TypeError('schema argument must be a Sensor instance')

    if key in registry:
        raise TypeError(f'a sensor binding named {key!r} was already registered')

    binding = SensorBinding(**dataclasses.asdict(sensor), schema=schema)

    class BoundSweep(sensor.sweep_spec, frozen=True, kw_only=True, **specs.kws):
        __bindings__ = binding

        source: schema.source = msgspec.field(default_factory=schema.source)  # type: ignore
        captures: tuple[schema.capture, ...] = ()  # type: ignore
        peripherals: schema.peripherals = msgspec.field(
            default_factory=schema.peripherals
        )  # type: ignore

    BoundSweep = tagged_subclass(key, BoundSweep, 'sensor_binding')  # type: ignore
    # BoundSweep.__qualname__ = binding.sweep_spec.__qualname__
    # BoundSweep.__name__ = binding.sweep_spec.__name__
    # BoundSweep.__module__ = binding.sweep_spec.__module__
    binding = dataclasses.replace(binding, sweep_spec=BoundSweep)
    registry[key] = binding

    global tagged_sweep_spec_type
    tagged_sweep_spec_type = typing.Union[tagged_sweep_spec_type, BoundSweep]

    return binding


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
