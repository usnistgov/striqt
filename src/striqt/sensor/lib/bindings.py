from __future__ import annotations as __
import functools
import typing
import dataclasses

import msgspec

from .peripherals import NoPeripherals, PeripheralsBase
from .sources import SourceBase
from . import sinks
from .. import specs
from ..specs import _TC, _TS, _TP

_TC2 = typing.TypeVar('_TC2', bound=specs.CaptureResampled)
_TP2 = typing.TypeVar('_TP2', bound=specs.Peripherals)
_TS2 = typing.TypeVar('_TS2', bound=specs.Source)


if typing.TYPE_CHECKING:

    class BoundSweep(specs.Sweep, frozen=True, kw_only=True):
        mock_sensor: specs.types.MockSensor = None


registry: dict[str, 'SensorBinding[typing.Any, typing.Any, typing.Any]'] = {}
tagged_sweeps: type[specs.Sweep] | None = None


def tagged_subclass(
    name: str, cls: type[specs.Sweep], tag_field: str
) -> type[specs.Sweep]:
    """build a subclass of binding.sweep_spec for use in a tagged union"""
    kls = msgspec.defstruct(
        name,
        (),
        bases=(cls,),
        frozen=True,
        tag=str,
        tag_field=tag_field,
        kw_only=True,
    )

    return typing.cast(type[specs.Sweep], kls)


@dataclasses.dataclass(frozen=True)
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


@dataclasses.dataclass(frozen=True)
class Schema(typing.Generic[_TS, _TP, _TC]):
    source: type[_TS]
    capture: type[_TC]
    peripherals: type[_TP]

    def __post_init__(self):
        assert issubclass(self.source, specs.Source)
        assert issubclass(self.capture, specs.CaptureResampled)
        assert issubclass(self.peripherals, specs.Peripherals)


@dataclasses.dataclass(frozen=True)
class SensorBinding(Sensor[_TS, _TP, _TC]):
    schema: Schema[_TS, _TP, _TC] = None  # type: ignore
    sweep_spec: type[BoundSweep[_TS, _TP, _TC]] = specs.Sweep  # type: ignore

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.schema, Schema)


def bind_sensor(
    key: str,
    sensor: Sensor[_TS2, _TP2, _TC2],
    schema: Schema[_TS, _TP, _TC],
    register: bool = True,
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

    if register and key in registry:
        raise TypeError(f'a sensor binding named {key!r} was already registered')

    binding = SensorBinding(**dataclasses.asdict(sensor), schema=schema)

    class BoundSweep(sensor.sweep_spec, frozen=True, kw_only=True):
        __bindings__ = binding

        mock_source: typing.Optional[str] = None
        source: __bindings__.schema.source = msgspec.field(
            default_factory=__bindings__.schema.source
        )  # type: ignore
        captures: tuple[__bindings__.schema.capture, ...] = ()  # type: ignore
        peripherals: __bindings__.schema.peripherals = msgspec.field(  # type: ignore
            default_factory=__bindings__.schema.peripherals
        )

        def __post_init__(self):
            if self.mock_source is None:
                pass
            elif self.mock_source not in registry.keys():
                raise TypeError(
                    f'mock_sensor {self.mock_source!r}: no sensor was bound with this name. '
                    f'valid binding names are {tuple(registry)!r}'
                )
            super().__post_init__()

    BoundSweep = tagged_subclass(key, BoundSweep, 'sensor_binding')  # type: ignore
    binding = dataclasses.replace(binding, sweep_spec=BoundSweep)

    if register:
        registry[key] = binding

    global tagged_sweeps
    if tagged_sweeps is None:
        tagged_sweeps = BoundSweep
    else:
        tagged_sweeps = typing.Union[tagged_sweeps, BoundSweep]

    return binding


def get_registry() -> dict[str, SensorBinding]:
    return dict(registry)


@functools.cache
def mock_binding(
    origin: SensorBinding, target: str | SensorBinding, register: bool = True
) -> SensorBinding:
    mock_name = f'mock_{target}_{origin.sweep_spec.__name__}'

    if isinstance(target, str):
        mock_binding = get_binding(target)
    else:
        mock_binding = target

    return bind_sensor(
        mock_name,
        Sensor(
            source=mock_binding.source,
            peripherals=mock_binding.peripherals,
            sweep_spec=origin.sweep_spec,
            sink=mock_binding.sink,
        ),
        Schema(
            source=mock_binding.schema.source,
            capture=origin.schema.capture,
            peripherals=origin.schema.peripherals,
        ),
        register=register,
    )


def get_binding(
    key: str | specs.Sweep, mock_source: str | None = None
) -> SensorBinding:
    if isinstance(key, specs.Sweep):
        binding = registry[type(key).__name__]
    elif isinstance(key, str):
        binding = registry[key]
    else:
        raise TypeError('key must be a string')

    if mock_source is not None:
        return mock_binding(binding, mock_source)
    else:
        return binding


def get_tagged_sweep_type() -> type[specs.Sweep]:
    """return a tagged union type that msgspec can decode"""
    if tagged_sweeps is None:
        raise TypeError('no bindings have been defined')
    return tagged_sweeps  # type: ignore
