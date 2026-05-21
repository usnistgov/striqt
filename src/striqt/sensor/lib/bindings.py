from __future__ import annotations as __
import functools
from typing import (
    overload,
    Any,
    Callable,
    cast,
    Generic,
    Literal,
    Optional,
    Protocol,
    TYPE_CHECKING,
    Union,
)
import dataclasses
from typing_extensions import ParamSpec

from .peripherals import NoPeripherals, PeripheralsBase
from . import sinks, sources, util
from .. import specs
from .typing import Peripherals, PC, PS, SC, SS, SP, SourceBackend, TypeVar
import striqt.waveform as sw
import msgspec

TC2 = TypeVar('TC2', bound=specs.SensorCapture)
TP2 = TypeVar('TP2', bound=specs.Peripherals)
TS2 = TypeVar('TS2', bound=specs.Source)
PS2 = ParamSpec('PS2')
PC2 = ParamSpec('PC2')


class BoundSweep(specs.Sweep[SS, SP, SC], frozen=True, kw_only=True):
    mock_sensor: specs.types.MockSensor = None


registry: dict[str, 'SensorBinding[Any, Any, Any, Any, Any]'] = {}
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

    return cast(type[specs.Sweep], kls)


@dataclasses.dataclass()
class Sensor(Generic[SS, SP, SC]):
    source: type[SourceBackend[SS, SC]]
    sweep_spec: type[specs.Sweep[SS, SP, SC]]
    peripherals: type[Peripherals[SP, SC]]
    sink: type[sinks.SinkBase[SC]]

    def __post_init__(self):
        assert issubclass(self.source, SourceBackend)
        assert issubclass(self.sweep_spec, specs.Sweep)
        assert issubclass(self.peripherals, Peripherals)
        assert issubclass(self.sink, sinks.SinkBase)


def sensor(
    *,
    source: type[SourceBackend[SS, SC]],
    sweep_spec: type[specs.Sweep[SS, SP, SC]] = specs.Sweep,
    peripherals: type[Peripherals[SP, SC]] = NoPeripherals,
    sink: type[sinks.SinkBase[SC]] = sinks.ZarrCaptureSink,
) -> Sensor[SS, SP, SC]:
    return Sensor(**locals())


@dataclasses.dataclass()
class SensorBinding(Sensor[SS, SP, SC], Generic[SS, SP, SC, PS, PC]):
    schema: specs.Schema[SS, SP, SC, PS, PC]
    sweep_spec: type[BoundSweep[SS, SP, SC]]  # pyright: ignore

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.schema, specs.Schema)

    @util.cached_property
    def controller(self) -> type[sources.Controller[SS, SC, PS, PC]]:
        class Controller(sources.Controller):
            _binding = self

        self_cls = type(self)
        Controller.__module__ = self_cls.__module__
        Controller.__name__ = f'{self_cls.__name__}.controller'
        Controller.__qualname__ = f'{self_cls.__qualname__}.controller'
        return Controller

    @util.cached_property
    def _raw_controller(self) -> type[sources.RawController[SS, SC, PS, PC]]:
        class Controller(sources.RawController):
            _binding = self

        self_cls = type(self)
        Controller.__module__ = self_cls.__module__
        Controller.__name__ = f'{self_cls.__name__}.controller'
        Controller.__qualname__ = f'{self_cls.__qualname__}.controller'
        return Controller


def bind_sensor(
    key: str,
    sensor: Sensor[TS2, TP2, TC2],
    schema: specs.Schema[SS, SP, SC, PS, PC],
    register: bool = True,
) -> SensorBinding[SS, SP, SC, PS, PC]:
    """register a binding between specifications and controller classes.

    Args:
        key: the key used when instantiating from yaml/json
        sensor: the binding classes
    """
    if not isinstance(sensor, Sensor):
        raise TypeError('sensor argument must be a Sensor instance')

    if not isinstance(schema, specs.Schema):
        raise TypeError('schema argument must be a Sensor instance')

    if register and key in registry:
        raise TypeError(f'a sensor binding named {key!r} was already registered')

    binding = SensorBinding(
        source=cast(type[SourceBackend[SS, SC]], sensor.source),
        sweep_spec=sensor.sweep_spec,  # type: ignore
        peripherals=sensor.peripherals,  # pyright: ignore
        sink=cast(type[sinks.SinkBase[SC]], sensor.sink),
        schema=schema,  # pyright: ignore
    )

    class BoundSweep(sensor.sweep_spec, frozen=True, kw_only=True):  # ty: ignore
        _binding = binding

        mock_source: Optional[str] = None
        source: _binding.schema.source = msgspec.field(
            default_factory=_binding.schema.source  # ty: ignore
        )
        captures: tuple[_binding.schema.capture, ...] = ()
        peripherals: _binding.schema.peripherals = msgspec.field(
            default_factory=_binding.schema.peripherals
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

    BoundSweep = tagged_subclass(key, BoundSweep, specs.SWEEP_TAG_FIELD)  # type: ignore
    binding = dataclasses.replace(binding, sweep_spec=BoundSweep)

    if register:
        registry[key] = binding

    global tagged_sweeps
    if tagged_sweeps is None:
        tagged_sweeps = BoundSweep
    else:
        tagged_sweeps = Union[tagged_sweeps, BoundSweep]  # pyright: ignore

    return binding  # type: ignore


def get_registry() -> dict[str, SensorBinding]:
    return dict(registry)


@sw.util.lru_cache()
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
        specs.Schema(
            source=mock_binding.schema.source,
            capture=origin.schema.capture,
            peripherals=origin.schema.peripherals,
            arm_like=origin.schema.arm_like,
            init_like=mock_binding.schema.init_like,
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
    return tagged_sweeps  # pyright: ignore
