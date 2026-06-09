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
import inspect
import sys

from typing_extensions import ParamSpec

from .controller import Controller, bind_controller
from .peripherals import NoPeripherals, PeripheralsBase
from . import sinks, util
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


registry: dict[str, 'type[Controller[Any, Any, Any, Any, Any]]'] = {}
tagged_sweeps: type[specs.Sweep] | None = None


@dataclasses.dataclass()
class Sensor(Generic[SS, SP, SC]):
    source_cls: type[SourceBackend[SS, SC]]
    sink_cls: type[sinks.SinkBase[SC]] = sinks.ZarrCaptureSink
    sweep_spec_cls: type[specs.Sweep[SS, SP, SC]] = specs.Sweep
    peripherals_cls: type[Peripherals[SP, SC]] = NoPeripherals

    def __post_init__(self):
        assert issubclass(self.source_cls, SourceBackend)
        assert issubclass(self.sweep_spec_cls, specs.Sweep)
        assert issubclass(self.peripherals_cls, Peripherals)
        assert issubclass(self.sink_cls, sinks.SinkBase)


@dataclasses.dataclass()
class SensorBinding(Sensor[SS, SP, SC]):
    # schema: specs.Schema[SS, SP, SC, PS, PC]
    sweep_spec_cls: type[BoundSweep[SS, SP, SC]]  # pyright: ignore

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.sweep_spec_cls, type) and (
            self.sweep_spec_cls,
            BoundSweep,
        )


def bind_sensor(
    key: str,
    sensor: Sensor[TS2, TP2, TC2],
    schema: specs.Schema[SS, SP, SC, PS, PC],
    register: bool = True,
) -> type[Controller[SS, SP, SC, PS, PC]]:
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
        source_cls=cast(type[SourceBackend[SS, SC]], sensor.source_cls),
        sweep_spec_cls=sensor.sweep_spec_cls,  # type: ignore
        peripherals_cls=sensor.peripherals_cls,  # pyright: ignore
        sink_cls=cast(type[sinks.SinkBase[SC]], sensor.sink_cls),
    )

    schema_ = schema

    class BoundSweep(sensor.sweep_spec_cls, frozen=True, kw_only=True):  # ty: ignore
        sensor = binding
        schema = schema_

        mock_source: Optional[str] = None
        source: schema.source = msgspec.field(
            default_factory=schema.source  # ty: ignore
        )
        captures: tuple[schema.capture, ...] = ()
        peripherals: schema.peripherals = msgspec.field(
            default_factory=schema.peripherals
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

    BoundSweep = _subclass_with_tag(key, BoundSweep, specs.SWEEP_TAG_FIELD)  # type: ignore
    binding = dataclasses.replace(binding, sweep_spec_cls=BoundSweep)

    global tagged_sweeps
    if tagged_sweeps is None:
        tagged_sweeps = BoundSweep
    else:
        tagged_sweeps = Union[tagged_sweeps, BoundSweep]  # pyright: ignore

    cls = bind_controller(cast(SensorBinding[SS, SP, SC], binding), schema)
    cls.__module__ = sys._getframe(1).f_globals.get('__name__') or schema.__module__

    if register:
        registry[key] = cls

    return cls


def get_registry() -> dict[str, 'type[Controller[Any, Any, Any, Any, Any]]']:
    return dict(registry)


@sw.util.lru_cache()
def mock_binding(
    origin: type[Controller], target: str | type[Controller], register: bool = True
) -> type[Controller[SS, SP, SC, PS, PC]]:
    mock_name = f'mock_{target}_{origin.sensor.sweep_spec_cls.__name__}'

    if isinstance(target, str):
        ctrl_cls = get_controller(target)
    else:
        ctrl_cls = target

    return bind_sensor(
        mock_name,
        Sensor(
            source_cls=ctrl_cls.sensor.source_cls,
            peripherals_cls=ctrl_cls.sensor.peripherals_cls,
            sweep_spec_cls=origin.sensor.sweep_spec_cls,
            sink_cls=ctrl_cls.sensor.sink_cls,
        ),
        specs.Schema(
            source=ctrl_cls.schema.source,
            capture=origin.schema.capture,
            peripherals=origin.schema.peripherals,
            arm_like=origin.schema.arm_like,
            init_like=ctrl_cls.schema.init_like,
        ),
        register=register,
    )


def get_controller(
    key: str | specs.Sweep, mock_source: str | None = None
) -> type[Controller]:
    if isinstance(key, specs.Sweep):
        ctrl_cls = registry[type(key).__name__]
    elif isinstance(key, str):
        ctrl_cls = registry[key]
    else:
        raise TypeError('key must be a string')

    if mock_source is not None:
        return mock_binding(ctrl_cls, mock_source)
    else:
        return ctrl_cls


def get_tagged_sweep_type() -> type[specs.Sweep]:
    """return a tagged union type that msgspec can decode"""
    if tagged_sweeps is None:
        raise TypeError('no bindings have been defined')
    return tagged_sweeps  # pyright: ignore


def _subclass_with_tag(
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
