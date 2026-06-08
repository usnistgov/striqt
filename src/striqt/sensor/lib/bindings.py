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

from .controller import Controller
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
    source_cls: type[SourceBackend[SS, SC]]
    sweep_spec_cls: type[specs.Sweep[SS, SP, SC]]
    peripherals_cls: type[Peripherals[SP, SC]]
    sink_cls: type[sinks.SinkBase[SC]]

    def __post_init__(self):
        assert issubclass(self.source_cls, SourceBackend)
        assert issubclass(self.sweep_spec_cls, specs.Sweep)
        assert issubclass(self.peripherals_cls, Peripherals)
        assert issubclass(self.sink_cls, sinks.SinkBase)


def sensor(
    *,
    source_cls: type[SourceBackend[SS, SC]],
    sweep_spec_cls: type[specs.Sweep[SS, SP, SC]] = specs.Sweep,
    peripherals_cls: type[Peripherals[SP, SC]] = NoPeripherals,
    sink_cls: type[sinks.SinkBase[SC]] = sinks.ZarrCaptureSink,
) -> Sensor[SS, SP, SC]:
    return Sensor(**locals())


@dataclasses.dataclass()
class SensorBinding(Sensor[SS, SP, SC], Generic[SS, SP, SC, PS, PC]):
    schema: specs.Schema[SS, SP, SC, PS, PC]
    sweep_spec_cls: type[BoundSweep[SS, SP, SC]]  # pyright: ignore

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.schema, specs.Schema)

    @util.cached_property
    def controller(self) -> type[Controller[SS, SP, SC, PS, PC]]:
        class C(Controller):
            _binding = self
            schema = self.schema

        self_cls = type(self)
        C.__module__ = self_cls.__module__
        C.__name__ = f'{self_cls.__name__}.controller'
        C.__qualname__ = f'{self_cls.__qualname__}.controller'
        source_spec = self.schema.source
        spec_longname = f'{source_spec.__module__}.{source_spec.__qualname__}'
        C.__doc__ = '\n\n'.join((
            Controller.__doc__ or '',
            self.source_cls.__doc__ or '',
            f"Parameters:\n   See :class:`{spec_longname}`"
        ))

        C.__signature__ = inspect.signature(self.schema.source) # ty: ignore

        return C

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
        schema=schema,  # pyright: ignore
    )

    class BoundSweep(sensor.sweep_spec_cls, frozen=True, kw_only=True):  # ty: ignore
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
    binding = dataclasses.replace(binding, sweep_spec_cls=BoundSweep)

    if register:
        registry[key] = binding

    global tagged_sweeps
    if tagged_sweeps is None:
        tagged_sweeps = BoundSweep
    else:
        tagged_sweeps = Union[tagged_sweeps, BoundSweep]  # pyright: ignore

    cls = binding.controller
    cls.__module__ = sys._getframe(1).f_globals.get('__name__') or schema.__module__
    return cls  # type: ignore


def get_registry() -> dict[str, SensorBinding]:
    return dict(registry)


@sw.util.lru_cache()
def mock_binding(
    origin: SensorBinding, target: str | SensorBinding, register: bool = True
) -> type[Controller[SS, SP, SC, PS, PC]]:
    mock_name = f'mock_{target}_{origin.sweep_spec_cls.__name__}'

    if isinstance(target, str):
        mock_binding = get_binding(target)
    else:
        mock_binding = target

    return bind_sensor(
        mock_name,
        Sensor(
            source_cls=mock_binding.source_cls,
            peripherals_cls=mock_binding.peripherals_cls,
            sweep_spec_cls=origin.sweep_spec_cls,
            sink_cls=mock_binding.sink_cls,
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
        return mock_binding(binding, mock_source)._binding
    else:
        return binding


def get_tagged_sweep_type() -> type[specs.Sweep]:
    """return a tagged union type that msgspec can decode"""
    if tagged_sweeps is None:
        raise TypeError('no bindings have been defined')
    return tagged_sweeps  # pyright: ignore
