from __future__ import annotations as __
import functools
from typing import (
    Any,
    cast,
    Generic,
    Optional,
    Union,
)
import dataclasses
import sys

from .controller import Controller, bind_controller
from .peripherals import NoPeripherals
from . import sinks
from .. import specs
from .typing import Peripherals, PC, PS, SC, SS, SP, SourceBackend, TypeVar
import msgspec

TC2 = TypeVar('TC2', bound=specs.SensorCapture)
TP2 = TypeVar('TP2', bound=specs.Peripherals)
TS2 = TypeVar('TS2', bound=specs.Source)


registry: dict[str, 'type[Controller[Any, Any, Any, Any, Any]]'] = {}
tagged_sweeps: Any = None


@dataclasses.dataclass()
class Sensor(Generic[SS, SP, SC]):
    """the implementation classes of a sensor.

    `sweep_spec_cls` is the action spec type to bind. Leaving it at the default
    `specs.Sweep` defers to `Schema.action`, and `bind_sensor` rejects a non-default
    value in both places. A binding selected this way runs the bind-time hooks of
    the action's runner; an explicit `sweep_spec_cls` is bound as given.
    """

    source_cls: type[SourceBackend[SS, SC]]
    sink_cls: type[sinks.SinkBase[SC]] = sinks.ZarrCaptureSink
    sweep_spec_cls: type[specs.Sweep[SS, SP, SC]] = specs.Sweep
    peripherals_cls: type[Peripherals[SP, SC]] | type[NoPeripherals[SP, SC]] = (
        NoPeripherals
    )

    def __post_init__(self):
        assert issubclass(self.source_cls, SourceBackend)
        assert issubclass(self.sweep_spec_cls, specs.Sweep)
        assert issubclass(self.peripherals_cls, Peripherals)
        assert issubclass(self.sink_cls, sinks.SinkBase)


@dataclasses.dataclass()
class SensorBinding(Sensor[SS, SP, SC]):
    """a `Sensor` after binding, whose `sweep_spec_cls` is the tagged sweep class"""

    def __post_init__(self) -> None:
        super().__post_init__()
        assert isinstance(self.sweep_spec_cls, type)
        if not issubclass(self.sweep_spec_cls, specs.Sweep):
            raise TypeError('sweep_spec_cls is not a Sweep subclass')


def _resolve_action_type(
    sensor: Sensor[Any, Any, Any], schema: specs.Schema[Any, Any, Any, Any, Any]
) -> type[specs.Sweep[Any, Any, Any]]:
    """return the action spec type named by `schema.action` or `sensor.sweep_spec_cls`.

    A `SensorBinding` names the tagged class of an earlier binding, which is not a
    base for another one, so its `sweep_spec_cls` is disregarded.

    Raises:
        TypeError: if both are set away from `specs.Sweep` and differ
    """
    if isinstance(sensor, SensorBinding):
        from_sensor = specs.Sweep
    else:
        from_sensor = sensor.sweep_spec_cls
    from_schema = schema.action

    if from_sensor is specs.Sweep:
        return from_schema
    if from_schema is specs.Sweep or from_schema is from_sensor:
        return from_sensor

    raise TypeError(
        f'sensor.sweep_spec_cls {from_sensor!r} and schema.action {from_schema!r} '
        'name different action spec types'
    )


def bind_sensor(
    key: str,
    sensor: Sensor[TS2, TP2, TC2],
    schema: specs.Schema[SS, SP, SC, PS, PC],
    register: bool = True,
    *,
    module: Optional[str] = None,
) -> type[Controller[SS, SP, SC, PS, PC]]:
    """bind a sensor's implementation classes to its spec types and register it.

    The returned `Controller` subclass carries `sensor` (a `SensorBinding`) and
    `schema`. Its `sensor.sweep_spec_cls` is a subclass of the action spec type
    (`schema.action`, or `sensor.sweep_spec_cls` when set) whose fields `source`,
    `captures`, and `peripherals` are annotated with the schema's types and which
    is tagged with `key` in the ``sensor_binding`` field. When the action is
    selected through `schema.action`, the `adapt_sensor` and `adapt_schema` hooks
    of its registered `ActionRunner` are applied to `sensor` and `schema` first.
    Passing an existing controller's `sensor` and `schema` with another `action`
    derives a binding of the same hardware for that action. Decoding a sweep file
    whose ``sensor_binding`` is `key` produces an instance of that class. A
    registered binding's tagged class is also an arm of the union returned by
    `get_tagged_sweep_type`; an unregistered one is decoded only by explicit type.

    Args:
        key: the ``sensor_binding`` value of sweep files, and the registry key
        sensor: the implementation classes
        schema: the spec types; `schema.action` selects the action spec type
        register: if True, make the binding available by `key` to `get_controller`
        module: the `__module__` recorded on the controller class, which the API
            documentation groups by; the caller's module when None

    Raises:
        TypeError: if `sensor` or `schema` has the wrong type, if `key` is already
            registered, or if `sensor.sweep_spec_cls` and `schema.action` disagree
    """
    if not isinstance(sensor, Sensor):
        raise TypeError('sensor argument must be a Sensor instance')

    if not isinstance(schema, specs.Schema):
        raise TypeError('schema argument must be a Schema instance')

    if register and key in registry:
        raise TypeError(f'a sensor binding named {key!r} was already registered')

    action_type = _resolve_action_type(sensor, schema)

    if isinstance(sensor, SensorBinding) or sensor.sweep_spec_cls is specs.Sweep:
        # the hooks run only when the action is selected through `schema.action`;
        # `mock_binding` passes an already adapted tagged class as `sweep_spec_cls`
        from . import actions

        runner_cls = actions.registry.lookup(action_type)
        sensor = runner_cls.adapt_sensor(sensor)
        schema = runner_cls.adapt_schema(schema)

    binding = SensorBinding(
        source_cls=cast(type[SourceBackend[SS, SC]], sensor.source_cls),
        sweep_spec_cls=cast('type[specs.Sweep[SS, SP, SC]]', action_type),
        peripherals_cls=cast(
            'type[Peripherals[SP, SC] | NoPeripherals[SP, SC]]', sensor.peripherals_cls
        ),
        sink_cls=cast(type[sinks.SinkBase[SC]], sensor.sink_cls),
    )

    schema_ = schema

    class BoundSweep(action_type, frozen=True, kw_only=True):  # ty: ignore
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
                    f'mock_source {self.mock_source!r}: no sensor was bound with this name. '
                    f'valid binding names are {tuple(registry)!r}'
                )
            super().__post_init__()

    tagged_cls = _subclass_with_tag(key, BoundSweep, specs.SWEEP_TAG_FIELD)

    # in place, so that the `sensor` class attribute of the bound sweep and the
    # controller's `sensor` are the same object
    binding.sweep_spec_cls = tagged_cls

    cls = bind_controller(binding, schema)

    if module is None:
        module = sys._getframe(1).f_globals.get('__name__') or schema.__module__
    cls.__module__ = module

    if register:
        registry[key] = cls

        global tagged_sweeps
        if tagged_sweeps is None:
            tagged_sweeps = tagged_cls
        else:
            tagged_sweeps = Union[tagged_sweeps, tagged_cls]

    return cls


def get_registry() -> dict[str, 'type[Controller[Any, Any, Any, Any, Any]]']:
    return dict(registry)


# functools rather than sw.util.lru_cache, which clear_caches() empties: a cleared
# cache would bind a second controller and sweep class for the same origin
@functools.lru_cache
def mock_binding(
    origin: type[Controller], target: str | type[Controller], register: bool = True
) -> type[Controller[Any, Any, Any, Any, Any]]:
    """bind `origin`'s sweep and capture types to `target`'s implementation classes.

    The result decodes `origin`'s sweep files but opens `target`'s source, sink, and
    peripherals. `origin`'s tagged sweep class is passed as `sweep_spec_cls`, so no
    action hooks run again.
    """
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
        module=origin.__module__,
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
    return tagged_sweeps


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
