"""binding of a sensor's implementation classes to its spec types.

Each binding is a `Controller` subclass paired with a tagged action spec class that
`msgspec.defstruct` builds. The registry of bindings and the tagged union of their
spec classes are module-global, and a registered key binds once per process.
"""

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
import warnings

from .controller import Controller, bind_controller
from .peripherals import NoPeripherals
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

    `sweep_spec_cls` is the action spec type to bind. At its default of
    `specs.Sweep`, `Schema.action` selects the type. `bind_sensor` rejects a
    `sweep_spec_cls` and a `Schema.action` that are both non-default and differ.
    """

    source_cls: type[SourceBackend[SS, SC]]
    sweep_spec_cls: type[specs.ActionSpec[SS, SP]] = specs.Sweep
    peripherals_cls: type[Peripherals[SP, SC]] | type[NoPeripherals[SP, SC]] = (
        NoPeripherals
    )

    def __post_init__(self):
        assert issubclass(self.source_cls, SourceBackend)
        assert issubclass(self.sweep_spec_cls, specs.ActionSpec)
        assert issubclass(self.peripherals_cls, Peripherals)


@dataclasses.dataclass()
class SensorBinding(Sensor[SS, SP, SC]):
    """a bound `Sensor`, whose `sweep_spec_cls` is the tagged action spec class"""

    def __post_init__(self) -> None:
        super().__post_init__()
        assert isinstance(self.sweep_spec_cls, type)
        if not issubclass(self.sweep_spec_cls, specs.ActionSpec):
            raise TypeError('sweep_spec_cls is not an ActionSpec subclass')


def _resolve_action_type(
    sensor: Sensor[Any, Any, Any], schema: specs.Schema[Any, Any, Any, Any, Any]
) -> type[specs.ActionSpec[Any, Any]]:
    """return the action spec type from `schema.action` or `sensor.sweep_spec_cls`.

    The `sweep_spec_cls` of a `SensorBinding` is the tagged class of an earlier
    binding and is disregarded.

    Raises:
        TypeError: if both name a type other than `specs.Sweep` and the types differ
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
    `schema`. Its `sensor.sweep_spec_cls` is a subclass of the action spec type,
    tagged with `key` in the ``sensor_binding`` field, in which `source`,
    `peripherals`, and (when the action declares it) `captures` are annotated with
    the schema's types. Decoding a sweep file whose ``sensor_binding`` is `key`
    produces an instance of that class. The tagged class carries the class
    attributes `sensor` (the `SensorBinding`), `schema`, `controller` (the returned
    class), and `sweep_cls` (the bound `Sweep` class with the same spec types; the
    class itself when the action is a `Sweep`). A registered binding's tagged class
    is also an arm of the union returned by `get_tagged_sweep_type`; an unregistered
    one is decoded only by explicit type. Passing an existing controller's `sensor`
    with its `schema` replaced by one with another `action` binds the same hardware
    for that action.

    Args:
        key: the ``sensor_binding`` value of sweep files, and the registry key
        sensor: the implementation classes
        schema: the spec types; `schema.action` selects the action spec type
        register: if True, make the binding available by `key` to `get_controller`
        module: the `__module__` recorded on the controller class, which the API
            documentation groups by; the caller's module when None

    Raises:
        TypeError: if `sensor` or `schema` has the wrong type, if `key` is already
            registered, if `sensor.sweep_spec_cls` and `schema.action` disagree, or if
            the action's `check_schema` rejects the spec types
    """
    if not isinstance(sensor, Sensor):
        raise TypeError('sensor argument must be a Sensor instance')

    if not isinstance(schema, specs.Schema):
        raise TypeError('schema argument must be a Schema instance')

    if register and key in registry:
        raise TypeError(f'a sensor binding named {key!r} was already registered')

    action_type = _resolve_action_type(sensor, schema)
    action_type.check_schema(schema)

    binding = SensorBinding(
        source_cls=cast(type[SourceBackend[SS, SC]], sensor.source_cls),
        sweep_spec_cls=cast('type[specs.ActionSpec[SS, SP]]', action_type),
        peripherals_cls=cast(
            'type[Peripherals[SP, SC] | NoPeripherals[SP, SC]]', sensor.peripherals_cls
        ),
    )

    tagged_cls = _bind_spec_class(key, action_type, schema, binding, tag=True)

    if issubclass(action_type, specs.Sweep):
        sweep_cls = tagged_cls
    else:
        sweep_cls = _bind_spec_class(f'{key}.sweep', specs.Sweep, schema, binding)
    _set_class_attrs(tagged_cls, sweep_cls, sweep_cls=sweep_cls)

    # in place, so that the `sensor` class attribute of the bound classes and the
    # controller's `sensor` are the same object
    binding.sweep_spec_cls = tagged_cls

    cls = bind_controller(binding, schema)
    _set_class_attrs(tagged_cls, sweep_cls, controller=cls)

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


def bind_manual_yfactor_calibration(
    name: str, ctrl_cls: 'type[Controller[SS, SP, Any, PS, PC]]'
) -> 'type[Controller[SS, SP, Any, PS, PC]]':
    """bind a manual Y-factor calibration of the sensor bound as `ctrl_cls`.

    Deprecated: call `bind_sensor` with `ctrl_cls.sensor` and
    ``dataclasses.replace(ctrl_cls.schema, action=specs.ManualYFactorCalibration)``,
    which is what this function does. A `DeprecationWarning` is emitted on each call.
    The controller class records the caller's module.
    """
    warnings.warn(
        'bind_manual_yfactor_calibration is deprecated; call bind_sensor with '
        'dataclasses.replace(ctrl_cls.schema, action=specs.ManualYFactorCalibration)',
        DeprecationWarning,
        stacklevel=2,
    )
    schema = dataclasses.replace(ctrl_cls.schema, action=specs.ManualYFactorCalibration)

    return bind_sensor(
        name,
        ctrl_cls.sensor,
        cast(specs.Schema[SS, SP, Any, PS, PC], schema),
        module=sys._getframe(1).f_globals.get('__name__'),
    )


def get_registry() -> dict[str, 'type[Controller[Any, Any, Any, Any, Any]]']:
    return dict(registry)


# functools rather than sw.util.lru_cache, which clear_caches() empties. A cleared
# cache would bind a second controller and sweep class for the same origin
@functools.lru_cache
def mock_binding(
    origin: type[Controller], target: str | type[Controller], register: bool = True
) -> type[Controller[Any, Any, Any, Any, Any]]:
    """bind `origin`'s sweep and capture types to `target`'s implementation classes.

    The result decodes `origin`'s sweep files but opens `target`'s source and
    peripherals.
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
    key: str | specs.ActionSpec, mock_source: str | None = None
) -> type[Controller]:
    if isinstance(key, specs.ActionSpec):
        ctrl_cls = registry[type(key).__name__]
    elif isinstance(key, str):
        ctrl_cls = registry[key]
    else:
        raise TypeError('key must be a string')

    if mock_source is not None:
        return mock_binding(ctrl_cls, mock_source)
    else:
        return ctrl_cls


def get_tagged_sweep_type() -> type[specs.ActionSpec]:
    """return a tagged union type that msgspec can decode"""
    if tagged_sweeps is None:
        raise TypeError('no bindings have been defined')
    return tagged_sweeps


def _bind_spec_class(
    name: str,
    base: type[specs.ActionSpec[Any, Any]],
    schema: specs.Schema[Any, Any, Any, Any, Any],
    binding: SensorBinding[Any, Any, Any],
    *,
    tag: bool = False,
) -> type[specs.ActionSpec[Any, Any]]:
    """return a subclass of `base` whose spec fields are the types in `schema`.

    The class re-annotates `source` and `peripherals`, and `captures` when `base`
    declares it, adds the ``mock_source`` field, and carries `binding` and `schema`
    as its `sensor` and `schema` class attributes. With `tag`, the class is an arm
    of the ``sensor_binding`` tagged union and its tag is `name`. The constructor of
    the returned class raises `TypeError` when ``mock_source`` names no registered
    binding.
    """
    fields: list[tuple[str, Any, Any]] = [
        ('mock_source', Optional[str], None),
        ('source', schema.source, msgspec.field(default_factory=schema.source)),
        (
            'peripherals',
            schema.peripherals,
            msgspec.field(default_factory=schema.peripherals),
        ),
    ]
    if 'captures' in base.__struct_fields__:
        fields.append(('captures', tuple[schema.capture, ...], ()))

    def __post_init__(self: Any) -> None:
        if self.mock_source is not None and self.mock_source not in registry:
            raise TypeError(
                f'mock_source {self.mock_source!r}: no sensor was bound with this name. '
                f'valid binding names are {tuple(registry)!r}'
            )
        # a function defined outside a class body has no __class__ cell for super()
        super(cls, self).__post_init__()

    tag_kws: dict[str, Any] = {}
    if tag:
        tag_kws = {'tag': str, 'tag_field': specs.SWEEP_TAG_FIELD}

    cls = msgspec.defstruct(
        name,
        fields,
        bases=(base,),
        namespace={'__post_init__': __post_init__},
        frozen=True,
        kw_only=True,
        **tag_kws,
    )
    _set_class_attrs(cls, sensor=binding, schema=schema)
    return cast('type[specs.ActionSpec[Any, Any]]', cls)


def _set_class_attrs(*classes: type, **attrs: Any) -> None:
    """set the class attributes `attrs` on each of `classes`"""
    for cls in classes:
        for name, value in attrs.items():
            setattr(cls, name, value)
