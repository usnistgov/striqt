"""bindings of sensor implementation classes to spec types.

Each binding is a `Controller` subclass paired with a tagged action spec class that
`msgspec.defstruct` builds. The registry of bindings and the tagged union of the
bound spec classes are module-global. A registered key binds once per process.
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
from .typing import Peripherals, PC, PS, SA, SC, SS, SP, SourceBackend, TypeVar
import msgspec
import striqt.analysis as sa
from striqt.analysis.lib.register import check_group_registered

TC2 = TypeVar('TC2', bound=specs.SensorCapture)
TP2 = TypeVar('TP2', bound=specs.Peripherals)
TS2 = TypeVar('TS2', bound=specs.Source)


registry: dict[str, 'type[Controller[Any, Any, Any, Any, Any, Any]]'] = {}
tagged_sweeps: Any = None


@dataclasses.dataclass()
class Sensor(Generic[SS, SP, SC]):
    """the implementation classes of a sensor"""

    source_cls: type[SourceBackend[SS, SC]]
    """the source backend class instantiated from the `source` block of a spec"""
    sweep_spec_cls: type[specs.ActionSpec[SS, SP, Any]] = specs.Sweep
    """the action spec type to bind; at the default of `specs.Sweep`, `Schema.action`
    selects the type"""
    peripherals_cls: type[Peripherals[SP, SC]] | type[NoPeripherals[SP, SC]] = (
        NoPeripherals
    )
    """the peripherals class instantiated from a `Sweep`; `NoPeripherals` opens
    nothing"""

    def __post_init__(self):
        assert issubclass(self.source_cls, SourceBackend)
        assert issubclass(self.sweep_spec_cls, specs.ActionSpec)
        assert issubclass(self.peripherals_cls, Peripherals)


@dataclasses.dataclass()
class SensorBinding(Sensor[SS, SP, SC]):
    """a bound `Sensor` with the tagged action spec class as `sweep_spec_cls`"""

    def __post_init__(self) -> None:
        super().__post_init__()
        assert isinstance(self.sweep_spec_cls, type)
        if not issubclass(self.sweep_spec_cls, specs.ActionSpec):
            raise TypeError('sweep_spec_cls is not an ActionSpec subclass')


def _resolve_action_type(
    sensor: Sensor[Any, Any, Any], schema: specs.Schema[Any, Any, Any, Any, Any, Any]
) -> type[specs.ActionSpec[Any, Any, Any]]:
    """return the action spec type from `schema.action` or `sensor.sweep_spec_cls`.

    The `sweep_spec_cls` of a `SensorBinding` is disregarded.

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
    schema: specs.Schema[SS, SP, SC, SA, PS, PC],
    register: bool = True,
    *,
    module: Optional[str] = None,
) -> type[Controller[SS, SP, SC, SA, PS, PC]]:
    """bind the implementation classes of a sensor to spec types under `key`.

    The returned `Controller` subclass has the class attributes `sensor` (a
    `SensorBinding`) and `schema`. Its `sensor.sweep_spec_cls` is a subclass of the
    action spec type, tagged with `key` in the ``sensor_binding`` field. In that
    subclass, `source`, `peripherals`, `captures` (when the action declares the
    field), and `analysis` (when the action is a `Sweep`) are annotated with the
    types in `schema`. A sweep file with ``sensor_binding`` equal to `key` decodes to
    an instance of that subclass. The tagged subclass has the class attributes
    `sensor`, `schema`, `controller` (the returned class), and `sweep_cls` (the
    bound `Sweep` class with the same spec types, or the tagged subclass when the
    action is a `Sweep`). The tagged subclass of an unregistered binding is decoded
    only by explicit type.

    Args:
        key: the ``sensor_binding`` value of sweep files, and the registry key
        sensor: the implementation classes; the `sensor` of an existing controller
            binds the same hardware for the action in `schema`
        schema: the spec types; `schema.action` selects the action spec type and
            `schema.analysis` the analysis group
        register: if True, make the binding available by `key` to `get_controller`
            and the tagged subclass an arm of the union from `get_tagged_sweep_type`
        module: the `__module__` of the returned class; the module of the caller
            when None

    Raises:
        TypeError: if `sensor` or `schema` has the wrong type, if `key` is already
            registered, if `sensor.sweep_spec_cls` and `schema.action` name
            different non-default types, if the `check_schema` of the action type
            rejects `schema`, or if a field of `schema.analysis` is not a
            registered measurement
    """
    if not isinstance(sensor, Sensor):
        raise TypeError('sensor argument must be a Sensor instance')

    if not isinstance(schema, specs.Schema):
        raise TypeError('schema argument must be a Schema instance')

    if register and key in registry:
        raise TypeError(f'a sensor binding named {key!r} was already registered')

    action_type = _resolve_action_type(sensor, schema)
    action_type.check_schema(schema)
    check_group_registered(schema.analysis, sa.registry)

    binding = SensorBinding(
        source_cls=cast(type[SourceBackend[SS, SC]], sensor.source_cls),
        sweep_spec_cls=cast('type[specs.ActionSpec[SS, SP, Any]]', action_type),
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

    # assigned in place: the bound spec classes and the controller reference this
    # same `SensorBinding` object
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
    name: str, ctrl_cls: 'type[Controller[SS, SP, Any, SA, PS, PC]]'
) -> 'type[Controller[SS, SP, Any, SA, PS, PC]]':
    """bind the sensor of `ctrl_cls` for `ManualYFactorCalibration` under `name`.

    Deprecated in favor of `bind_sensor` with `ctrl_cls.sensor` and
    ``dataclasses.replace(ctrl_cls.schema, action=specs.ManualYFactorCalibration)``.
    A `DeprecationWarning` is emitted on each call. The `__module__` of the returned
    class is the module of the caller.
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
        cast(specs.Schema[SS, SP, Any, SA, PS, PC], schema),
        module=sys._getframe(1).f_globals.get('__name__'),
    )


def get_registry() -> dict[str, 'type[Controller[Any, Any, Any, Any, Any, Any]]']:
    return dict(registry)


# functools rather than sw.util.lru_cache, which clear_caches() empties. A cleared
# cache would bind a second controller and sweep class for the same origin
@functools.lru_cache
def mock_binding(
    origin: type[Controller], target: str | type[Controller], register: bool = True
) -> type[Controller[Any, Any, Any, Any, Any, Any]]:
    """bind the spec types of `origin` to the implementation classes of `target`.

    The returned controller decodes the sweep files of `origin` and opens the source
    and peripherals of `target`. Repeated calls with equal arguments return the
    cached class.

    Args:
        origin: the controller with the sweep, capture, and peripherals spec types
        target: the registry key or controller class with the source backend
        register: if True, register the binding under ``mock_<target>_<sweep class>``
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
            analysis=origin.schema.analysis,
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
    base: type[specs.ActionSpec[Any, Any, Any]],
    schema: specs.Schema[Any, Any, Any, Any, Any, Any],
    binding: SensorBinding[Any, Any, Any],
    *,
    tag: bool = False,
) -> type[specs.ActionSpec[Any, Any, Any]]:
    """return a subclass of `base` with the spec fields typed from `schema`.

    With `tag`, the class is tagged `name` in the ``sensor_binding`` field. The
    constructor of the returned class raises `TypeError` when `mock_source` names
    no registered binding.
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
    # only `Sweep.analysis` is the analysis group. The calibration and cellular specs
    # each declare `analysis` with a different block type
    if issubclass(base, specs.Sweep):
        fields.append((
            'analysis',
            schema.analysis,
            msgspec.field(default_factory=schema.analysis),
        ))

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
    return cast('type[specs.ActionSpec[Any, Any, Any]]', cls)


def _set_class_attrs(*classes: type, **attrs: Any) -> None:
    """set the class attributes `attrs` on each of `classes`"""
    for cls in classes:
        for name, value in attrs.items():
            setattr(cls, name, value)
