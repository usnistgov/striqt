import typing
from dataclasses import dataclass

import msgspec

from .peripherals import NoPeripherals, PeripheralsBase
from .sources import SourceBase
from .specs import _TC, _TS, _TP, WaveformCaptureSpec, SourceSpec, SweepSpec


def _tagged_sweep_subclass(name: str, cls: type[SweepSpec]) -> type[SweepSpec]:
    """build a subclass of binding.sweep_spec for use in a tagged union"""
    kls = msgspec.defstruct(
        name,
        (),
        bases=(cls,),
        frozen=True,
        forbid_unknown_fields=True,
        cache_hash=True,
        tag_field='sensor_bindings',
        kw_only=True,
    )

    return typing.cast(type[SweepSpec], kls)


@dataclass(kw_only=True, frozen=True)
class SensorBinding(typing.Generic[_TS, _TP, _TC]):
    source_spec: type[_TS]
    capture_spec: type[_TC]
    source: type[SourceBase[_TS, _TC]]
    peripherals: type[PeripheralsBase[_TP, _TC]] = NoPeripherals
    sweep_spec: type[SweepSpec[_TS, _TP, _TC]] = SweepSpec

    def __post_init__(self):
        assert issubclass(self.source_spec, SourceSpec)
        assert issubclass(self.capture_spec, WaveformCaptureSpec)
        assert issubclass(self.source, SourceBase)
        assert issubclass(self.sweep_spec, SweepSpec)
        assert issubclass(self.peripherals, PeripheralsBase)


registry: dict[str, SensorBinding[typing.Any, typing.Any, typing.Any]] = {}
tagged_sweep_spec_type = _tagged_sweep_subclass('SweepSpec', SweepSpec)


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
    registry[key] = sensor

    global tagged_sweep_spec_type
    tagged_cls = _tagged_sweep_subclass(key, sensor.sweep_spec)
    tagged_sweep_spec_type = typing.Union[tagged_sweep_spec_type, tagged_cls]

    return sensor


def get_registry() -> dict[str, SensorBinding]:
    return dict(registry)


def get_binding(key: str | SweepSpec) -> SensorBinding:
    if isinstance(key, str):
        return registry[key]
    elif not isinstance(key, SweepSpec):
        raise TypeError('key must be a string or a SweepSpec')
    elif key is SweepSpec:
        raise TypeError('must provide a tagged SweepSpec')
    else:
        return registry[type(key).__name__]


def get_tagged_sweep_spec() -> type:
    """return a tagged union type that msgspec can decode"""
    return tagged_sweep_spec_type
