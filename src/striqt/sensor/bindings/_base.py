import functools
import msgspec
import typing
from dataclasses import dataclass

from ..lib.specs import SourceSpec, CaptureSpec, SweepSpec, _TS, _TC
from ..lib.peripherals import PeripheralsBase, NoPeripherals
from ..lib.sources import SourceBase


@dataclass(kw_only=True, frozen=True)
class SensorBinding(typing.Generic[_TS, _TC]):
    source_spec: type[_TS]
    capture_spec: type[_TC]
    source: type[SourceBase[_TS, _TC]]
    peripherals: type[PeripheralsBase[_TS, _TC]] = NoPeripherals
    sweep_spec: type[SweepSpec[_TS, _TC]] = SweepSpec

    def __post_init__(self):
        assert issubclass(self.source_spec, SourceSpec)
        assert issubclass(self.capture_spec, CaptureSpec)
        assert issubclass(self.source, SourceBase)
        assert issubclass(self.sweep_spec, SweepSpec)
        assert issubclass(self.peripherals, PeripheralsBase)


_registry: dict[str, SensorBinding[typing.Any, typing.Any]] = {}


def bind(key: str, sensor: SensorBinding[_TS, _TC]) -> SensorBinding[_TS, _TC]:
    """register a binding between specifications and controller classes.

    Args:
        key: the key used when instantiating from yaml/json
        sensor: the binding classes
    """
    if not isinstance(sensor, SensorBinding):
        raise TypeError('sensor argument must be a SensorBindings instance')
    _registry[key] = sensor
    return sensor


def get(key: str | type[SweepSpec]) -> SensorBinding:
    if isinstance(key, str):
        return _registry[key]
    elif not issubclass(key, SweepSpec):
        raise TypeError('key must be a string or a SweepSpec')
    elif key is SweepSpec:
        raise TypeError('must provide a tagged SweepSpec')
    else:
        return _registry[key.__name__]


@functools.lru_cache
def _tag(name: str, binding: SensorBinding):
    """build a subclass of binding.sweep_spec for use in a tagged union"""
    return msgspec.defstruct(
        name,
        (),
        bases=(binding.sweep_spec,),
        frozen=True,
        forbid_unknown_fields=True,
        cache_hash=True,
        tag_field='bind',
        kw_only=True,
    )


def tagged_union_spec():
    """return a tagged union type that msgspec can decode"""
    types = [_tag(n, b) for n, b in _registry.items()]

    return typing.Union[*types]
