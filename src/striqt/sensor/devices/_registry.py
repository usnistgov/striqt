from ..lib import specs, sources
import typing


class RegistryItem(typing.TypedDict):
    setup: type[specs.RadioSetup]
    capture: type[specs.RadioCapture]
    sweep: type[specs.Sweep]
    source: type[sources.SourceBase]


registry: dict[str, RegistryItem] = {}


def register_source(
    name: str,
    *,
    setup_cls: type[specs.RadioSetup],
    capture_cls: type[specs.RadioCapture],
    source_cls: type[sources.SourceBase],
    sweep_cls: type[specs.Sweep]=specs.Sweep,
    schema: typing.Any = None,
    **setup_defaults: dict[str, typing.Any],
):
    registry[name] = RegistryItem(capture=capture_cls, setup=setup_cls, source=source_cls, sweep=sweep_cls)


def lookup(name: str) -> RegistryItem:
    return registry[name]
