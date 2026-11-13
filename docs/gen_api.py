"""generate the API reference pages under ``docs/api`` from the public striqt namespaces.

Each page lists one public module. A member is a public name of that module (a class,
function, or instance defined under ``striqt``) whose defining module is the page module
or one of its non-public submodules; a name defined under another page module is listed
on that page instead. The members of a group are ordered by kind (classes, functions,
instances) and then by name. Running the module writes the pages under ``docs/api``,
or under ``--out``; the ``docs`` and ``docs-serve`` pixi tasks run it before the site
build, so the pages are not tracked.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import sys
import typing
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
API_DIR = ROOT / 'docs' / 'api'

Selector = Callable[[str, Any], bool]


@dataclasses.dataclass(frozen=True)
class Group:
    heading: Optional[str]
    """the ``##`` heading, or None for the single group of a page"""
    desc: str
    """one sentence under the heading"""
    select: Optional[Selector] = None
    """membership test, or None for the members no other group selects"""


@dataclasses.dataclass(frozen=True)
class Page:
    path: str
    """path under ``docs/api``"""
    module: str
    """the public module the page lists"""
    groups: tuple[Group, ...]
    show_if_no_docstring: bool = False
    """render the members without a docstring (module attributes of bindings)"""
    exclude: tuple[str, ...] = ()
    """public names left off the page (re-exports of other packages)"""


def _origin(obj: Any) -> str:
    """return the defining module of a class, function, or instance."""
    if inspect.isclass(obj) or inspect.isroutine(obj):
        return getattr(obj, '__module__', '') or ''
    return type(obj).__module__


def _in_submodule(obj: Any, *parts: str) -> bool:
    return any(f'.lib.{part}' in _origin(obj) for part in parts)


def _is_measurement(name: str, obj: Any) -> bool:
    from striqt.analysis import registry

    return any(info.func is obj for info in registry.values())


def _is_acquisition_spec(name: str, obj: Any) -> bool:
    import dataclasses as dc

    from striqt.sensor import specs

    bases = (
        specs.Source,
        specs.Sink,
        specs.Peripherals,
        specs.AcquisitionInfo,
        specs.SourceInfo,
        specs.SharedPlotOptions,
        specs.DataOptions,
    )
    if inspect.isclass(obj) and issubclass(obj, bases):
        return True
    return dc.is_dataclass(obj) and name != 'Schema'


PAGES: tuple[Page, ...] = (
    Page(
        'waveform/index.md',
        'striqt.waveform',
        (
            Group(
                'Arrays and backends',
                'Array-namespace dispatch between numpy and cupy, axis helpers, and the '
                'cupy configuration.',
                lambda n, o: _in_submodule(o, 'arrays'),
            ),
            Group(
                'Fourier analysis and resampling',
                'FFT and STFT evaluation, window design, overlap-add convolution, and '
                'rational resampling.',
                lambda n, o: _in_submodule(o, 'fourier'),
            ),
            Group(
                'Power',
                'Power detectors, empirical distributions, and conversions between dB, '
                'power, and envelope.',
            ),
        ),
    ),
    Page(
        'waveform/units.md',
        'striqt.waveform.units',
        (
            Group(
                None,
                'Unit strings for power, amplitude, and dB quantities, and their SI '
                'prefixes.',
            ),
        ),
    ),
    Page(
        'analysis/index.md',
        'striqt.analysis',
        (
            Group(
                'Evaluation and storage',
                'Evaluation of measurement specs against IQ captures into xarray '
                'objects, the measurement registry, and zarr stores.',
            ),
            Group(
                'Measurements',
                'The registered measurements, each evaluated from an IQ capture into '
                'one `xarray.DataArray`.',
                _is_measurement,
            ),
        ),
    ),
    Page(
        'analysis/specs.md',
        'striqt.analysis.specs',
        (
            Group(
                None,
                'Spec structs for captures, measurement parameters, and analysis '
                'groups, which also define the YAML, JSON, and zarr schema.',
            ),
        ),
        exclude=('frozendict',),
    ),
    Page(
        'analysis/testing.md',
        'striqt.analysis.testing',
        (
            Group(
                None,
                'Synthetic IQ generators with known spectra for tests and examples.',
            ),
        ),
    ),
    Page(
        'sensor/index.md',
        'striqt.sensor',
        (
            Group(
                'Running sweeps',
                'Sweep execution against a bound sensor, and readers for sweep specs '
                'and calibration data.',
            ),
        ),
    ),
    Page(
        'sensor/specs.md',
        'striqt.sensor.specs',
        (
            Group(
                'Sweeps and captures',
                'Spec structs for sweeps, actions, capture settings, and the loops '
                'that expand one capture into several.',
            ),
            Group(
                'Sources, sinks, and peripherals',
                'Spec structs for IQ sources, data sinks, peripherals, and the '
                'acquisition metadata recorded with each capture.',
                _is_acquisition_spec,
            ),
        ),
    ),
    Page(
        'sensor/bindings.md',
        'striqt.sensor.bindings',
        (
            Group(
                None,
                'Sensor bindings, which pair a sweep spec type with the source, sink, '
                'and peripheral implementations that run it.',
            ),
        ),
        show_if_no_docstring=True,
    ),
    Page(
        'sensor/calibration.md',
        'striqt.sensor.calibration',
        (
            Group(
                None,
                'Y-factor calibration corrections and their lookup by capture settings.',
            ),
        ),
    ),
    Page(
        'sensor/sinks.md',
        'striqt.sensor.sinks',
        (Group(None, 'Sinks that receive the analyzed captures of a sweep.'),),
    ),
    Page(
        'sensor/peripherals.md',
        'striqt.sensor.peripherals',
        (
            Group(
                None, 'Peripheral controllers that a sweep sets up around each capture.'
            ),
        ),
    ),
    Page(
        'figures/index.md',
        'striqt.figures',
        (
            Group(
                'Plotting data variables',
                'One plot function per measurement data variable, and the axis and '
                'legend labels derived from dataset coordinates.',
            ),
        ),
    ),
    Page(
        'figures/backend.md',
        'striqt.figures.backend',
        (
            Group(
                None,
                'The plot backend protocol and its matplotlib and plotly '
                'implementations.',
            ),
        ),
    ),
)

PAGE_MODULES = tuple(page.module for page in PAGES)


def _is_public_object(name: str, obj: Any) -> bool:
    """return whether `obj` is a class, function, or instance defined under striqt.

    A typing construct (an alias, a `TypeVar`, a `Literal`) is not; a class with a
    typing metaclass (a `TypedDict`, a `Protocol` subclass) is.
    """
    if name.startswith('_') or isinstance(obj, ModuleType):
        return False
    if not inspect.isclass(obj) and (
        type(obj).__module__ == 'typing' or typing.get_origin(obj) is not None
    ):
        return False
    return _origin(obj).startswith('striqt.')


def _owner(obj: Any) -> str:
    """return the page module under which the defining module of `obj` falls."""
    origin = _origin(obj)
    owners = [m for m in PAGE_MODULES if origin == m or origin.startswith(m + '.')]
    return max(owners, key=len) if owners else ''


def _kind(obj: Any) -> int:
    """return the sort rank of `obj`: classes, then functions, then instances."""
    if inspect.isclass(obj):
        return 0
    return 1 if inspect.isroutine(obj) else 2


def _members(page: Page, listed: dict[int, str]) -> dict[str, Any]:
    """return the public names of the page module listed on `page`.

    An object bound to several names is listed under its first name in the module,
    and an object already in `listed` (keyed by id, from an earlier page) is skipped.
    The names are ordered by kind (classes, functions, instances), then alphabetically.
    """
    module = importlib.import_module(page.module)
    members: dict[str, Any] = {}
    for name, obj in vars(module).items():
        if name in page.exclude or not _is_public_object(name, obj):
            continue
        owner = _owner(obj)
        if (
            owner
            and owner != page.module
            and getattr(importlib.import_module(owner), name, None) is obj
        ):
            continue
        if id(obj) in listed:
            continue
        listed[id(obj)] = f'{page.module}.{name}'
        members[name] = obj
    return dict(sorted(members.items(), key=lambda kv: (_kind(kv[1]), kv[0].lower())))


def _grouped(page: Page, listed: dict[int, str]) -> list[tuple[Group, list[str]]]:
    members = _members(page, listed)
    groups: list[tuple[Group, list[str]]] = [(g, []) for g in page.groups]
    catch_all = [names for g, names in groups if g.select is None]
    for name, obj in members.items():
        for group, names in groups:
            if group.select is not None and group.select(name, obj):
                names.append(name)
                break
        else:
            catch_all[0].append(name)
    return groups


def _directive(page: Page, names: list[str], *, module_docstring: bool) -> str:
    lines = [f'::: {page.module}', '    options:']
    if not module_docstring:
        lines += ['      extra:', '        module_docstring: false']
    if page.show_if_no_docstring:
        lines.append('      show_if_no_docstring: true')
    lines.append('      members:')
    lines += [f'        - {name}' for name in names]
    return '\n'.join(lines) + '\n'


def render(page: Page, listed: dict[int, str]) -> str:
    """return the Markdown of `page`, adding its objects to `listed`."""
    groups = _grouped(page, listed)
    parts = [f'# {page.module}\n']
    if len(groups) == 1:
        group, names = groups[0]
        parts.extend((
            f'{group.desc}\n',
            _directive(page, names, module_docstring=True),
        ))
        return '\n'.join(parts)

    parts.append(f'::: {page.module}\n    options:\n      members: false\n')
    for group, names in groups:
        parts.extend((
            f'## {group.heading}\n',
            f'{group.desc}\n',
            _directive(page, names, module_docstring=False),
        ))
    return '\n'.join(parts)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--out',
        type=Path,
        default=API_DIR,
        help='the directory that receives the pages (default: docs/api)',
    )
    args = parser.parse_args(argv)

    listed: dict[int, str] = {}
    for page in PAGES:
        path = args.out / page.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render(page, listed))
    return 0


if __name__ == '__main__':
    sys.exit(main())
