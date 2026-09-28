"""`str.format` fields for sink and calibration paths in a sweep specification.

The fields come from the sweep, the open source and the fixed `adjust_captures`
values, so a path is resolved only once the source id is known; `PathFormatter`
defers that lookup to the call."""

from __future__ import annotations as __

import string
from typing import Callable
from datetime import datetime
from pathlib import Path

import striqt.analysis as sa

from . import sequencing, structs


@sa.util.lru_cache()
def get_format_fields(s: str, exclude: tuple[str, ...] = ()) -> list[str]:
    """list the field names of a `str.format` template in order of use.

    An auto-numbered ``{}`` appears as ``''``, a repeated field appears once per use,
    and names in `exclude` are skipped.
    """
    formatter = string.Formatter()
    return [
        name
        for _, name, *_ in formatter.parse(s)
        if name is not None and name not in exclude
    ]


@sa.util.lru_cache()
def get_path_fields(
    sweep: structs.Sweep,
    *,
    source_id: str | Callable[[], str],
    spec_path: Path | str | None = None,
) -> dict[str, str]:
    """return a mapping for string `'{field_name}'.format()` style mapping values"""

    assert isinstance(sweep, structs.Sweep)

    if isinstance(source_id, str):
        id_ = source_id
    else:
        id_ = source_id()

    fields = {}
    fields['start_time'] = datetime.now().strftime('%Y%m%d-%Hh%Mm%S')
    fields['sensor_binding'] = type(sweep).__name__
    if spec_path is not None:
        fields['spec_name'] = Path(spec_path).stem
        fields['parent_name'] = Path(spec_path).parent.absolute().name
    fields['source_id'] = id_

    labels = sequencing._get_source_capture_adjustments(sweep.adjust_captures, id_)
    fields.update(labels)

    return fields


class PathFormatter:
    def __init__(
        self,
        sweep: structs.Sweep,
        spec_path: Path | str | None = None,
        id_timeout: float = 5,
    ):
        self.sweep_spec = sweep
        if spec_path is None:
            self.spec_path = spec_path
        else:
            self.spec_path = Path(spec_path).resolve()
        self.id_timeout = id_timeout

    def __call__(self, path: str | Path) -> str:
        path_fields = get_format_fields(str(path))
        if len(path_fields) == 0:
            return str(path)

        from ..lib.controller import lookup

        id_ = lookup.id(self.sweep_spec.source, timeout=self.id_timeout)
        path = Path(path).expanduser()

        fields = get_path_fields(
            self.sweep_spec, source_id=id_, spec_path=self.spec_path
        )

        try:
            path = Path(str(path).format(**fields))
        except KeyError as ex:
            key = ex.args[0]
            available = tuple(fields.keys())
            raise KeyError(
                f'sink path format field {key!r}, only {available!r} are allowed'
            ) from ex

        return str(path)
