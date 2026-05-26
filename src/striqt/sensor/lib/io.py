"""just a stub for now in case we change this in the future"""

from __future__ import annotations as __

import sys
import msgspec
from pathlib import Path
from typing import Any, Optional, overload, TYPE_CHECKING

import striqt.analysis as sa

from .. import specs
from . import util

if TYPE_CHECKING:
    from .typing import ZarrStore

    import numpy as np
    import xarray as xr
else:
    np = util.lazy_import('numpy')
    xr = util.lazy_import('xarray')


def open_store(
    spec: specs.Sink,
    *,
    format_path: specs.helpers.PathFormatter | None = None,
    force=False,
) -> ZarrStore:
    spec_path = spec.path

    if format_path is not None:
        spec_path = format_path(spec_path)

    spec_path = Path(spec_path)

    if spec.store == 'directory' and not spec_path.stem.endswith('.zarr'):
        fixed_path = spec_path.with_suffix('.zarr')
    elif spec.store == 'zip' and not spec_path.stem.endswith('.zarr.zip'):
        fixed_path = spec_path.with_suffix('.zarr.zip')
    else:
        fixed_path = spec_path

    fixed_path.parent.mkdir(parents=True, exist_ok=True)
    store_backend = sa.open_store(fixed_path, mode='w' if force else 'a')
    return store_backend


def read_yaml_spec(
    path: str | Path,
    *,
    output_path: Optional[str] = None,
    store_backend: Optional[str] = None,
) -> specs.Sweep:
    """Build a Sweep specification object from the specified yaml file.

    Args:
        path: path to the yaml file
        output_path: optional override for the specification's output path
        store_backend: optional override for the specification's output store backend

    Returns:
        an instance of specs.Sweep
    """

    from .bindings import get_tagged_sweep_type

    tree = sa.lib.io.decode_from_yaml_file(path)

    if not isinstance(tree, dict):
        raise TypeError('yaml file does not specify a dict structure')

    mock_source = tree.get('mock_source', None)
    if mock_source is not None:
        assert 'sensor_binding' in tree, TypeError('missing "sensor_binding"')
        from .bindings import get_binding

        mock_name = get_binding(tree['sensor_binding'], mock_source).sweep_spec.__name__
        tree['sensor_binding'] = mock_name

    if 'extensions' in tree:
        # import now, so that sensor_binding keys can use definitions
        # in extension modules
        ext = specs.Extension.from_dict(tree['extensions'])
        _import_extensions_from_spec(ext, root_dir=Path(path).parent)

    spec = sa.specs.helpers.convert_dict(tree, type=get_tagged_sweep_type())

    sink = spec.sink
    if store_backend is not None:
        sink = sink.replace(store=store_backend)

    replace: dict[str, Any] = dict(sink=sink)
    if output_path is not None:
        replace['path'] = output_path

    return spec.replace(**replace)


def read_json_spec(
    path: str | Path,
    *,
    type:type[specs.Sweep]|None=None,
    output_path: Optional[str] = None,
    store_backend: Optional[str] = None,
) -> specs.Sweep:
    """Build a specs.Sweep specification object from a json file at the given path.
    
    This supports the same specifications as a yaml file, but without !include
    directives.

    Args:
        path: path to the yaml file
        output_path: optional override for the specification's output path
        store_backend: optional override for the specification's output store backend

    Returns:
        an instance of specs.Sweep
    """
    with open(path, 'rb') as buf:
        s = buf.read()

    if type is None:
        from .bindings import get_tagged_sweep_type
        sweep_cls = get_tagged_sweep_type()
    else:
        sweep_cls = type

    spec = msgspec.json.decode(s, type=sweep_cls, dec_hook=specs.helpers._dec_hook)

    sink = spec.sink
    if store_backend is not None:
        sink = sink.replace(store=store_backend)

    replace: dict[str, Any] = dict(sink=sink)
    if output_path is not None:
        replace['path'] = output_path

    return spec.replace(**replace)


def read_tdms_iq(
    path: Path | str,
    duration: float | None = None,
    *,
    master_clock_rate,
    dtype='complex64',
    skip_samples=0,
    array_backend: specs.types.ArrayBackend,
) -> tuple['np.ndarray', specs.FileCapture]:
    from ..bindings import tdms_file
    from .sources.file import TDMSSource

    source_spec = tdms_file.schema.source(
        master_clock_rate=master_clock_rate, path=str(path)
    )
    source = tdms_file._raw_controller(source_spec)
    source.arm(source.armed_capture)
    iq, _ = source.read_iq()

    return iq, source.armed_capture


@overload
def read_calibration(
    path: None, format_path: specs.helpers.PathFormatter | None = None
) -> None: ...


@overload
def read_calibration(
    path: str | Path, format_path: specs.helpers.PathFormatter | None = None
) -> 'xr.Dataset': ...


@sa.util.lru_cache()
def read_calibration(
    path: str | Path | None, format_path: specs.helpers.PathFormatter | None = None
) -> 'xr.Dataset|None':
    if path is None:
        return None

    if format_path is not None:
        path = format_path(path)

    return xr.open_dataset(path)


def save_calibration(path, corrections: 'xr.Dataset'):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    corrections.to_netcdf(path)


def _import_extensions_from_spec(
    spec: specs.Extension,
    format_path: specs.helpers.PathFormatter | None = None,
    root_dir: Path | str = '.',
) -> None:
    """import an extension class from a dict representation of structs.Extensions

    Arguments:
        spec: specification structure for the extension imports
        format_path: formatter to fill aliases in the import path
    """

    import importlib
    from .bindings import get_registry

    if spec.import_path is None:
        pass
    else:
        if format_path is None:
            p = Path(spec.import_path)
        else:
            p = Path(format_path(spec.import_path))

        if not p.is_absolute():
            root = Path(root_dir)
            if not root.exists():
                raise FileNotFoundError(f'root_path {root_dir!r} does not exist')
            elif not root.is_dir():
                raise IOError(f'root_path {root_dir!r} is not a directory')
            p = root / p

        if p != sys.path[0]:
            assert isinstance(p, (str, Path))
            sys.path.insert(0, str(p))

    if spec.import_name is None:
        return

    start_count = len(get_registry())
    importlib.import_module(spec.import_name)
    if len(get_registry()) - start_count == 0:
        logger = sa.util.get_logger('sweep')
        import_name = spec.import_name
        logger.warning(
            f'imported extension module {import_name!r}, but it did not bind a sensor'
        )
