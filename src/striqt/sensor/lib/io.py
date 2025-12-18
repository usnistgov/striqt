"""just a stub for now in case we change this in the future"""

from __future__ import annotations as __

import sys
import typing
from pathlib import Path

from striqt import analysis
from striqt.analysis.lib.io import decode_from_yaml_file
from striqt.analysis.lib.io import dump as dump_data  # noqa: F401
from striqt.analysis.lib.io import load as load_data  # noqa: F401
from ..specs.helpers import convert_dict
from .. import specs

from . import util

if typing.TYPE_CHECKING:
    import numpy as np
    import xarray as xr
    from striqt.analysis.lib.io import StoreType
else:
    np = util.lazy_import('numpy')
    xr = util.lazy_import('xarray')

__all__ = [
    'decode_from_yaml_file',
    'open_store',
    'read_yaml_spec',
    'dump_data',
    'load_data',
    'read_calibration',
    'save_calibration',
]


def open_store(
    spec: specs.Sink,
    *,
    alias_func: specs.helpers.PathAliasFormatter | None = None,
    force=False,
) -> StoreType:
    util.safe_import('xarray')

    spec_path = spec.path

    if alias_func is not None:
        spec_path = alias_func(spec_path)

    spec_path = Path(spec_path)

    if spec.store == 'directory' and not spec_path.stem.endswith('.zarr'):
        fixed_path = spec_path.with_suffix('.zarr')
    elif spec.store == 'zip' and not spec_path.stem.endswith('.zarr.zip'):
        fixed_path = spec_path.with_suffix('.zarr.zip')
    else:
        fixed_path = spec_path

    fixed_path.parent.mkdir(parents=True, exist_ok=True)
    store_backend = analysis.open_store(fixed_path, mode='w' if force else 'a')
    return store_backend


def _import_extensions_from_spec(
    spec: specs.Extension, alias_func: specs.helpers.PathAliasFormatter | None = None
) -> None:
    """import an extension class from a dict representation of structs.Extensions

    Arguments:
        spec: specification structure for the extension imports
        alias_func: formatter to fill aliases in the import path
    """

    import importlib
    from .bindings import get_registry

    if spec.import_path is None:
        pass
    else:
        if alias_func is None:
            p = spec.import_path
        else:
            p = alias_func(spec.import_path)

        if p != sys.path[0]:
            assert isinstance(p, (str, Path))
            sys.path.insert(0, str(p))

    if spec.import_name is None:
        return

    start_count = len(get_registry())
    importlib.import_module(spec.import_name)
    if len(get_registry()) - start_count == 0:
        logger = util.get_logger('sweep')
        import_name = spec.import_name
        logger.warning(
            f'imported extension module {import_name!r}, but it did not bind a sensor'
        )


def read_yaml_spec(
    path: str | Path,
    *,
    output_path: typing.Optional[str] = None,
    store_backend: typing.Optional[str] = None,
) -> specs.Sweep:
    """build a Sweep specification object from the specified yaml file.

    Args:
        path: path to the yaml file
        output_path: optional override for the specification's output path
        store_backend: optional override for the specification's output store backend

    Returns:
        an instance of structs.SweepSpec (or subclass)
    """

    from .bindings import get_tagged_sweep_type

    tree = decode_from_yaml_file(path)

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
        _import_extensions_from_spec(ext)

    spec = convert_dict(tree, type=get_tagged_sweep_type())

    sink = spec.sink
    if output_path is not None:
        sink = sink.replace(path=output_path)
    if store_backend is not None:
        sink = sink.replace(store=store_backend)
    return spec.replace(sink=sink)


def read_tdms_iq(
    path: Path | str,
    duration: float | None = None,
    *,
    base_clock_rate,
    num_rx_ports=1,
    dtype='complex64',
    skip_samples=0,
    array_backend: specs.types.ArrayBackend,
) -> tuple['np.ndarray', specs.FileCapture]:
    from .sources._file import TDMSSource

    source_spec = specs.TDMSSource(
        base_clock_rate=base_clock_rate, path=str(path), num_rx_ports=num_rx_ports
    )
    source = TDMSSource.from_spec(source_spec)

    capture = source.capture_spec

    source.arm(capture)
    iq, _ = source.read_iq()

    return iq, capture


@typing.overload
def read_calibration(
    path: None, alias_func: specs.helpers.PathAliasFormatter | None = None
) -> None: ...


@typing.overload
def read_calibration(
    path: str | Path, alias_func: specs.helpers.PathAliasFormatter | None = None
) -> 'xr.Dataset': ...


@util.lru_cache()
def read_calibration(
    path: str | Path | None, alias_func: specs.helpers.PathAliasFormatter | None = None
) -> 'xr.Dataset|None':
    if path is None:
        return None

    util.safe_import('xarray')

    if alias_func is not None:
        path = alias_func(path)

    return xr.open_dataset(path)


def save_calibration(path, corrections: 'xr.Dataset'):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    corrections.to_netcdf(path)
