"""just a stub for now in case we change this in the future"""

from __future__ import annotations as __

import sys
import msgspec
from pathlib import Path
from typing import Optional, overload, TYPE_CHECKING

import striqt.analysis as sa
from striqt.analysis.lib.util import np, xr

from .. import specs

import msgspec

if TYPE_CHECKING:
    from .typing import ZarrStore


def open_store(
    spec: specs.Sink,
    *,
    format_path: specs.paths.PathFormatter | None = None,
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
    type: type[specs.Sweep] | None = None,
    output_path: Optional[str] = None,
    store_backend: Optional[str] = None,
) -> specs.Sweep:
    """Build a Sweep specification object from the specified yaml file.

    Args:
        path: path to the yaml file
        output_path: optional override for the specification's output path
        type: the type of sweep specification to load, or None to use specs.Sweep
        store_backend: optional override for the specification's output store backend

    Returns:
        an instance of specs.Sweep
    """

    tree = sa.lib.io.decode_from_yaml_file(path)
    return _convert_zarr_attrs_spec(
        tree,
        Path(path).resolve().parent,
        type=type,
        output_path=output_path,
        store_backend=store_backend,
    )


def read_json_spec(
    path: str | Path,
    *,
    type: type[specs.Sweep] | None = None,
    output_path: Optional[str] = None,
    store_backend: Optional[str] = None,
) -> specs.Sweep:
    """Build a specs.Sweep specification object from a json file at the given path.

    This supports the same specifications as a yaml file, but without !include
    directives.

    Args:
        path: path to the yaml file
        type: the type of sweep specification to load, or None to use specs.Sweep
        output_path: optional override for the specification's output path
        store_backend: optional override for the specification's output store backend

    Returns:
        an instance of specs.Sweep
    """
    with open(path, 'rb') as buf:
        tree = msgspec.json.decode(buf.read(), type=dict)

    return _convert_zarr_attrs_spec(
        tree,
        Path(path).resolve().parent,
        type=type,
        output_path=output_path,
        store_backend=store_backend,
    )


def read_zarr_spec(
    path: str | Path,
    *,
    type: type[specs.Sweep] | None = None,
    output_path: Optional[str] = None,
    store_backend: Optional[str] = None,
    extension_root: str | Path | None = None,
) -> specs.Sweep:
    """rebuild the `Sweep` that `iterate_sweep` saved in the attrs of a zarr store.

    Only attrs that name `Sweep` fields (plus the binding tag and ``mock_source``)
    are read; other dataset attrs are ignored. The `extensions` block is imported
    before the sweep is decoded, so a `sensor_binding` it defines is available. As
    for `read_yaml_spec`, passing the store path to `open_resources` derives the
    ``{spec_name}`` and ``{parent_name}`` path fields from it.

    Args:
        path: the zarr store, a ``.zarr`` directory or ``.zarr.zip`` archive
        type: the sweep class to decode against, or None to decode against the
            tagged union of registered sensor bindings and filter attrs by the fields
            of `specs.Sweep`. A saved ``mock_source`` attr selects that binding's
            sweep class instead.
        output_path: replaces `sink.path` in the returned spec when given
        store_backend: replaces `sink.store` in the returned spec when given
        extension_root: directory that a relative `extensions.import_path` is
            resolved against, or None for the store's directory

    Returns:
        the decoded sweep, an instance of `type` or of the bound sweep class the tag
        selects

    Raises:
        msgspec.ValidationError: the attrs do not validate as a sweep of that class
    """
    attrs = sa.lib.io.load_attrs(path)
    sweep_cls = specs.Sweep if type is None else type
    fields = set(sweep_cls.__struct_fields__)
    fields |= {specs.structs.SWEEP_TAG_FIELD, 'mock_source'}
    tree = {k: v for k, v in attrs.items() if k in fields}

    if extension_root is None:
        extension_root = Path(path).resolve().parent

    return _convert_zarr_attrs_spec(
        tree,
        extension_root,
        type=type,
        output_path=output_path,
        store_backend=store_backend,
    )


def _convert_zarr_attrs_spec(
    tree: dict,
    extension_root: str | Path,
    *,
    type: type[specs.Sweep] | None = None,
    output_path: Optional[str] = None,
    store_backend: Optional[str] = None,
) -> specs.Sweep:
    mock_source = tree.get('mock_source', None)
    if mock_source is not None:
        assert 'sensor_binding' in tree, TypeError('missing "sensor_binding"')
        from .bindings import get_controller

        ctrl_cls = get_controller(tree['sensor_binding'], mock_source)
        type = ctrl_cls.sensor.sweep_spec_cls
        tree['sensor_binding'] = type.__name__

    if 'extensions' in tree:
        # import now, so that sensor_binding keys can use definitions
        # in extension modules
        ext = specs.Extension.from_dict(tree['extensions'])
        _import_extensions_from_spec(ext, root_dir=extension_root)

    if type is None:
        from .bindings import get_tagged_sweep_type

        sweep_cls = get_tagged_sweep_type()
    else:
        sweep_cls = type

    spec = sa.specs.helpers.convert_dict(tree, type=sweep_cls)

    sink = spec.sink
    if store_backend is not None:
        sink = sink.replace(store=store_backend)
    if output_path is not None:
        sink = sink.replace(path=output_path)

    return spec.replace(sink=sink)


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
    with tdms_file.from_source_spec(source_spec) as source:
        source.arm(**source.capture_spec.to_dict())
        iq, _ = source.read_iq()
        capture = source.capture_spec

    return iq, capture


@overload
def read_calibration(
    path: None, format_path: specs.paths.PathFormatter | None = None
) -> None: ...


@overload
def read_calibration(
    path: str | Path, format_path: specs.paths.PathFormatter | None = None
) -> 'xr.Dataset': ...


@sa.util.lru_cache()
def read_calibration(
    path: str | Path | None, format_path: specs.paths.PathFormatter | None = None
) -> 'xr.Dataset|None':
    if path is None:
        return None

    if format_path is not None:
        path = format_path(path)

    # the result is cached and read from several threads, so leave no open netCDF
    # handle behind: the HDF5 library is not thread safe in every build
    with xr.open_dataset(path) as dataset:
        return dataset.load()


def save_calibration(path, corrections: 'xr.Dataset'):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    corrections.to_netcdf(path)


def _import_extensions_from_spec(
    spec: specs.Extension,
    format_path: specs.paths.PathFormatter | None = None,
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

        if not p.is_dir():
            raise FileNotFoundError(
                f'extension import_path {str(p)!r} is not a directory'
            )

        if p != Path(sys.path[0]):
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
