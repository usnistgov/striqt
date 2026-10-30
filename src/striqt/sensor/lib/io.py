"""file I/O for action specs, calibration data, and zarr stores.

The spec readers import the `extensions` block before decoding, and decode against
the class bound under the ``sensor_binding`` tag unless a `type` is given.
"""

from __future__ import annotations as __

import contextlib
import shutil
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, overload

import msgspec

import striqt.analysis as sa
from striqt.analysis.lib.util import np, xr

from .. import specs

if TYPE_CHECKING:
    from .typing import ZarrStore


def open_store(
    spec: specs.ZarrSink,
    *,
    format_path: specs.paths.PathFormatter | None = None,
    force=False,
) -> ZarrStore:
    """open a new zarr store at `spec.path` for writing.

    A path ending in ``.zip``, or any path when `spec.store` is ``'zip'``, opens a
    zip store at a ``.zarr.zip`` path; any other path opens a directory store at a
    ``.zarr`` path. Missing parent directories are created.

    Args:
        spec: the sink block with the `path` and `store` type
        format_path: expands the ``{fields}`` of `spec.path`; None leaves the path
            as is
        force: if True, remove a store already at the path; if False, refuse one

    Raises:
        ValueError: if `spec.path` is None
        FileExistsError: if a non-empty store is at the path and `force` is False
    """
    if spec.path is None:
        raise ValueError('sink.path is null, but a zarr store needs a path')
    spec_path = spec.path

    if format_path is not None:
        spec_path = format_path(spec_path)

    spec_path = Path(spec_path)
    name = spec_path.name.lower()

    if spec.store == 'zip' or name.endswith('.zip'):
        if name.endswith('.zarr.zip'):
            fixed_path = spec_path
        elif name.endswith('.zip'):
            fixed_path = spec_path.with_name(
                spec_path.name[: -len('.zip')] + '.zarr.zip'
            )
        else:
            fixed_path = spec_path.with_name(spec_path.name + '.zarr.zip')
    elif name.endswith('.zarr'):
        fixed_path = spec_path
    else:
        fixed_path = spec_path.with_name(spec_path.name + '.zarr')

    fixed_path.parent.mkdir(parents=True, exist_ok=True)
    occupied = fixed_path.is_file() or (
        fixed_path.is_dir() and any(fixed_path.iterdir())
    )
    if occupied and not force:
        raise FileExistsError(f'a store is already at {str(fixed_path)!r}')
    if occupied:
        if fixed_path.is_dir():
            shutil.rmtree(fixed_path)
        else:
            fixed_path.unlink()
    return sa.open_store(fixed_path, mode='w')


def read_yaml_spec(
    path: str | Path,
    *,
    type: type[specs.ActionSpec] | None = None,
    output_path: Optional[str] = None,
    store_backend: Optional[str] = None,
) -> specs.ActionSpec:
    """decode the action spec in the yaml file at `path`.

    The `extensions` block of the file is imported before decoding. A relative
    `extensions.import_path` is resolved against the directory of the file.

    Args:
        path: the yaml file
        type: the action spec type to decode into, or None to select the type
            bound under the ``sensor_binding`` tag
        output_path: replaces `sink.path` in the returned spec when given
        store_backend: replaces `sink.store` of a `Sweep` spec when given

    Returns:
        an instance of `type`, or of the bound class when `type` is None

    Raises:
        msgspec.ValidationError: if the file does not validate as the spec class, or
            if the tag names no registered binding
        TypeError: if `store_backend` is given and the `sink` of the spec is not a
            `ZarrSink`
        FileNotFoundError: if `extensions.import_path` does not exist
    """

    tree = sa.lib.io.decode_from_yaml_file(path)
    return _decode_spec_tree(
        tree,
        Path(path).resolve().parent,
        type=type,
        output_path=output_path,
        store_backend=store_backend,
    )


def read_json_spec(
    path: str | Path,
    *,
    type: type[specs.ActionSpec] | None = None,
    output_path: Optional[str] = None,
    store_backend: Optional[str] = None,
) -> specs.ActionSpec:
    """decode the action spec in the json file at `path`.

    The file has the schema of a yaml spec without ``!include`` directives. The
    `extensions` block of the file is imported before decoding. A relative
    `extensions.import_path` is resolved against the directory of the file.

    Args:
        path: the json file
        type: the action spec type to decode into, or None to select the type
            bound under the ``sensor_binding`` tag
        output_path: replaces `sink.path` in the returned spec when given
        store_backend: replaces `sink.store` of a `Sweep` spec when given

    Returns:
        an instance of `type`, or of the bound class when `type` is None

    Raises:
        msgspec.ValidationError: if the file does not validate as the spec class, or
            if the tag names no registered binding
        TypeError: if `store_backend` is given and the `sink` of the spec is not a
            `ZarrSink`
        FileNotFoundError: if `extensions.import_path` does not exist
    """
    with open(path, 'rb') as buf:
        tree = msgspec.json.decode(buf.read(), type=dict)

    return _decode_spec_tree(
        tree,
        Path(path).resolve().parent,
        type=type,
        output_path=output_path,
        store_backend=store_backend,
    )


def read_zarr_spec(
    path: str | Path,
    *,
    type: type[specs.ActionSpec] | None = None,
    output_path: Optional[str] = None,
    store_backend: Optional[str] = None,
    extension_root: str | Path | None = None,
) -> specs.ActionSpec:
    """decode the sweep spec saved in the attrs of the zarr store at `path`.

    The `extensions` block of the attrs is imported before decoding. Only the attrs
    that name fields of the sweep class, the ``sensor_binding`` tag, and
    ``mock_source`` are read. An ``extensions.sink`` attr is dropped with a
    `DeprecationWarning`.

    Args:
        path: the zarr store, a ``.zarr`` directory or ``.zarr.zip`` archive
        type: the sweep class to decode against, or None to select the class bound
            under the saved ``sensor_binding`` tag; a saved ``mock_source`` attr
            selects the mock class of that binding instead
        output_path: replaces `sink.path` in the returned spec when given
        store_backend: replaces `sink.store` of a `Sweep` spec when given
        extension_root: the directory against which a relative
            `extensions.import_path` is resolved, or None for the directory of the
            store

    Returns:
        an instance of `type`, or of the bound sweep class when `type` is None

    Raises:
        msgspec.ValidationError: if the attrs do not validate as the sweep class, or
            if the tag names no registered binding
        TypeError: if `store_backend` is given and the `sink` of the spec is not a
            `ZarrSink`
        FileNotFoundError: if `extensions.import_path` or `extension_root` does not
            exist
    """
    attrs = sa.lib.io.load_attrs(path)

    if extension_root is None:
        extension_root = Path(path).resolve().parent

    attrs = _drop_removed_extension_sink(attrs)
    sweep_cls = _resolve_sweep_cls(attrs, extension_root, type=type)

    # without a registered binding, the `Sweep` fields select the attrs and the
    # tagged union decode reports the unknown tag
    fields_cls = specs.Sweep if sweep_cls is None else sweep_cls
    fields = set(fields_cls.__struct_fields__)
    fields |= {specs.structs.SWEEP_TAG_FIELD, 'mock_source'}
    tree = {k: v for k, v in attrs.items() if k in fields}

    return _decode_sweep(
        tree, sweep_cls, output_path=output_path, store_backend=store_backend
    )


def _drop_removed_extension_sink(attrs: dict[str, Any]) -> dict[str, Any]:
    """return `attrs` without the ``extensions.sink`` key.

    A `DeprecationWarning` is emitted when the key is present. The sink class the
    key named is not applied.
    """
    extensions = attrs.get('extensions')
    if not isinstance(extensions, dict) or 'sink' not in extensions:
        return attrs
    warnings.warn(
        'ignoring the removed extensions.sink attr of this data store',
        DeprecationWarning,
        stacklevel=3,
    )
    extensions = {k: v for k, v in extensions.items() if k != 'sink'}
    return {**attrs, 'extensions': extensions}


def _decode_spec_tree(
    tree: dict[str, Any],
    extension_root: str | Path,
    *,
    type: type[specs.ActionSpec] | None = None,
    output_path: Optional[str] = None,
    store_backend: Optional[str] = None,
) -> specs.ActionSpec:
    """decode an action spec from the dict form of a YAML or JSON file.

    The `extensions` block of `tree` is imported before decoding. A ``mock_source``
    entry rewrites ``tree['sensor_binding']`` in place.
    """
    sweep_cls = _resolve_sweep_cls(tree, extension_root, type=type)
    return _decode_sweep(
        tree, sweep_cls, output_path=output_path, store_backend=store_backend
    )


def _resolve_sweep_cls(
    tree: dict[str, Any],
    extension_root: str | Path,
    *,
    type: type[specs.ActionSpec] | None = None,
) -> type[specs.ActionSpec] | None:
    """return the spec class of `tree` after importing the `extensions` block of `tree`.

    A binding registered by the imported extension module is found. A
    ``mock_source`` entry with a registered tag selects the mock class of that
    binding, rewrites ``tree['sensor_binding']`` in place to the mock class name,
    and overrides `type`. Otherwise `type` is returned when given, and None when the
    tag names no registered binding.

    Raises:
        FileNotFoundError: if `extensions.import_path` or `extension_root` does not
            exist
    """
    from .bindings import get_controller, get_registry

    if 'extensions' in tree:
        ext = specs.Extension.from_dict(tree['extensions'])
        _import_extensions_from_spec(ext, root_dir=extension_root)

    tag = tree.get(specs.structs.SWEEP_TAG_FIELD, None)
    mock_source = tree.get('mock_source', None)

    if not isinstance(tag, str) or tag not in get_registry():
        bound_cls = None
    elif mock_source is not None:
        bound_cls = get_controller(tag, mock_source).sensor.sweep_spec_cls
        # msgspec rejects a tagged struct whose tag field does not match the class
        tree[specs.structs.SWEEP_TAG_FIELD] = bound_cls.__name__
        return bound_cls
    else:
        bound_cls = get_controller(tag).sensor.sweep_spec_cls

    return bound_cls if type is None else type


def _decode_sweep(
    tree: dict[str, Any],
    sweep_cls: type[specs.ActionSpec] | None,
    *,
    output_path: str | None = None,
    store_backend: str | None = None,
) -> specs.ActionSpec:
    """decode `tree` as `sweep_cls` and apply the sink path and store overrides.

    When `sweep_cls` is None, `tree` is decoded against the tagged union of all
    registered bindings.

    Raises:
        msgspec.ValidationError: if `tree` does not validate as `sweep_cls`, or if
            `sweep_cls` is None and no binding is registered under the tag of `tree`
        TypeError: if `store_backend` is given and the decoded `sink` is not a
            `ZarrSink`
    """
    if sweep_cls is None:
        from .bindings import get_tagged_sweep_type

        sweep_cls = get_tagged_sweep_type()

    spec = sa.specs.helpers.convert_dict(tree, type=sweep_cls)

    sink = spec.sink
    if store_backend is not None:
        if not isinstance(sink, specs.ZarrSink):
            raise TypeError(
                'store_backend applies only to a Sweep spec with a ZarrSink'
            )
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
) -> tuple[np.ndarray, specs.FileCapture]:
    from ..bindings import tdms_file

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
) -> xr.Dataset: ...


@sa.util.lru_cache()
def read_calibration(
    path: str | Path | None, format_path: specs.paths.PathFormatter | None = None
) -> xr.Dataset | None:
    if path is None:
        return None

    # `format_path` blocks on the source id until the radio opens; the modules that
    # the read needs load during that wait
    _ = xr.__version__
    with contextlib.suppress(ImportError):
        import netCDF4  # noqa: F401

    if format_path is not None:
        path = format_path(path)

    # the cached result is read from several threads. The HDF5 library is not thread
    # safe in every build, so the netCDF handle is closed before return
    with xr.open_dataset(path) as dataset:
        return dataset.load()


def save_calibration(path, corrections: xr.Dataset):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    corrections.to_netcdf(path)


def _import_extensions_from_spec(
    spec: specs.Extension,
    format_path: specs.paths.PathFormatter | None = None,
    root_dir: Path | str = '.',
) -> None:
    """insert `spec.import_path` into ``sys.path`` and import `spec.import_name`.

    The insertion is skipped when `import_path` is None or already first on
    ``sys.path``, and the import when `import_name` is None. A WARNING is logged on
    the ``sweep`` logger when the imported module registers no sensor binding.

    Args:
        spec: the `extensions` block of an action spec
        format_path: expands the ``{fields}`` of `import_path` when given
        root_dir: the directory against which a relative `import_path` is resolved

    Raises:
        FileNotFoundError: if `root_dir` or the resolved `import_path` does not
            exist
        IOError: if `root_dir` is not a directory
        ModuleNotFoundError: if `import_name` cannot be imported
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
                raise OSError(f'root_path {root_dir!r} is not a directory')
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
