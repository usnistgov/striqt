"""file I/O for sweep specifications, calibration data, and zarr stores"""

from __future__ import annotations as __

import sys
import warnings
import msgspec
from pathlib import Path
from typing import Any, Optional, overload, TYPE_CHECKING

import striqt.analysis as sa
from striqt.analysis.lib.util import np, xr

from .. import specs

import msgspec

if TYPE_CHECKING:
    from .typing import ZarrStore


def open_store(
    spec: specs.ZarrSink,
    *,
    format_path: specs.paths.PathFormatter | None = None,
    force=False,
) -> ZarrStore:
    """open the zarr store at `spec.path`, formatted by `format_path`, for appending.

    `spec.store` fixes the suffix to ``.zarr`` or ``.zarr.zip``; with `force` an
    existing store is truncated.

    Raises:
        ValueError: if `spec.path` is None
    """
    if spec.path is None:
        raise ValueError('sink.path is null, but a zarr store needs a path')
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
    type: type[specs.ActionSpec] | None = None,
    output_path: Optional[str] = None,
    store_backend: Optional[str] = None,
) -> specs.ActionSpec:
    """Build a Sweep specification object from the specified yaml file.

    Args:
        path: path to the yaml file
        output_path: optional override for the specification's output path
        type: the action spec type to decode into, or None to select it by the
            ``sensor_binding`` tag
        store_backend: replaces `sink.store` of a `Sweep` spec when given

    Returns:
        the decoded spec, an instance of the bound `ActionSpec` class
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
    type: type[specs.ActionSpec] | None = None,
    output_path: Optional[str] = None,
    store_backend: Optional[str] = None,
) -> specs.ActionSpec:
    """Build a specs.Sweep specification object from a json file at the given path.

    JSON supports the same specifications as yaml, but without !include
    directives.

    Args:
        path: path to the json file
        type: the action spec type to decode into, or None to select it by the
            ``sensor_binding`` tag
        output_path: optional override for the specification's output path
        store_backend: replaces `sink.store` of a `Sweep` spec when given

    Returns:
        the decoded spec, an instance of the bound `ActionSpec` class
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
    type: type[specs.ActionSpec] | None = None,
    output_path: Optional[str] = None,
    store_backend: Optional[str] = None,
    extension_root: str | Path | None = None,
) -> specs.ActionSpec:
    """rebuild the `Sweep` that `iterate_sweep` saved in the attrs of a zarr store.

    The `extensions` block is imported before the sweep is decoded. A
    `sensor_binding` that the extension module defines is accepted. Only attrs
    that name fields of the sweep class being decoded, the binding tag, and
    ``mock_source`` are read. Other dataset attrs are ignored. Passing the store
    path to `open_resources` derives the ``{spec_name}`` and ``{parent_name}`` path
    fields from it, as for `read_yaml_spec`.

    Args:
        path: the zarr store, a ``.zarr`` directory or ``.zarr.zip`` archive
        type: the sweep class to decode against, or None to decode against the
            sweep class bound under the saved `sensor_binding` tag, including any
            fields that class adds to `specs.Sweep`. A saved ``mock_source`` attr
            selects that binding's mock sweep class instead.
        output_path: replaces `sink.path` in the returned spec when given
        store_backend: replaces `sink.store` of a `Sweep` spec when given
        extension_root: directory that a relative `extensions.import_path` is
            resolved against, or None for the store's directory

    Returns:
        the decoded sweep, an instance of `type` or of the bound sweep class the tag
        selects

    Raises:
        msgspec.ValidationError: the attrs do not validate as a sweep of that class,
            or the tag names no registered binding
        FileNotFoundError: `extensions.import_path` or `extension_root` does not
            exist
    """
    attrs = sa.lib.io.load_attrs(path)

    if extension_root is None:
        extension_root = Path(path).resolve().parent

    attrs = _drop_removed_extension_sink(attrs)
    sweep_cls = _resolve_sweep_cls(attrs, extension_root, type=type)

    # with no registered binding to filter by, the base fields stand in and the
    # tagged union reports the unknown tag
    fields_cls = specs.Sweep if sweep_cls is None else sweep_cls
    fields = set(fields_cls.__struct_fields__)
    fields |= {specs.structs.SWEEP_TAG_FIELD, 'mock_source'}
    tree = {k: v for k, v in attrs.items() if k in fields}

    return _decode_sweep(
        tree, sweep_cls, output_path=output_path, store_backend=store_backend
    )


def _drop_removed_extension_sink(attrs: dict[str, Any]) -> dict[str, Any]:
    """return `attrs` without an ``extensions.sink`` key, which `Extension` does not define.

    Stores written by earlier releases carry the key. A `DeprecationWarning` reports
    the drop. The sink class the key named is not applied; a caller that wants no
    output sets ``sink.path`` to None on the returned spec.
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


def _convert_zarr_attrs_spec(
    tree: dict[str, Any],
    extension_root: str | Path,
    *,
    type: type[specs.ActionSpec] | None = None,
    output_path: Optional[str] = None,
    store_backend: Optional[str] = None,
) -> specs.ActionSpec:
    """decode a sweep from the dict form of its YAML, JSON, or zarr attrs.

    Imports the `extensions` block of `tree` before decoding. A ``mock_source`` entry
    rewrites ``tree['sensor_binding']`` in place (see `_resolve_sweep_cls`).
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
    """return the sweep class for `tree` after importing its `extensions` block.

    The `sensor_binding` tag of `tree` is looked up after the import and may name a
    binding that the extension module registers. A ``mock_source`` entry selects the
    mock of that binding and rewrites ``tree['sensor_binding']`` in place to the mock
    class name. This selection takes precedence over `type`. Otherwise `type` is
    returned when given. The result is None when `type` is None and the tag names no
    registered binding.

    Raises:
        FileNotFoundError: `extensions.import_path` or `extension_root` does not
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
        msgspec.ValidationError: `tree` does not validate as `sweep_cls`, or no
            binding is registered under its tag
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
    # handle behind. The HDF5 library is not thread safe in every build
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
