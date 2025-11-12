"""just a stub for now in case we change this in the future"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
import sys
import typing

from striqt import analysis
from striqt.analysis.lib.specs import convert_dict
from striqt.analysis.lib.io import decode_from_yaml_file, load, dump  # noqa: F401

from . import specs, sinks, util, captures

if typing.TYPE_CHECKING:
    import numpy as np
else:
    np = util.lazy_import('numpy')


def _get_capture_format_fields(
    capture: specs.CaptureSpec,
    sweep: specs.SweepSpec,
    *,
    source_id: str | None = None,
    yaml_path: Path | str | None,
) -> dict[str, str]:
    """return a mapping for string `'{field_name}'.format()` style mapping values"""
    fields = captures.capture_fields_with_aliases(
        capture, source_id=source_id, output=sweep.sink
    )

    fields['start_time'] = datetime.now().strftime('%Y%m%d-%Hh%Mm%S')
    fields['driver'] = sweep.source.driver
    if yaml_path is not None:
        fields['yaml_name'] = Path(yaml_path).stem
    fields['source_id'] = source_id

    return fields


def expand_path(
    path: str | Path,
    sweep: specs.SweepSpec[specs._TS, specs._TC] | None = None,
    *,
    source_id: str | None = None,
    file_anchor: str | Path | None = None,
) -> str:
    """return an absolute path, allowing for user tokens (~) and {field} in the input."""
    path = Path(path).expanduser()
    if sweep is not None and len(sweep.get_captures(False)) > 0:
        captures = sweep.get_captures(False)
        fields = _get_capture_format_fields(
            captures[0], sweep, source_id=source_id, yaml_path=file_anchor
        )
        try:
            path = Path(str(path).format(**fields))
        except KeyError as ex:
            valid_fields = ', '.join(fields.keys())
            raise ValueError(f'valid formatting fields are {valid_fields!r}') from ex

    if file_anchor is not None and not path.is_absolute():
        path = Path(file_anchor).parent.absolute() / path
    return str(path.absolute())


def open_store(
    sweep,
    *,
    source_id: str,
    yaml_path: str | None = None,
    output_path=None,
    store_backend=None,
    force=False,
):
    if store_backend is None:
        store_backend = sweep.output.store.lower()
    else:
        store_backend = store_backend.lower()

    if output_path is None:
        spec_path = sweep.output.path
    else:
        spec_path = expand_path(
            output_path, sweep, source_id=source_id, file_anchor=yaml_path
        )

    if store_backend == 'directory':
        fixed_path = Path(spec_path).with_suffix('.zarr')
    else:
        fixed_path = Path(spec_path).with_suffix('.zarr.zip')

    fixed_path.parent.mkdir(parents=True, exist_ok=True)
    store_backend = analysis.open_store(fixed_path, mode='w' if force else 'a')
    return store_backend


def _import_extensions(spec: specs.ExtensionSpec, path: typing.Optional[str | Path]):
    """import an extension class from a dict representation of structs.Extensions

    Arguments:
        spec: specification structure for the extension imports
        name: extensions key containing the name to import
    """
    import importlib

    if spec.import_path is None:
        pass
    else:
        p = expand_path(spec.import_path, file_anchor=path)
        if p != sys.path[0]:
            assert isinstance(p, (str, Path))
            sys.path.insert(0, str(p))

    from ..bindings import get_registry

    if spec.import_name is None:
        return

    start_count = len(get_registry())
    importlib.import_module(spec.import_name)
    if len(get_registry()) - start_count == 0:
        logger = util.get_logger('controller')
        import_name = spec.import_name
        logger.warning(
            f'imported extension module {import_name!r}, but it did not bind a sensor'
        )


def _reanchor_path(path: str | Path, anchor: str | Path) -> Path:
    """renormalize path relative to the parent of anchor, if path is a relative path"""

    path = Path(path)
    anchor_parent = Path(anchor).parent

    if path.is_relative_to(anchor_parent):
        return path.relative_to(anchor_parent)
    else:
        return path


def read_yaml_sweep(
    path: str | Path,
    *,
    source_id: str | None = None,
) -> specs.SweepSpec:
    """build a Sweep struct from the contents of specified yaml file.

    Args:
        path: path to the yaml file
        source_id: unique hardware identifier of the radio for filename substitutions

    Returns:
        an instance of a structs.SweepSpec subclass
    """

    from ..bindings import get_tagged_sweep_spec

    # first pass is a simple dict
    tree = decode_from_yaml_file(path)
    assert isinstance(tree, dict), 'yaml file does not specify a dict structure'

    extensions = convert_dict(tree.get('extensions', {}), specs.ExtensionSpec)
    _import_extensions(extensions, path)

    sweep: specs.SweepSpec = convert_dict(tree, type=get_tagged_sweep_spec())

    return fill_aliases(path, sweep, source_id)


def fill_aliases(
    root_file: Path | str,
    sweep: specs.SweepSpec[specs._TS, specs._TC],
    source_id: str | None,
) -> specs.SweepSpec[specs._TS, specs._TC]:
    """replace formatting fields like {source_id} with aliases"""

    def expand(path):
        return expand_path(
            path, file_anchor=root_file, sweep=sweep, source_id=source_id
        )

    replace = {}

    sink = sweep.sink
    if sweep.sink.path is not None:
        sink = sink.replace(path=expand(sweep.sink.path))
    if sweep.sink.log_path is not None:
        sink = sink.replace(log_path=expand(sweep.sink.log_path))
    replace['sink'] = sink

    if sweep.source.calibration is not None:
        cal_path = _reanchor_path(sweep.source.calibration, root_file)
        cal_path = expand(cal_path)
        replace['source'] = sweep.source.replace(calibration=cal_path)

    return sweep.replace(**replace)


def read_tdms_iq(
    path: Path | str,
    duration: float | None = None,
    *,
    base_clock_rate,
    num_rx_ports=1,
    dtype='complex64',
    skip_samples=0,
    array_backend: specs.ArrayBackendType,
) -> tuple['np.ndarray', specs.FileCaptureSpec]:
    from .sources.testing import TDMSFileSource, TDMSSourceSpec

    source_spec = TDMSSourceSpec(
        base_clock_rate=base_clock_rate, path=Path(path), num_rx_ports=num_rx_ports
    )
    source = TDMSFileSource(source_spec)

    capture = source.get_capture_spec()

    source.arm(capture)
    iq, _ = source.read_iq()

    return iq, capture
