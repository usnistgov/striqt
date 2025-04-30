"""just a stub for now in case we change this in the future"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
import sys
import typing

import msgspec

import channel_analysis
from channel_analysis import load, dump  # noqa: F401

from . import util, captures, structs

if typing.TYPE_CHECKING:
    import iqwaveform
    import numpy as np
    import pandas as pd
else:
    iqwaveform = util.lazy_import('iqwaveform')
    np = util.lazy_import('numpy')
    pd = util.lazy_import('pandas')


SweepType = typing.TypeVar('SweepType', bound=structs.Sweep)


def _dec_hook(type_, obj):
    if typing.get_origin(type_) is pd.Timestamp:
        return pd.to_datetime(obj)
    else:
        return obj


def _get_default_format_fields(
    sweep: structs.Sweep, *, radio_id: str | None = None, yaml_path: Path | str | None
) -> dict[str, str]:
    """return a mapping for string `'{field_name}'.format()` style mapping values"""
    fields = captures.capture_fields_with_aliases(
        sweep.defaults, radio_id=radio_id, output=sweep.output
    )

    fields['start_time'] = datetime.now().strftime('%Y%m%d-%Hh%Mm%S')
    fields['driver'] = sweep.radio_setup.driver
    if yaml_path is not None:
        fields['yaml_name'] = Path(yaml_path).stem
    fields['radio_id'] = radio_id

    return fields


def expand_path(
    path: str | Path,
    sweep: structs.Sweep | None = None,
    *,
    radio_id: str | None = None,
    relative_to_file=None,
) -> str:
    """return an absolute path, allowing for user tokens (~) and {field} in the input."""
    if path is None:
        return None

    path = Path(path).expanduser()
    if sweep is not None:
        fields = _get_default_format_fields(
            sweep, radio_id=radio_id, yaml_path=relative_to_file
        )
        path = Path(str(path).format(**fields))

    if relative_to_file is not None and not path.is_absolute():
        path = Path(relative_to_file).parent.absolute() / path
    return str(path.absolute())


def open_store(
    sweep,
    *,
    radio_id: str,
    yaml_path: str = None,
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
            output_path, sweep, radio_id=radio_id, relative_to_file=yaml_path
        )

    if store_backend == 'directory':
        fixed_path = Path(spec_path).with_suffix('.zarr')
    else:
        fixed_path = Path(spec_path).with_suffix('.zarr.zip')

    fixed_path.parent.mkdir(parents=True, exist_ok=True)
    store_backend = channel_analysis.open_store(fixed_path, mode='w' if force else 'a')
    return store_backend


def _import_extension(extensions: dict[str, str], name: str) -> type:
    """import an extension class from a dict representation of structs.Extensions

    Arguments:
        extensions: builtin representation of structs.Extensions
        name: extensions key containing the name to import

    Returns:

    """
    import importlib

    if extensions['import_path'] is None:
        pass
    elif extensions['import_path'] not in sys.path:
        sys.path.insert(0, extensions['import_path'])

    spec = extensions[name]

    mod_name, *sub_names, obj_name = spec.rsplit('.')
    mod = importlib.import_module(mod_name)
    for name in sub_names:
        mod = getattr(mod, name)
    return getattr(mod, obj_name)


@typing.overload
def read_yaml_sweep(
    path: str | Path, *, sweep_cls: type[SweepType], radio_id: str | None = None
) -> SweepType:
    pass


@typing.overload
def read_yaml_sweep(
    path: str | Path, *, sweep_cls=None, radio_id: str | None = None
) -> dict:
    pass


def read_yaml_sweep(
    path: str | Path,
    *,
    radio_id: str | None = None,
) -> SweepType | dict:
    """build a Sweep struct from the contents of specified yaml file.

    Args:
        path: path to the yaml file
        sweep_cls: instantiate and return this subclass of structs.Sweep
        radio_id: unique hardware identifier of the radio for filename substitutions
    """

    with open(path, 'rb') as fd:
        text = fd.read()

    # build a dict to extract the list of sweep fields and apply defaults
    tree = msgspec.yaml.decode(text, type=dict, strict=False)

    # apply default capture settings
    defaults = tree['defaults']
    if tree['radio_setup'].get('calibration', None):
        cal_path = Path(tree['radio_setup']['calibration'])
        sweep_parent = Path(path).parent
        if cal_path.is_relative_to(sweep_parent):
            # take relative paths with respect to a yaml file,
            # not the interpreter working directory
            cal_path = cal_path.relative_to(sweep_parent)
        cal_path = str(cal_path)
        tree['radio_setup']['calibration'] = cal_path

    tree['captures'] = [defaults | c for c in tree.get('captures', ())]

    extensions = tree['extensions']
    extensions['import_path'] = expand_path(
        extensions.get('import_path', None), relative_to_file=path
    )
    sweep_cls = _import_extension(extensions, 'sweep_struct')

    sweep: structs.Sweep = channel_analysis.builtins_to_struct(
        tree, type=sweep_cls, strict=False, dec_hook=_dec_hook
    )

    # fill formatting fields in paths
    kws = dict(relative_to_file=path, sweep=sweep, radio_id=radio_id)

    output_path = expand_path(sweep.output.path, **kws)
    output_spec = msgspec.structs.replace(sweep.output, path=output_path)

    cal_path = expand_path(sweep.radio_setup.calibration, **kws)
    setup_spec = msgspec.structs.replace(sweep.radio_setup, calibration=cal_path)

    import_path = expand_path(sweep.extensions.import_path, **kws)
    extensions_spec = msgspec.structs.replace(sweep.extensions, import_path=import_path)

    sweep = msgspec.structs.replace(
        sweep, output=output_spec, radio_setup=setup_spec, extensions=extensions_spec
    )

    return sweep


def read_tdms_iq(
    path: Path | str,
    duration: float = None,
    *,
    rx_channel_count=1,
    dtype='complex64',
    skip_samples=0,
    xp=np,
) -> tuple['iqwaveform.type_stubs.ArrayLike', structs.FileSourceCapture]:
    from .sources.testing import TDMSFileSource

    source = TDMSFileSource(path=path, rx_channel_count=rx_channel_count)
    capture = source.get_capture_struct()

    source.arm(capture)
    iq, _ = source.read_iq(capture)

    return iq
