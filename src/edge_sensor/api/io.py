"""just a stub for now in case we change this in the future"""

from __future__ import annotations
from datetime import datetime
import numpy as np
from pathlib import Path
import typing

import msgspec

from .structs import Sweep, RadioCapture, FileSourceCapture  # noqa: F401
from . import util, captures
import channel_analysis
from channel_analysis import load, dump  # noqa: F401

if typing.TYPE_CHECKING:
    import pandas as pd
    import iqwaveform
else:
    pd = util.lazy_import('pandas')
    iqwaveform = util.lazy_import('iqwaveform')


SweepType = typing.TypeVar('SweepType', bound=Sweep)


def _dec_hook(type_, obj):
    if typing.get_origin(type_) is pd.Timestamp:
        return pd.to_datetime(obj)
    else:
        return obj


def _get_default_format_fields(
    sweep: Sweep, *, radio_id: str | None = None, yaml_path
) -> dict[str, str]:
    """return a mapping for string `'{field_name}'.format()` style mapping values"""
    fields = captures.capture_fields_with_aliases(
        sweep.captures[0], radio_id=radio_id, output=sweep.output
    )

    fields['start_time'] = datetime.now().strftime('%Y%m%d-%Hh%Mm%S')
    fields['yaml_name'] = Path(yaml_path).stem
    fields['radio_id'] = radio_id

    return fields


def expand_path(
    path: str | Path,
    sweep: Sweep,
    *,
    radio_id: str | None = None,
    yaml_path=None,
    yaml_relative=False,
) -> str:
    """return an absolute path, allowing for user tokens (~) and {field} in the input."""
    if path is None:
        return None

    fields = _get_default_format_fields(sweep, radio_id=radio_id, yaml_path=yaml_path)
    path = Path(path).expanduser()
    path = Path(str(path).format(**fields))

    if yaml_relative and not path.is_absolute():
        path = Path(yaml_path).parent.absolute() / path
    return str(path.absolute())


def open_store(
    sweep,
    *,
    radio_id: str,
    yaml_path: str,
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
            output_path, sweep, radio_id=radio_id, yaml_path=yaml_path
        )

    if store_backend == 'directory':
        fixed_path = Path(spec_path).with_suffix('.zarr')
    else:
        fixed_path = Path(spec_path).with_suffix('.zarr.zip')

    fixed_path.parent.mkdir(parents=True, exist_ok=True)
    store_backend = channel_analysis.open_store(fixed_path, mode='w' if force else 'a')
    return store_backend


def read_yaml_sweep(
    path: str | Path,
    *,
    adjust_captures={},
    sweep_cls: type[SweepType] = Sweep,
    radio_id=None,
) -> tuple[SweepType, tuple[str, ...]]:
    """build a Sweep struct from the contents of specified yaml file.

    Args:
        path: path to the yaml file
        adjust_captures: update (and override) all fields with the given values

    """

    with open(path, 'rb') as fd:
        text = fd.read()

    # validate first
    msgspec.yaml.decode(text, type=sweep_cls, strict=False, dec_hook=_dec_hook)

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

    tree['captures'] = [
        dict(defaults, **c, **adjust_captures) for c in tree['captures']
    ]

    sweep: Sweep = channel_analysis.builtins_to_struct(
        tree, type=sweep_cls, strict=False, dec_hook=_dec_hook
    )

    # fill formatting fields in paths
    kws = dict(sweep=sweep, radio_id=radio_id, yaml_path=path)

    output_path = expand_path(sweep.output.path, **kws)
    output_spec = msgspec.structs.replace(sweep.output, path=output_path)

    cal_path = expand_path(sweep.radio_setup.calibration, yaml_relative=True, **kws)
    setup_spec = msgspec.structs.replace(sweep.radio_setup, calibration=cal_path)

    sweep = msgspec.structs.replace(sweep, output=output_spec, radio_setup=setup_spec)

    return sweep


def read_tdms_iq(
    path: Path | str,
    duration: float = None,
    *,
    rx_channel_count=1,
    dtype='complex64',
    skip_samples=0,
    xp=np,
) -> tuple['iqwaveform.type_stubs.ArrayLike', FileSourceCapture]:
    from .radio.testing import TDMSFileSource

    source = TDMSFileSource(path=path, rx_channel_count=rx_channel_count)
    capture = source.get_capture_struct()

    source.arm(capture)
    iq, _ = source.read_iq(capture)

    return iq
