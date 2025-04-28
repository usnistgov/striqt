"""just a stub for now in case we change this in the future"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
import typing

import msgspec

import channel_analysis
from channel_analysis import load, dump  # noqa: F401

from . import util, captures, structs, xarray_ops

if typing.TYPE_CHECKING:
    import iqwaveform
    import labbench as lb
    import numpy as np
    import pandas as pd
    import xarray as xr
else:
    iqwaveform = util.lazy_import('iqwaveform')
    lb = util.lazy_import('labbench')
    np = util.lazy_import('numpy')
    pd = util.lazy_import('pandas')
    xr = util.lazy_import('xarray')


SweepType = typing.TypeVar('SweepType', bound=structs.Sweep)


def _dec_hook(type_, obj):
    if typing.get_origin(type_) is pd.Timestamp:
        return pd.to_datetime(obj)
    else:
        return obj


def _get_default_format_fields(
    sweep: structs.Sweep, *, radio_id: str | None = None, yaml_path
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
    sweep: structs.Sweep,
    *,
    radio_id: str | None = None,
    relative_to_file=None,
) -> str:
    """return an absolute path, allowing for user tokens (~) and {field} in the input."""
    if path is None:
        return None

    fields = _get_default_format_fields(
        sweep, radio_id=radio_id, yaml_path=relative_to_file
    )
    path = Path(path).expanduser()
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


def read_yaml_sweep(
    path: str | Path,
    *,
    adjust_captures={},
    sweep_cls: type[SweepType] = structs.Sweep,
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

    sweep: structs.Sweep = channel_analysis.builtins_to_struct(
        tree, type=sweep_cls, strict=False, dec_hook=_dec_hook
    )

    # fill formatting fields in paths
    kws = dict(sweep=sweep, radio_id=radio_id)

    output_path = expand_path(sweep.output.path, **kws)
    output_spec = msgspec.structs.replace(sweep.output, path=output_path)

    cal_path = expand_path(sweep.radio_setup.calibration, relative_to_file=path, **kws)
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
) -> tuple['iqwaveform.type_stubs.ArrayLike', structs.FileSourceCapture]:
    from .radio.testing import TDMSFileSource

    source = TDMSFileSource(path=path, rx_channel_count=rx_channel_count)
    capture = source.get_capture_struct()

    source.arm(capture)
    iq, _ = source.read_iq(capture)

    return iq


class SweepDataManager:
    def __init__(
        self,
        sweep_spec: structs.Sweep | str | Path,
        *,
        radio_id: str,
        output_path: str | None = None,
        store_kind: str | None = None,
        force: bool = False,
    ):
        if isinstance(sweep_spec, structs.Sweep):
            self.sweep_spec = sweep_spec
            yaml_path = None
        elif isinstance(sweep_spec, (str, Path)):
            self.sweep_spec = read_yaml_sweep(sweep_spec)
            yaml_path = sweep_spec

        if output_path is None:
            output_path = self.sweep_spec.output.path

        self.output_path = expand_path(
            self.output_path,
            self.sweep_spec,
            radio_id=self.radio_id,
            relative_to_file=yaml_path,
        )

        if store_kind is None:
            self.store_kind = self.sweep_spec.output.store.lower()
        else:
            self.store_kind = store_kind.lower()

        self.store = None
        self.radio_id = radio_id
        self.force = force

        self.clear()

    def clear(self):
        self.pending_data: list[xr.Dataset] = []

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    def open(self):
        raise NotImplementedError

    def close(self):
        self.flush()

    def append(self, capture_data: xr.Dataset):
        raise NotImplementedError

    def flush(self):
        raise NotImplementedError


class SweepStoreManager(SweepDataManager):
    """concatenates the data from each capture and dumps to a zarr data store"""

    def append(self, capture_data: xr.Dataset | None):
        if capture_data is None:
            return
        else:
            self.pending_data.append(capture_data)

    def open(self):
        if self.store_kind == 'directory':
            fixed_path = Path(self.output_path).with_suffix('.zarr')
        elif self.store_kind == 'zip':
            fixed_path = Path(self.output_path).with_suffix('.zarr.zip')
        else:
            raise ValueError(f'unsupported store type {self.store_kind!r}')

        fixed_path.parent.mkdir(parents=True, exist_ok=True)

        self.store = open_store(fixed_path, mode='w' if self.force else 'a')

    def close(self):
        super().close()
        self.store.close()

    def flush(self):
        if len(self.pending_data) == 0:
            return

        with lb.stopwatch('build dataset'):
            data_captures = xr.concat(self.pending_data, xarray_ops.CAPTURE_DIM)

        with lb.stopwatch('dump data'):
            channel_analysis.dump(self.store, data_captures)

        self.clear()
