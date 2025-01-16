"""utility functions for building xarray datasets"""

from __future__ import annotations

import dataclasses
import functools
import msgspec
import numbers
import pickle
import typing

from . import iq_corrections, captures, structs, util
from . import radio

if typing.TYPE_CHECKING:
    import labbench as lb
    import numpy as np
    import pandas as pd
    import xarray as xr
    import channel_analysis
else:
    lb = util.lazy_import('labbench')
    np = util.lazy_import('numpy')
    pd = util.lazy_import('pandas')
    xr = util.lazy_import('xarray')
    channel_analysis = util.lazy_import('channel_analysis')

CAPTURE_DIM = 'capture'
SWEEP_TIMESTAMP_NAME = 'sweep_start_time'
RADIO_ID_NAME = 'radio_id'


def concat_time_dim(datasets: list['xr.Dataset'], time_dim: str) -> 'xr.Dataset':
    """concatenate captured datasets into one along a time axis.

    This can be used to e.g. transform a contiguous sequence
    of spectrogram captures into a single spectrogram.

    Preconditions:
    - all datasets share the same dimension and type.
    - time coordinates based on time_dim are uniformly spaced

    """
    pad_dims = {time_dim: (0, len(datasets[0][time_dim]) * (len(datasets) - 1))}
    ds = datasets[0].pad(pad_dims, constant_values=float('nan'))

    for data_name, var in ds.data_vars.items():
        if time_dim not in var.dims:
            continue
        else:
            axis = var.dims[1:].index(time_dim)

        values = np.concatenate(
            [sub[data_name].isel(capture=0).values for sub in datasets], axis=axis
        )
        var.values[:] = values

    for coord_name, coord in ds.coords.items():
        if time_dim not in coord.dims:
            continue
        time_step = float(coord[1] - coord[0])
        ds[coord_name] = pd.RangeIndex(ds.sizes[coord_name]) * time_step

    return ds


@functools.lru_cache
def coord_template(
    capture_cls: type[structs.RadioCapture],
    channels: tuple[int, ...],
    **alias_dtypes: dict[str, type],
):
    """returns a cached xr.Coordinates object to use as a template for data results"""

    def broadcast_defaults(v, allow_mismatch=False):
        # match the number of channels, duplicating if necessary
        (values,) = captures.broadcast_to_channels(
            channels, v, allow_mismatch=allow_mismatch
        )
        return list(values)

    capture = capture_cls()
    vars = {}

    for field in capture_cls.__struct_fields__:
        entry = getattr(capture, field)

        vars[field] = xr.Variable(
            (CAPTURE_DIM,),
            broadcast_defaults(entry, allow_mismatch=True),
            fastpath=True,
            attrs=get_attrs(capture_cls, field),
        )

    for field, dtype in alias_dtypes.items():
        vars[field] = xr.Variable(
            (CAPTURE_DIM,),
            broadcast_defaults(dtype.type()),
            fastpath=True,
        ).astype(dtype)

    vars[SWEEP_TIMESTAMP_NAME] = xr.Variable(
        (CAPTURE_DIM,),
        broadcast_defaults(pd.Timestamp('now')),
        fastpath=True,
        attrs={'standard_name': 'Sweep start time'},
    )

    vars[RADIO_ID_NAME] = xr.Variable(
        (CAPTURE_DIM,),
        broadcast_defaults('unspecified-radio'),
        fastpath=True,
        attrs={'standard_name': 'Radio hardware ID'},
    ).astype('object')

    return xr.Coordinates(vars)


@functools.lru_cache
def _get_alias_dtypes(output: structs.Output):
    aliases = output.coord_aliases

    alias_dtypes = {}
    for field, entries in aliases.items():
        alias_dtypes[field] = np.array(list(entries.keys())).dtype
    return alias_dtypes


@functools.lru_cache
def get_attrs(struct: type[msgspec.Struct], field: str) -> dict[str, str]:
    """introspect an attrs dict for xarray from the specified field in `struct`"""
    hints = typing.get_type_hints(struct, include_extras=True)

    try:
        metas = hints[field].__metadata__
    except (AttributeError, KeyError):
        return {}

    if len(metas) == 0:
        return {}
    elif len(metas) == 1 and isinstance(metas[0], msgspec.Meta):
        return metas[0].extra
    else:
        raise TypeError(
            'Annotated[] type hints must contain exactly one msgspec.Meta object'
        )


def build_coords(
    capture: structs.RadioCapture, output: structs.Output, radio_id: str, sweep_time
):
    alias_dtypes = _get_alias_dtypes(output)

    if isinstance(capture.channel, numbers.Number):
        channels = (capture.channel,)
    else:
        channels = tuple(capture.channel)

    coords = coord_template(type(capture), channels, **alias_dtypes).copy(deep=True)

    updates = {}

    for c in captures.split_capture_channels(capture):
        alias_hits = captures.evaluate_aliases(c, radio_id=radio_id, output=output)

        for field in coords.keys():
            if field == RADIO_ID_NAME:
                updates.setdefault(field, []).append(radio_id)
                continue

            try:
                value = captures.get_field_value(
                    field, c, radio_id, alias_hits, {SWEEP_TIMESTAMP_NAME: sweep_time}
                )
            except KeyError:
                if field in output.coord_aliases and radio_id is not None:
                    lb.logger.warning(f'warning: no alias name matches in "{field}"')
                continue

            updates.setdefault(field, []).append(value)

    for field, values in updates.items():
        coords[field].values[:] = np.array(values)

    return coords


def _alias_is_in_coord(dataset, alias_spec) -> bool:
    """return whether the given mapping matches coordinate values in dataset"""
    for match_name, match_value in alias_spec.items():
        if match_name in dataset.coords:
            match_coord = dataset.coords[match_name]
        else:
            raise KeyError

        if match_coord.values[0] != match_value:
            # no match
            return False
    else:
        return False


def _assign_alias_coords(capture_data: 'xr.Dataset', aliases):
    for coord_name, coord_spec in aliases.items():
        for alias_value, alias_spec in coord_spec.items():
            if _alias_is_in_coord(capture_data, alias_spec):
                new_coords = {coord_name: (CAPTURE_DIM, [alias_value])}
                capture_data = capture_data.assign_coords(new_coords)
                break

    return capture_data


@dataclasses.dataclass
class ChannelAnalysisWrapper:
    """Inject radio device and capture metadata and coordinates into a channel analysis result"""

    __name__ = 'analyze'

    radio: radio.RadioDevice
    sweep: structs.Sweep
    analysis_spec: list[structs.ChannelAnalysis]
    extra_attrs: dict[str, typing.Any] | None = None

    def __call__(
        self,
        iq: 'channel_analysis.ArrayType',
        sweep_time,
        capture: structs.RadioCapture,
        pickled=False,
    ) -> 'xr.Dataset':
        """Inject radio device and capture info into a channel analysis result."""

        with lb.stopwatch('analysis', logger_level='debug'):
            with lb.stopwatch('analysis: resample/calibrate', logger_level='debug'):
                iq = iq_corrections.resampling_correction(iq, capture, self.radio)

            with lb.stopwatch('build coords', threshold=10e-3, logger_level='debug'):
                coords = build_coords(
                    capture,
                    output=self.sweep.output,
                    radio_id=self.radio.id,
                    sweep_time=sweep_time,
                )

            analysis = channel_analysis.analyze_by_spec(
                iq, capture, spec=self.analysis_spec, expand_dims=(CAPTURE_DIM,)
            )

            analysis = analysis.assign_coords(coords)

            # these are coordinates - drop from attrs
            for name in coords.keys():
                analysis.attrs.pop(name, None)

        if self.extra_attrs is not None:
            analysis.attrs.update(self.extra_attrs)

        analysis[SWEEP_TIMESTAMP_NAME].attrs.update(label='Sweep start time')

        if pickled:
            return pickle.dumps(analysis)
        else:
            return analysis
