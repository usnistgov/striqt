"""utility functions for building xarray datasets"""

from __future__ import annotations

import dataclasses
import logging
import msgspec
import numbers
import pickle
import typing

from . import captures, specs, util
from .sources import SourceBase

from striqt.analysis import register
from striqt.analysis.lib.dataarrays import CAPTURE_DIM, PORT_DIM, AcquiredIQ  # noqa: F401
from striqt.analysis.lib.util import log_capture_context, stopwatch

from iqwaveform.util import is_cupy_array

if typing.TYPE_CHECKING:
    import labbench as lb
    import numpy as np
    import pandas as pd
    import xarray as xr
    import striqt.analysis as striqt_analysis
else:
    lb = util.lazy_import('labbench')
    np = util.lazy_import('numpy')
    pd = util.lazy_import('pandas')
    xr = util.lazy_import('xarray')
    striqt_analysis = util.lazy_import('striqt.analysis')


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


@util.lru_cache()
def coord_template(
    capture_cls: type[specs.RadioCapture],
    ports: tuple[int, ...],
    **alias_dtypes: dict[str, type],
):
    """returns a cached xr.Coordinates object to use as a template for data results"""

    def broadcast_defaults(v, allow_mismatch=False):
        # match the number of ports, duplicating if necessary
        (values,) = captures.broadcast_to_ports(ports, v, allow_mismatch=allow_mismatch)
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

    vars[RADIO_ID_NAME] = xr.Variable(
        (CAPTURE_DIM,),
        broadcast_defaults('unspecified-radio'),
        fastpath=True,
        attrs={'standard_name': 'Radio hardware ID'},
    ).astype('object')

    return xr.Coordinates(vars)


@util.lru_cache()
def _get_alias_dtypes(output: specs.Output):
    aliases = output.coord_aliases

    alias_dtypes = {}
    for field, entries in aliases.items():
        alias_dtypes[field] = np.array(list(entries.keys())).dtype
    return alias_dtypes


@util.lru_cache()
def get_attrs(struct: type[specs.SpecBase], field: str) -> dict[str, str]:
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
    capture: specs.RadioCapture, output: specs.Output, radio_id: str, sweep_time
):
    alias_dtypes = _get_alias_dtypes(output)

    if isinstance(capture.port, numbers.Number):
        ports = (capture.port,)
    else:
        ports = tuple(capture.port)

    coords = coord_template(type(capture), ports, **alias_dtypes).copy(deep=True)

    updates = {}

    for c in captures.split_capture_ports(capture):
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
class AnalysisCaller:
    """Inject radio device and capture metadata and coordinates into a channel analysis result"""

    radio: SourceBase
    sweep: specs.Sweep
    analysis_spec: list[specs.Measurement]
    extra_attrs: dict[str, typing.Any] | None = None
    correction: bool = False

    @util.stopwatch('✓', 'analysis')
    def __call__(
        self,
        iq: AcquiredIQ,
        sweep_time,
        capture: specs.RadioCapture,
        pickled=False,
        overwrite_x=True,
        block_each=False,
        delayed=True,
    ) -> 'xr.Dataset' | dict[str] | str:
        """Inject radio device and capture info into a channel analysis result."""

        # wait to import until here to avoid a circular import
        from . import iq_corrections

        with register.measurement.cache_context():
            if self.correction:
                with stopwatch(
                    'resample, filter, calibrate', logger_level=logging.DEBUG
                ):
                    iq = iq_corrections.resampling_correction(
                        iq.raw, capture, self.radio, overwrite_x=overwrite_x
                    )

            result = striqt_analysis.lib.dataarrays.analyze_by_spec(
                iq,
                capture=capture,
                spec=self.analysis_spec,
                block_each=block_each,
                as_xarray='delayed' if delayed else True,
                expand_dims=(CAPTURE_DIM,),
            )

        if iq.unscaled_peak is not None:
            peak = iq.unscaled_peak
            if is_cupy_array(peak):
                peak = peak.get()
            extra_data = {'unscaled_iq_peak': peak}
        else:
            extra_data = {}

        if delayed:
            result = DelayedDataset(
                delayed=result,
                capture=capture,
                sweep=self.sweep,
                radio_id=self.radio.id,
                sweep_time=sweep_time,
                extra_attrs=self.extra_attrs,
                extra_data=extra_data
            )

        else:
            result.update(extra_data)

        if pickled:
            return pickle.dumps(result)
        else:
            return result


@dataclasses.dataclass()
class DelayedDataset:
    delayed: dict[str, striqt_analysis.lib.dataarrays.DelayedDataArray]
    capture: specs.Capture
    sweep: specs.Sweep
    radio_id: str
    sweep_time: typing.Any
    extra_attrs: dict | None = None
    extra_data: dict | None = None

    def set_extra_data(self, extra_data: dict[str]) -> None:
        self.extra_data = self.extra_data | extra_data

    def to_xarray(self) -> 'xr.Dataset':
        """complete any remaining calculations, transfer from the device, and build an output dataset"""

        with stopwatch(
            'package xarray',
            'analysis',
            threshold=10e-3,
            logger_level=logging.DEBUG,
        ):
            analysis = striqt_analysis.lib.dataarrays.package_analysis(
                self.capture, self.delayed, expand_dims=(CAPTURE_DIM,)
            )

        with stopwatch(
            'build coords',
            'analysis',
            threshold=10e-3,
            logger_level=logging.DEBUG,
        ):
            coords = build_coords(
                self.capture,
                output=self.sweep.output,
                radio_id=self.radio_id,
                sweep_time=self.sweep_time,
            )
            analysis = analysis.assign_coords(coords)

            # these are coordinates - drop from attrs
            for name in coords.keys():
                analysis.attrs.pop(name, None)

            if self.extra_attrs is not None:
                analysis.attrs.update(self.extra_attrs)

            analysis[SWEEP_TIMESTAMP_NAME].attrs.update(label='Sweep start time')

        with stopwatch(
            'add peripheral data',
            'analysis',
            threshold=10e-3,
            logger_level=logging.DEBUG,
        ):
            if self.extra_data is not None:
                new_arrays = {}
                allowed_capture_shapes = (0, 1, analysis.capture.size)

                for k, v in self.extra_data.items():
                    if not isinstance(v, xr.DataArray):
                        dims = [CAPTURE_DIM] + [f'{k}_dim{n}' for n in range(1,v.ndim)]
                        v = xr.DataArray(v, dims=dims)

                    elif v.dims[0] != CAPTURE_DIM:
                        v = v.expand_dims({CAPTURE_DIM: analysis.capture.size})

                    if v.sizes[CAPTURE_DIM] not in allowed_capture_shapes:
                        raise ValueError(
                            f'size of first axis of extra data "{k}" must be one '
                            f'of {allowed_capture_shapes}'
                        )

                    new_arrays[k] = v

                analysis = analysis.assign(new_arrays)

        return analysis


def analyze_capture(
    radio: SourceBase,
    iq: util.ArrayType,
    capture: specs.Capture,
    sweep: specs.Sweep,
    *,
    sweep_start_time=None,
    correction: bool = False,
) -> 'xr.Dataset':
    """a convenience function to analyze the output of a radio.acquire().

    Arguments:
        iq: the array containing
        capture: the specification structure the IQ capture returned from .acquire()
        correction: set True if the corresponding argument to .acquire() was False

    Returns:
        an xarray dataset containing the analysis specified by sweep.analysis
    """

    # metadata fields
    attrs = sweep.radio_setup.todict() | sweep.description.todict()

    func = AnalysisCaller(
        radio=radio,
        sweep=sweep,
        analysis_spec=sweep.analysis,
        extra_attrs=attrs,
        correction=correction,
    )

    return func(iq, sweep_time=sweep_start_time, capture=capture)
