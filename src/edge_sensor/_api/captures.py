"""wrap channel_analysis for _radio_ captures, together with utility functions for RadioCapture"""

from __future__ import annotations
import dataclasses
import functools
import pickle
import typing

from frozendict import frozendict
import msgspec

import channel_analysis
from . import iq_corrections, structs, util
from . import radio


if typing.TYPE_CHECKING:
    import pandas as pd
    import xarray as xr
    import labbench as lb
    from matplotlib import ticker
else:
    pd = util.lazy_import('pandas')
    xr = util.lazy_import('xarray')
    lb = util.lazy_import('labbench')
    ticker = util.lazy_import('matplotlib.ticker')


CAPTURE_DIM = 'capture'
SWEEP_TIMESTAMP_NAME = 'sweep_start_time'
RADIO_ID_NAME = 'radio_id'


@functools.lru_cache
def _get_unit_formatter(units: str) -> ticker.EngFormatter:
    return ticker.EngFormatter(unit=units)


def _describe_field(capture: channel_analysis.Capture, name: str):
    meta = channel_analysis.structs.get_capture_type_attrs(type(capture))
    attrs = meta[name]
    value = getattr(capture, name)

    if value is None:
        value_str = 'None'
    if attrs.get('units', None) is not None:
        value_str = _get_unit_formatter(attrs['units'])(value)
    else:
        value_str = repr(value)

    return f'{name}={value_str}'


def describe_capture(
    this: structs.RadioCapture | None,
    prev: structs.RadioCapture | None = None,
    *,
    index: int,
    count: int,
) -> str:
    """generate a description of a capture"""
    if this is None:
        if prev is None:
            return 'saving last analysis'
        else:
            return 'performing last analysis'

    diffs = []

    for name in type(this).__struct_fields__:
        if name == 'start_time':
            continue
        value = getattr(this, name)
        if prev is None or value != getattr(prev, name):
            diffs.append(_describe_field(this, name))

    capture_diff = ', '.join(diffs)

    if index is not None:
        progress = str(index+1)

        if count is not None:
            progress = f'{progress}/{count}'

        progress = progress + ' '
    else:
        progress = ''

    return progress + capture_diff


@functools.lru_cache
def coord_template(capture_cls: type[structs.RadioCapture], aliases: tuple[str, ...]):
    """returns a cached xr.Coordinates object to use as a template for data results"""

    capture = capture_cls()
    vars = {}

    for field in capture_cls.__struct_fields__:
        value = getattr(capture, field)

        vars[field] = xr.Variable(
            (CAPTURE_DIM,),
            [value],
            fastpath=True,
            attrs=structs.get_attrs(capture_cls, field),
        )

    for field in aliases:
        vars[field] = xr.Variable(
            (CAPTURE_DIM,),
            [''],
            fastpath=True,
        ).astype('object')

    vars[SWEEP_TIMESTAMP_NAME] = xr.Variable(
        (CAPTURE_DIM,),
        [pd.Timestamp('now')],
        fastpath=True,
        attrs={'standard_name': 'Sweep start time'},
    )
    vars[RADIO_ID_NAME] = xr.Variable(
        (CAPTURE_DIM,),
        ['unspecified-radio'],
        fastpath=True,
        attrs={'standard_name': 'Radio hardware ID'},
    ).astype('object')

    return xr.Coordinates(vars)


def _get_capture_field(
    name,
    capture: structs.Capture,
    radio_id: str,
    aliases: dict[str, typing.Any],
    sweep_time=None,
):
    if hasattr(capture, name):
        value = getattr(capture, name)
    elif name in aliases:
        value = aliases[name]
    elif name == 'radio_id':
        value = radio_id
    elif name == SWEEP_TIMESTAMP_NAME:
        value = sweep_time
    else:
        raise KeyError
    return value


def build_coords(
    capture: structs.RadioCapture, aliases: dict, radio_id: str, sweep_time
):
    coords = coord_template(type(capture), tuple(aliases.keys())).copy(deep=True)

    for field in coords.keys():
        value = _get_capture_field(field, capture, radio_id, aliases, sweep_time)

        if isinstance(value, str):
            # to coerce strings as variable-length types later for storage
            coords[field] = coords[field].astype('object')
        coords[field].values[:] = value

    coords[RADIO_ID_NAME].values[:] = radio_id

    return coords


def _alias_is_in_coord(dataset, alias_spec) -> bool:
    """return whether or not the given mapping matches coordinate values in dataset"""
    for match_name, match_value in alias_spec.items():
        if match_name in dataset.coords:
            match_coord = dataset.coords[match_name]
        else:
            raise KeyError

        if match_coord.values[0] != match_value:
            # no match
            return False
    else:
        return True


def _assign_alias_coords(capture_data: xr.Dataset, aliases):
    for coord_name, coord_spec in aliases.items():
        for alias_value, alias_spec in coord_spec.items():
            if _alias_is_in_coord(capture_data, alias_spec):
                capture_data = capture_data.assign_coords(
                    {coord_name: (CAPTURE_DIM, [alias_value])}
                )
                break

    return capture_data


def _alias_is_in_capture(
    capture: structs.Capture, radio_id: str, aliases: dict[str, typing.Any], alias_spec
) -> bool:
    """return whether or not the given mapping matches coordinate values in dataset"""

    for match_name, match_value in alias_spec.items():
        capture_value = _get_capture_field(match_name, capture, radio_id, aliases)
        if capture_value != match_value:
            # no match
            return False
    else:
        return True


def _evaluate_aliases(capture: structs.Capture, radio_id: str, alias_spec: dict):
    """evaluate the field values"""
    ret = {}
    for coord_name, coord_spec in alias_spec.items():
        for alias_value, alias_spec in coord_spec.items():
            if _alias_is_in_capture(capture, radio_id, ret, alias_spec):
                ret[coord_name] = alias_value
                break
    return ret


def capture_fields_with_aliases(
    capture: structs.Capture, radio_id: str, alias_spec: dict
) -> dict:
    attrs = structs.struct_to_builtins(capture)
    aliases = _evaluate_aliases(capture, radio_id, alias_spec)

    return dict(attrs, **aliases)


@dataclasses.dataclass
class ChannelAnalysisWrapper:
    """Inject radio device and capture metadata and coordinates into a channel analysis result"""

    __name__ = 'analyze'

    radio: radio.RadioDevice
    sweep: structs.Sweep
    analysis_spec: list[channel_analysis.ChannelAnalysis]
    extra_attrs: dict[str, typing.Any] | None = None
    calibration: xr.Dataset | None = None

    def __call__(
        self,
        iq: channel_analysis.ArrayType,
        sweep_time,
        capture: structs.RadioCapture,
        pickled=False,
    ) -> xr.Dataset:
        """Inject radio device and capture info into a channel analysis result."""

        with lb.stopwatch('analyze', logger_level='debug'):
            # for performance, GPU operations are all here in the same thread
            iq = iq_corrections.resampling_correction(
                iq, capture, self.radio, force_calibration=self.calibration
            )
            coords = build_coords(
                capture,
                aliases=self.sweep.output.coord_aliases,
                radio_id=self.radio.id,
                sweep_time=sweep_time,
            )

            analysis = channel_analysis.analyze_by_spec(
                iq, capture, spec=self.analysis_spec
            )

            analysis = analysis.expand_dims((CAPTURE_DIM,)).assign_coords(coords)
            analysis = _assign_alias_coords(analysis, self.sweep.output.coord_aliases)

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
