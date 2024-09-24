"""Integrate radio capture information with channel_analysis evaluation and results"""

from __future__ import annotations
import dataclasses
import functools
import pickle
import typing

from frozendict import frozendict

import channel_analysis
from . import iq_corrections, structs, util
from .. import radio

if typing.TYPE_CHECKING:
    import pandas as pd
    import xarray as xr
    import labbench as lb
else:
    pd = util.lazy_import('pandas')
    xr = util.lazy_import('xarray')
    lb = util.lazy_import('labbench')


CAPTURE_DIM = 'capture'
CAPTURE_TIMESTAMP_NAME = 'capture_time'
SWEEP_TIMESTAMP_NAME = 'sweep_time'
RADIO_ID_NAME = 'radio_id'


@functools.lru_cache
def coord_template(external_fields: frozendict[str, typing.Any]):
    """returns a cached xr.Coordinates object to use as a template for data results"""

    capture = structs.RadioCapture()
    coords = {}

    for field in structs.RadioCapture.__struct_fields__:
        if field == 'external':
            continue
        value = getattr(capture, field)

        coords[field] = xr.Variable((CAPTURE_DIM,), [value], fastpath=True)

    for field, value in external_fields.items():
        coords[field] = xr.Variable((CAPTURE_DIM,), [value], fastpath=True)

    coords[CAPTURE_TIMESTAMP_NAME] = xr.Variable(
        (CAPTURE_DIM,), [pd.Timestamp('now')], fastpath=True
    )
    coords[SWEEP_TIMESTAMP_NAME] = xr.Variable(
        (CAPTURE_DIM,), [pd.Timestamp('now')], fastpath=True
    )
    coords[RADIO_ID_NAME] = xr.Variable(
        (CAPTURE_DIM,), ['unspecified-radio'], fastpath=True
    ).astype('object')

    return xr.Coordinates(coords)


@dataclasses.dataclass
class ChannelAnalysisWrapper:
    """Inject radio device and capture metadata and coordinates into a channel analysis result"""

    __name__ = 'analyze'

    radio: radio.RadioDevice
    analysis_spec: list[channel_analysis.ChannelAnalysis]
    extra_attrs: dict[str, typing.Any] | None = None
    calibration: xr.Dataset | None = None

    def __call__(
        self,
        iq: channel_analysis.ArrayType,
        capture_time,
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
            coords = self.get_coords(
                capture, capture_time=capture_time, sweep_time=sweep_time
            )

            analysis = channel_analysis.analyze_by_spec(
                iq, capture, spec=self.analysis_spec
            )

            analysis = analysis.expand_dims((CAPTURE_DIM,)).assign_coords(coords)

            # these are coordinates - drop from attrs
            for name in coords.keys():
                analysis.attrs.pop(name, None)

        if self.extra_attrs is not None:
            analysis.attrs.update(self.extra_attrs)

        analysis[CAPTURE_TIMESTAMP_NAME].attrs.update(label='Capture start time')
        analysis[SWEEP_TIMESTAMP_NAME].attrs.update(label='Sweep start time')

        if pickled:
            return pickle.dumps(analysis)
        else:
            return analysis

    def get_coords(self, capture: structs.RadioCapture, capture_time, sweep_time):
        coords = coord_template(capture.external).copy(deep=True)

        for field in coords.keys():
            if field in capture.__struct_fields__:
                value = getattr(capture, field)
            elif field in capture.external:
                value = capture.external[field]
            elif field == CAPTURE_TIMESTAMP_NAME:
                value = capture_time
            elif field == SWEEP_TIMESTAMP_NAME:
                value = sweep_time

            if isinstance(value, str):
                # to coerce strings as variable-length types later for storage
                coords[field] = coords[field].astype('object')
            coords[field].values[:] = value

        coords[RADIO_ID_NAME].values[:] = self.radio.id

        return coords
