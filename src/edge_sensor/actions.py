import labbench as lb
import xarray as xr
import msgspec

from channel_analysis import waveform
from channel_analysis.structs import ChannelAnalysis

from .radio import base
from .structs import (
    CAPTURE_DIM,
    Sweep,
    RadioCapture,
    TIMESTAMP_NAME,
    FIELD_ATTRS,
    capture_to_coords,
)


def sense(
    radio: base.RadioDevice,
    capture: RadioCapture,
    analysis: ChannelAnalysis,
    coord_fields: list[str],
) -> xr.Dataset:
    """use the supplied radio device to arm, acquire, and analyze a single capture"""

    desc = ', '.join([
        f'{k}={v}'
        for k, v in msgspec.to_builtins(capture).items()
        if k in coord_fields
    ])

    with lb.stopwatch(f'{desc}: '):
        radio.arm(capture)
        iq, timestamp = radio.acquire()
        coords = capture_to_coords(capture, coord_fields, timestamp=timestamp)
        analysis = waveform.analyze_by_spec(iq, capture, spec=analysis).assign_coords(coords)

    for f in coord_fields:
        del analysis.attrs[f]

    return analysis

def sweep(
    radio: base.RadioDevice, run_spec: Sweep, sweep_fields: list[str]
) -> xr.Dataset:
    data = []
    sweep_fields = tuple(sweep_fields)

    for capture in run_spec.captures:
        analysis = sense(radio, capture, run_spec.channel_analysis, sweep_fields)

        data.append(analysis)

    ds = xr.concat(data, CAPTURE_DIM)

    for k in tuple(sweep_fields) + (TIMESTAMP_NAME,):
        ds[k].attrs.update(FIELD_ATTRS[k])

    ds[k].attrs.update()

    return ds
