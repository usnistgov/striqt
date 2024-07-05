from .radio import base
from .structs import Sweep

import labbench as lb
from channel_analysis import waveform
import xarray as xr
import pandas as pd


def sweep(radio: base.RadioDevice, run_spec: Sweep, sweep_fields: list[str]) -> xr.Dataset:
    data = []
    for capture in run_spec.captures:
        # treat swept fields as coordinates/indices
        coords = {f: [getattr(capture, f)] for f in sweep_fields}
        desc = ', '.join([f'{k}={v[0]}' for k,v in coords.items()])

        with lb.stopwatch(f'{desc}: '):
            radio.arm(capture)
            iq, timestamp = radio.acquire()
            coords['timestamp'] = [timestamp]
            analysis = (
                waveform
                .analyze_by_spec(iq, capture, spec=run_spec.channel_analysis)
                .assign_coords(coords)
            )

        # remove swept fields from the metadata
        for f in sweep_fields:
            del analysis.attrs[f]

        data.append(analysis)

    return xr.combine_by_coords(data)