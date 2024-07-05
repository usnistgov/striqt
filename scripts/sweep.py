from edge_sensor import structs
from edge_sensor.radio import airt
import labbench as lb
from channel_analysis import waveform
import xarray as xr

lb.show_messages('info')

run_spec, sweep_fields = structs.read_yaml_sweep('notebooks/run.yaml')
data = []

with airt.AirT7201B() as sdr:
    for capture in run_spec.captures:
        coords = {f: [getattr(capture, f)] for f in sweep_fields}
        desc = ', '.join([f'{k}={v[0]}' for k,v in coords.items()])

        with lb.stopwatch(f'{desc}: '):
            sdr.arm(capture)
            iq = sdr.acquire()
            analysis = (
                waveform
                .analyze_by_spec(iq, capture, spec=run_spec.channel_analysis)
                .assign_coords(coords)
            )

        for f in sweep_fields:
            del analysis.attrs[f]

        data.append(analysis)

data = xr.combine_by_coords(data)