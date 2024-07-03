import xarray as xr

from edge_sensor.radio import airt
from edge_sensor import structs
from channel_analysis import waveform
import labbench as lb

lb.show_messages('info')
sdr = airt.AirT7201B()

run_spec, variables = structs.read_yaml_sweep('run.yaml')

data = []

with sdr:
    for capture in run_spec.captures:
        coords = {f: [getattr(capture, f)] for f in variables}
        print(coords)
        count = sdr.configure(capture)
        iq = sdr.acquire(count)
        analysis = (
            waveform
            .analyze_by_spec(iq, sdr, spec=run_spec.channel_analysis)
            .assign_coords(coords)
        )
        data.append(analysis)

data = xr.combine_by_coords(data)