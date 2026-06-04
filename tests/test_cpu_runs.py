import striqt.analysis as sa
import striqt.sensor as ss
from striqt.cli import sensor_sweep
import sys

def test_run(cpu_sweep_file):
    ss.util.log_verbosity(-1)
    sensor_sweep.execute(cpu_sweep_file)