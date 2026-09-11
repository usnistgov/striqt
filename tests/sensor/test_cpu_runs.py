import striqt.sensor as ss
from striqt.cli import sensor_sweep


def test_run(cpu_sweep_file):
    ss.util.log_verbosity(1)
    sensor_sweep.run(cpu_sweep_file)
