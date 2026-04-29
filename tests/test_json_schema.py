import pytest
import striqt.sensor as ss
import striqt.analysis as sa

def test_schema_generation():
    sweep = ss.read_yaml_spec('synthetic_waveforms/cw-cpu.yaml')
    sa.specs.helpers.json_schema(type(sweep))