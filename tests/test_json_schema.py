import striqt.sensor as ss
import striqt.analysis as sa


def test_schema_generation(spec_dir):
    sweep = ss.read_yaml_spec(spec_dir / 'cw-cpu.yaml')
    sa.specs.helpers.json_schema(type(sweep))
