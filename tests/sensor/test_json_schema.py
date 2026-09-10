import striqt.analysis as sa
import striqt.sensor as ss


def test_schema_generation(spec_dir):
    sweep = ss.read_yaml_spec(spec_dir / 'cw-cpu.yaml')
    sa.specs.helpers.json_schema(type(sweep))
