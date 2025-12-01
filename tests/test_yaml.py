from striqt import sensor
import msgspec
from pathlib import Path

SPEC_PATH = Path(__file__).parent / 'spec.yaml'


def test_read():
    sensor.read_yaml_spec(SPEC_PATH)


def test_yaml_schema_binding():
    spec = sensor.read_yaml_spec(SPEC_PATH)
    assert isinstance(spec.captures[0], sensor.bindings.air7201b.schema.capture), (
        'capture spec type'
    )
    assert isinstance(spec.peripherals, sensor.bindings.air7201b.schema.peripherals), (
        'peripheral spec type'
    )
    assert isinstance(spec.source, sensor.bindings.air7201b.schema.source), (
        'source spec type'
    )
    assert isinstance(spec, sensor.bindings.air7201b.sweep_spec), 'schema spec type'


def test_binding_args():
    union = sensor._lib.bindings.get_tagged_sweep_type()
    for struct in union.__args__:  # type: ignore
        msgspec.inspect.type_info(struct)
        assert issubclass(struct, sensor.Sweep)


def test_binding_introspect():
    union = sensor._lib.bindings.get_tagged_sweep_type()
    msgspec.inspect.type_info(union)
