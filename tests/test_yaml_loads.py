import striqt.sensor as ss
import msgspec


def test_read(spec_dir):
    ss.read_yaml_spec(spec_dir / 'cw-cpu.yaml')


def test_yaml_schema_binding(spec_dir):
    spec = ss.read_yaml_spec(spec_dir / 'air7101b.yaml')
    ctrl_cls = ss.bindings.air7101b
    schema = ctrl_cls.schema

    assert isinstance(spec.captures[0], schema.capture), 'capture binding mismatch'
    assert isinstance(spec.peripherals, schema.peripherals), 'periph binding mismatch'
    assert isinstance(spec.source, schema.source), 'source binding mismatch'
    assert isinstance(spec, ctrl_cls.sensor.sweep_spec_cls), 'sweep spec binding mismatch'


def test_binding_args():
    union = ss.lib.bindings.get_tagged_sweep_type()
    for struct in union.__args__:  # type: ignore
        msgspec.inspect.type_info(struct)
        assert issubclass(struct, ss.specs.Sweep)


def test_binding_introspect():
    union = ss.lib.bindings.get_tagged_sweep_type()
    msgspec.inspect.type_info(union)
