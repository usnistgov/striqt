from striqt import sensor
import msgspec

def test_sweep_union_args():
    union = sensor._lib.bindings.get_tagged_sweep_type()
    for struct in union.__args__: # type: ignore
        msgspec.inspect.type_info(struct)
        assert issubclass(struct, sensor.Sweep)


def test_sweep_union_introspect():
    union = sensor._lib.bindings.get_tagged_sweep_type()
    msgspec.inspect.type_info(union)