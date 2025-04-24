from . import base, null, testing

from .base import (
    RadioDevice,
    get_channel_read_buffer_count,
    design_capture_filter,
)

from .null import NullSource
from .testing import (
    SingleToneSource,
    SawtoothSource,
    NoiseSource,
    DiracDeltaSource,
    FileSource,
    TDMSFileSource,
    ZarrIQSource,
)

# soapy device is not imported here to allow edge_sensor imports
# for testing when SoapySDR is not installed


def find_radio_cls_by_name(
    name: str, parent_cls: type[RadioDevice] = RadioDevice
) -> RadioDevice:
    """returns a list of radio subclasses that have been imported"""

    try:
        # first: without optional imports
        radio_cls = base._find_radio_cls_helper(name)
    except AttributeError:
        # then: with optional imports
        from . import soapy
        from ... import radios

        radio_cls = base._find_radio_cls_helper(name)

    return radio_cls


def is_same_resource(radio: RadioDevice, radio_setup: 'base.RadioSetup') -> bool:
    expect_cls = find_radio_cls_by_name(radio_setup.driver)
    if not isinstance(radio, expect_cls):
        print('wrong type')
        return False
    if (
        getattr(radio, 'resource', None) is not None
        and radio.resource != radio_setup.resource
    ):
        print('wrong resource: expected ', radio_setup.resource, ' got ', radio.resource)
        return False
    for name, value in radio_setup.device_args.items():
        if not hasattr(radio, name):
            print('radio is missing argument')
            return False
        if getattr(radio, name) != value:
            print('expected radio value ', value, ' but got ', getattr(radio, name))
            return False
    return True
