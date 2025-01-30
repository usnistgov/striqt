from . import base, null, testing

from .base import (
    RadioDevice,
    is_same_resource,
    get_channel_read_buffer_count,
    design_capture_filter,
)

from .null import NullSource
from .testing import SingleToneSource, SawtoothSource, NoiseSource, TDMSFileSource

# soapy device is not imported here to allow edge_sensor imports
# for testing when SoapySDR is not installed

def _find_radio_cls_helper(
    name: str, parent_cls: type[RadioDevice] = RadioDevice
) -> RadioDevice:
    """returns a list of radio subclasses that have been imported"""

    mapping = base._list_radio_classes(parent_cls)

    if name in mapping:
        return mapping[name]
    else:
        raise AttributeError(
            f'invalid driver {repr(name)}. valid names: {tuple(mapping.keys())}'
        )

def find_radio_cls_by_name(
    name: str, parent_cls: type[RadioDevice] = RadioDevice
) -> RadioDevice:
    """returns a list of radio subclasses that have been imported"""

    try:
        radio_cls = find_radio_cls_by_name(name)
    except AttributeError:
        # try importing optional modules and repeat
        from . import soapy
        from ... import radios

        radio_cls = find_radio_cls_by_name(name)

    return radio_cls
