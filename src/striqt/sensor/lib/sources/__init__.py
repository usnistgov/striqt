from . import base, null, testing

from .base import (
    SourceBase,
    get_channel_read_buffer_count,
    design_capture_resampler,
)

from .null import NullSetup, NullSource, WarmupSource
from .testing import (
    SingleToneSource,
    SawtoothSource,
    NoiseSource,
    DiracDeltaSource,
    FileSource,
    TDMSFileSource,
    ZarrIQSource,
)

# soapy device is not imported here to allow striqt.sensor imports
# for testing when SoapySDR is not installed


def find_radio_cls_by_name(
    name: str, parent_cls: type[SourceBase] = SourceBase
) -> SourceBase:
    """returns a list of radio subclasses that have been imported"""

    try:
        # first: without optional imports
        radio_cls = base.find_radio_cls_helper(name)
    except AttributeError:
        # then: with optional imports
        from . import soapy
        from ... import devices

        radio_cls = base.find_radio_cls_helper(name)

    return radio_cls


def is_same_resource(radio: SourceBase, radio_setup: base.specs.RadioSetup) -> bool:
    expect_cls = find_radio_cls_by_name(radio_setup.driver)
    if not isinstance(radio, expect_cls):
        return False
    elif radio_setup.resource != radio.resource:
        return False
    else:
        return True
