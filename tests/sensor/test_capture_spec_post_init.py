"""__post_init__ validation of the sensor capture specs"""

from __future__ import annotations

import pytest
from conftest import raises_on_both_paths
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sweep_strategies import (
    consistent_port_gain,
    mismatched_port_gain,
    scalar_port_tuple_gain,
)

import striqt.sensor as ss

SoapyCapture = ss.specs.SoapyCapture
FileCapture = ss.specs.FileCapture
BASE = {'center_frequency': 1e9, 'duration': 1e-3, 'sample_rate': 1e6}
SCALAR_GAIN_MSG = 'gain must be a single number unless multiple ports are specified'
GAIN_COUNT_MSG = 'gain, when specified as a tuple, must match port count'
FILE_RATE_MSG = 'backend_sample_rate is fixed by the file source'
PROPERTY = settings(
    suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None
)


class TestSoapyCapture:
    @given(port_gain=consistent_port_gain())
    @PROPERTY
    def test_consistent_port_gain_accepted(self, construct, port_gain):
        port, gain = port_gain
        capture = construct(SoapyCapture, port=port, gain=gain, **BASE)
        assert capture.port == port
        assert capture.gain == gain

    @given(port_gain=scalar_port_tuple_gain())
    @PROPERTY
    def test_tuple_gain_with_scalar_port_rejected(self, port_gain):
        port, gain = port_gain
        raises_on_both_paths(
            SoapyCapture, ValueError, SCALAR_GAIN_MSG, port=port, gain=gain, **BASE
        )

    @given(port_gain=mismatched_port_gain())
    @PROPERTY
    def test_gain_tuple_length_must_match_ports(self, port_gain):
        port, gain = port_gain
        raises_on_both_paths(
            SoapyCapture, ValueError, GAIN_COUNT_MSG, port=port, gain=gain, **BASE
        )

    def test_replace_revalidates_gain(self):
        capture = SoapyCapture(port=(0, 1), gain=0.0, **BASE)
        with pytest.raises(ValueError, match=GAIN_COUNT_MSG):
            capture.replace(gain=(1.0,))

    def test_center_frequency_tuple_length_is_unchecked(self, construct):
        # only gain is validated against the port count
        kws = dict(BASE, center_frequency=(1e9, 2e9, 3e9))
        capture = construct(SoapyCapture, port=(0, 1), gain=0.0, **kws)
        assert capture.center_frequency == (1e9, 2e9, 3e9)

    def test_list_port_from_dict_is_coerced(self):
        capture = SoapyCapture.from_dict(dict(port=[0, 1], gain=[1.0, 2.0], **BASE))
        assert capture.port == (0, 1)
        assert capture.gain == (1.0, 2.0)
        hash(capture)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            'port is a VarTuple union that SpecBase never freezes on direct '
            'construction; the list then fails the lru_cache in _validate_multichannel'
        ),
    )
    def test_list_port_direct_is_frozen(self):
        capture = SoapyCapture(port=[0, 1], gain=0.0, **BASE)
        assert capture.port == (0, 1)

    def test_air7101b_capture_inherits_gain_check(self):
        cls = ss.bindings.air7101b.schema.capture
        assert issubclass(cls, SoapyCapture)
        with pytest.raises(ValueError, match=GAIN_COUNT_MSG):
            cls(port=(0, 1), gain=(1.0,), **BASE)


class TestFileCapture:
    def test_backend_sample_rate_none_ok(self, construct):
        capture = construct(FileCapture, port=0, duration=1e-3, sample_rate=1e6)
        assert capture.backend_sample_rate is None

    @given(rate=st.floats(min_value=1e3, max_value=1e9))
    @PROPERTY
    def test_backend_sample_rate_rejected(self, rate):
        raises_on_both_paths(
            FileCapture,
            TypeError,
            FILE_RATE_MSG,
            port=0,
            duration=1e-3,
            sample_rate=1e6,
            backend_sample_rate=rate,
        )

    def test_replace_backend_sample_rate_rejected(self):
        capture = FileCapture(port=0, duration=1e-3, sample_rate=1e6)
        with pytest.raises(TypeError, match=FILE_RATE_MSG):
            capture.replace(backend_sample_rate=1e6)
