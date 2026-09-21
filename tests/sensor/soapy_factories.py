"""factories for SoapySDR-backed spec instances and the fake device, shared by the
sensor tests.

Not a conftest: importable by bare name from the sensor test modules only; fixtures
live in the root conftest.
"""

from __future__ import annotations

import functools

import msgspec

import striqt.sensor as ss

MCR = 125e6

CAPTURE_DEFAULTS = {
    'port': 0,
    'center_frequency': 1e9,
    'gain': 0,
    'sample_rate': MCR,
    'duration': 1e-3,
    'analysis_bandwidth': 40e6,
    'host_resample': False,
}

# class attributes that SoapySource reads but does not expose as fields, so a preset
# with a different value has to be a subclass
_CLASS_ATTRS = (
    'rx_enable_delay',
    'stream_all_rx_ports',
    'transport_dtype',
    'shared_rx_sample_clock',
    'transient_holdoff_time',
)


def soapy_capture(**kws) -> ss.specs.SoapyCapture:
    """a SoapyCapture at the fake device's master clock rate, with `kws` overriding
    the defaults"""
    return ss.specs.SoapyCapture(**{**CAPTURE_DEFAULTS, **kws})


@functools.cache
def _source_preset_cls(class_attrs: tuple):
    return msgspec.defstruct(
        'SoapySourcePreset',
        [],
        bases=(ss.specs.SoapySource,),
        namespace=dict(class_attrs),
        kw_only=True,
        frozen=True,
    )


def source_spec(**overrides) -> ss.specs.SoapySource:
    """a SoapySource at the fake device's master clock rate.

    Field overrides are passed through; overrides of the class attributes
    `rx_enable_delay`, `stream_all_rx_ports`, `transport_dtype`,
    `shared_rx_sample_clock` and `transient_holdoff_time` select a cached SoapySource
    subclass that carries them.
    """
    class_attrs = tuple(
        sorted((k, overrides.pop(k)) for k in list(overrides) if k in _CLASS_ATTRS)
    )
    cls = _source_preset_cls(class_attrs) if class_attrs else ss.specs.SoapySource
    return cls(**{'master_clock_rate': MCR, **overrides})


def call_names(device, *names) -> list[str]:
    """the names of the calls recorded by the fake `device` that match `names`, in
    call order"""
    return [c[0] for c in device.calls_named(*names)]
