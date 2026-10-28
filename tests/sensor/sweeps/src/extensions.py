"""a site-style sensor binding over the synthetic single-tone source.

The module has the shape of an out-of-tree extension module loaded through the
`extensions:` block of a sweep YAML file: capture and peripheral spec subclasses, a
source with a fixed hardware id, and a Y-factor calibration binding derived from the
sensor.
"""

# ruff: noqa: UP007, UP045  msgspec resolves these annotations at runtime on py3.9
from __future__ import annotations

import dataclasses
from math import nan
from typing import Annotated, Optional, Union

import striqt.analysis as sa
from striqt.sensor import bindings, peripherals, sources, specs

RADIO_ID = '48b02d17a587'

MastHeight = Annotated[float, specs.Meta('Mast height', 'm', ge=0)]
Elevation = Annotated[
    float, specs.Meta('Elevation pointing angle', 'deg', ge=-10, le=90)
]
Azimuth = Annotated[float, specs.Meta('Azimuth pointing angle', 'deg', ge=-180, le=180)]
SwitchInput = Annotated[int, specs.Meta('RF input switch index', ge=0, le=1)]
Label = Union[str, tuple[str, ...], None]


class SiteCapture(specs.SingleToneCapture, frozen=True, kw_only=True):
    center_frequency: specs.types.CenterFrequency
    gain: specs.types.Gain

    mast_height: Optional[MastHeight] = None
    elevation: Optional[Elevation] = None
    azimuth: Optional[Azimuth] = None
    switch_input: Optional[SwitchInput] = 0
    azimuth_offset: float = 0
    elevation_offset: float = 0

    radio_name: Label = ''
    site_name: Label = ''
    channel_name: Label = ''
    antenna_name: Label = ''
    antenna_model: Label = ''
    antenna_polarization: Label = ''
    antenna_index: Union[int, tuple[int, ...], None] = None

    def __post_init__(self):
        super().__post_init__()
        if (self.azimuth is None) != (self.elevation is None):
            raise ValueError('azimuth and elevation must be set together')


class SiteSurveyCapture(SiteCapture, frozen=True, kw_only=True):
    latitude: float = nan
    longitude: float = nan
    azimuth_repeat: int = 0
    tx1_power: float = nan
    test_case: int = 0


class SitePeripherals(specs.Peripherals, frozen=True, kw_only=True):
    control_orientation: bool = False
    connect_x410: bool = True


class SiteToneSource(sources.SingleToneSource):
    def get_id(self):
        return RADIO_ID


def _schema(capture_cls):
    return bindings.Schema(
        capture=capture_cls,
        arm_like=capture_cls,
        source=specs.FunctionSource,
        init_like=specs.FunctionSource,
        peripherals=SitePeripherals,
    )


_site_sensor = bindings.Sensor(
    source_cls=SiteToneSource, peripherals_cls=peripherals.NoPeripherals
)

site_single_tone = bindings.bind_sensor(
    'site_single_tone', _site_sensor, _schema(SiteCapture)
)

site_survey = bindings.bind_sensor(
    'site_survey', _site_sensor, _schema(SiteSurveyCapture)
)

site_single_tone_calibration = bindings.bind_sensor(
    'site_single_tone_calibration',
    _site_sensor,
    dataclasses.replace(_schema(SiteCapture), action=specs.ManualYFactorCalibration),
)

site_cellular = bindings.bind_sensor(
    'site_cellular',
    _site_sensor,
    dataclasses.replace(_schema(SiteCapture), action=specs.Cellular5GNRSweep),
)


class SiteQuickAnalyses(
    specs.AnalysisGroup, frozen=True, kw_only=True, omit_defaults=True
):
    """the analysis group of the measurements in ``fragments/analysis/quick.yaml``, and
    no others"""

    spectrogram: Optional[sa.specs.Spectrogram] = None
    power_spectral_density: Optional[sa.specs.PowerSpectralDensity] = None
    channel_power_time_series: Optional[sa.specs.ChannelPowerTimeSeries] = None


site_quick = bindings.bind_sensor(
    'site_quick',
    _site_sensor,
    dataclasses.replace(_schema(SiteCapture), analysis=SiteQuickAnalyses),
)
