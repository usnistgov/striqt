"""factories for the site-style binding defined in sweeps/src/extensions.py

Importing this module registers the binding by reading a site sweep through the
public API, exactly as the sensor-sweep CLI does for out-of-tree extension modules.
"""

from __future__ import annotations

import sys

from conftest import SITE_DIR
from hypothesis import strategies as st
from sweep_strategies import SOURCE

import striqt.sensor as ss

if 'site_single_tone' not in ss.lib.bindings.registry:
    ss.read_yaml_spec(SITE_DIR / 'site-cpu.yaml')

EXT = sys.modules['extensions']
RADIO_ID: str = EXT.RADIO_ID
OTHER_ID = 'a1b2c3d4e5f6'

SiteCaptureCls = EXT.SiteCapture
SurveyCaptureCls = EXT.SiteSurveyCapture
SiteSweepCls = EXT.site_single_tone.sensor.sweep_spec_cls
SurveySweepCls = EXT.site_survey.sensor.sweep_spec_cls
SiteCalCaptureCls = EXT.site_single_tone_calibration.schema.capture

Remap = ss.specs.CaptureRemap

CHANNEL_NAMES = {3750e6: '3750 MHz', 3829.8e6: '3830 MHz', 3900e6: '3900 MHz'}
ANTENNA_MODELS = {'Omni': 'OmniModel', '1x32': 'PanelModel'}
POLARIZATIONS = {'Omni': 'Vertical', '1x32': 'Linear +45'}
SWITCHED_ANTENNA_INDEX = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
CHANNEL_NAME_REMAP = Remap(key='center_frequency', lookup=CHANNEL_NAMES, default=None)

# python mirror of sweeps/sites/*.yaml
SITE_ADJUST = {
    'defaults': {
        'channel_name': CHANNEL_NAME_REMAP,
        'antenna_index': Remap(key='port', lookup={0: 0, 1: 1}),
        'antenna_name': 'Unspecified',
        'antenna_polarization': 'Unspecified',
        'antenna_model': Remap(key='antenna_name', lookup=ANTENNA_MODELS),
    },
    RADIO_ID: {
        'radio_name': 'radio02',
        'site_name': 'WAPA-north',
        'gain': 0,
        'mast_height': 1.7,
        'azimuth_offset': 150,
        'elevation_offset': -0,
        'antenna_name': Remap(key='port', lookup={0: 'Omni', 1: '1x32'}),
        'antenna_polarization': Remap(key='antenna_name', lookup=POLARIZATIONS),
        'antenna_model': Remap(key='antenna_name', lookup=ANTENNA_MODELS),
    },
    OTHER_ID: {
        'radio_name': 'radio05',
        'site_name': 'bench',
        'mast_height': None,
        'antenna_name': 'Omni',
        'gain': Remap(key=('channel_name',), lookup={'3750 MHz': 0}, required=False),
        'antenna_index': Remap(
            key=('switch_input', 'port'), lookup=SWITCHED_ANTENNA_INDEX
        ),
    },
}


def make_site_capture_kws(**kws) -> dict:
    return {
        'port': 0,
        'center_frequency': 3750e6,
        'gain': -10,
        'sample_rate': 53.76e6,
        'duration': 1e-3,
        **kws,
    }


def make_site_capture(cls=SiteCaptureCls, **kws):
    return cls(**make_site_capture_kws(**kws))


def make_site_sweep(
    captures=(), loops=(), adjust_captures=None, cls=SiteSweepCls, **kws
):
    if adjust_captures is not None:
        kws['adjust_captures'] = adjust_captures
    if len(captures) == 0:
        captures = (make_site_capture(cls=cls.schema.capture),)
    return cls(source=SOURCE, captures=tuple(captures), loops=tuple(loops), **kws)


center_frequency_keys = st.sampled_from(tuple(CHANNEL_NAMES))
