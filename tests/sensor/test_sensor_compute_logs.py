"""striqt.sensor.lib.compute.logs: the ADC and IF overload warnings derived from the
headroom that a SoapySDR source reports per port"""

from __future__ import annotations

import logging

import pytest
from soapy_factories import soapy_capture
from synthetic_sources import SCALE_ONLY, make_capture, make_sweep

import striqt.analysis as sa
import striqt.sensor as ss
from striqt.sensor.lib import compute
from striqt.sensor.lib.compute import logs

AIR = ss.bindings.air7101b

# the capture under test spans two ports at 1 GHz and 2 GHz, and the sweep also
# visits 2 GHz on port 0 at 30 dB more gain
CAPTURE = soapy_capture(port=(0, 1), center_frequency=(1e9, 2e9), gain=0.0)
HIGH_GAIN = soapy_capture(port=0, center_frequency=2e9, gain=30.0)
SWEEP = AIR.sensor.sweep_spec_cls(
    source=AIR.schema.source(), captures=(CAPTURE, HIGH_GAIN)
)
TONE = make_capture('single_tone', **SCALE_ONLY)
TONE_SWEEP = make_sweep('single_tone', (TONE,))

ADC_OVERLOAD = 'adc overload on port 0 (1000 MHz)'
IF_OVERLOAD = 'if overload at port 0 (onto 2000 MHz)'


# %% _adc_overload_message


def test_adc_overload_names_the_port_at_or_below_zero_headroom():
    extra = {'adc_headroom': (-1.0, 3.0)}
    assert logs._adc_overload_message(extra, CAPTURE) == ADC_OVERLOAD


def test_adc_overload_zero_headroom_counts():
    message = logs._adc_overload_message({'adc_headroom': (3.0, 0.0)}, CAPTURE)
    assert message == 'adc overload on port 1 (2000 MHz)'


ADC_SILENT = {
    'positive_headroom': ({'adc_headroom': (1.0, 3.0)}, CAPTURE),
    'no_headroom_key': ({'if_headroom': (-1.0, 3.0)}, CAPTURE),
    'not_a_soapy_capture': ({'adc_headroom': (-1.0, -1.0)}, TONE),
}


@pytest.mark.parametrize('case', list(ADC_SILENT), ids=list(ADC_SILENT))
def test_adc_overload_is_none(case):
    extra, capture = ADC_SILENT[case]
    assert logs._adc_overload_message(extra, capture) is None


# %% _if_overload_message


def test_if_overload_follows_the_im3_rule():
    """third-order products of a channel visited at higher gain grow 2 dB per dB of
    gain difference relative to the signal, so 10 dB of IF headroom at 0 dB gain is
    overrun by the 30 dB gain capture at 2 GHz on port 0 (10 + 2/3 * (0 - 30) < 0)
    but not by the 1 GHz capture at equal gain, nor on port 1"""
    extra = {'if_headroom': (10.0, 10.0)}
    assert logs._if_overload_message(extra, CAPTURE, SWEEP) == IF_OVERLOAD


IF_SILENT = {
    'ample_headroom': ({'if_headroom': (21.0, 21.0)}, CAPTURE, SWEEP),
    'no_headroom_key': ({'adc_headroom': (-1.0, 3.0)}, CAPTURE, SWEEP),
    'not_a_soapy_capture': ({'if_headroom': (-1.0, -1.0)}, TONE, TONE_SWEEP),
}


@pytest.mark.parametrize('case', list(IF_SILENT), ids=list(IF_SILENT))
def test_if_overload_is_none(case):
    extra, capture, sweep = IF_SILENT[case]
    assert logs._if_overload_message(extra, capture, sweep) is None


# %% log_info


def delayed(extra_data, capture=CAPTURE, sweep=SWEEP):
    config = compute.EvaluationOptions(
        sweep_spec=sweep, registry=sa.registry, as_xarray='delayed', extra_attrs={}
    )
    return compute.DelayedDataset(
        delayed={},
        capture=capture,
        extra_coords=ss.specs.AcquisitionInfo(),
        extra_data=extra_data,
        config=config,
    )


def test_log_info_joins_both_overload_messages_in_one_warning(caplog):
    dd = delayed({'adc_headroom': (-1.0, 3.0), 'if_headroom': (10.0, 10.0)})
    with caplog.at_level(logging.WARNING, logger='striqt.analysis'):
        logs.log_info(dd)
    assert [r.getMessage() for r in caplog.records] == [
        f'{ADC_OVERLOAD}, {IF_OVERLOAD}'
    ]
    assert caplog.records[0].levelno == logging.WARNING


def test_log_info_is_silent_without_overloads(caplog):
    dd = delayed({'adc_headroom': (1.0, 3.0), 'if_headroom': (21.0, 21.0)})
    with caplog.at_level(logging.DEBUG, logger='striqt.analysis'):
        logs.log_info(dd)
    assert caplog.records == []
