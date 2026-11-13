"""striqt.cli.sensor_sweep: the logging callbacks of the sweep CLI

The tests assert record counts and levels, not message text.
"""

from __future__ import annotations

import logging
import types

import click
import pytest
from soapy_factories import soapy_capture
from sweep_strategies import air_sweep, make_delayed, spectrogram_result
from synthetic_sources import tone_sweep

import striqt.sensor as ss
from striqt.cli import sensor_sweep
from striqt.cli.sensor_sweep import confirm_labels, log_overloads, log_spectrogram_snr

AIR = ss.bindings.air7101b
CAPTURE = soapy_capture(port=(0, 1), center_frequency=(1e9, 2e9), gain=0.0)


def delayed(extra_data, calibration=None):
    sweep = air_sweep(CAPTURE, calibration=calibration)
    return make_delayed(extra_data, sweep=sweep, capture=CAPTURE)


# %% log_overloads


def test_log_overloads_warns_once_for_an_overload(caplog):
    with caplog.at_level(logging.DEBUG, logger='striqt.analysis'):
        log_overloads(delayed({'adc_headroom': (-1.0, 3.0)}))
    assert [r.levelno for r in caplog.records] == [logging.WARNING]


def test_log_overloads_is_silent_without_an_overload(caplog):
    with caplog.at_level(logging.DEBUG, logger='striqt.analysis'):
        log_overloads(delayed({'adc_headroom': (1.0, 3.0)}))
    assert caplog.records == []


# %% log_spectrogram_snr


def resources(calibration):
    dd = delayed({}, calibration)
    return {
        'sweep_spec': dd.config.sweep_spec,
        'source': types.SimpleNamespace(source_id='beef'),
        'format_path': None,
    }


def test_log_spectrogram_snr_logs_one_info_record_per_port(calibration_nc, caplog):
    cache = types.SimpleNamespace(name='_cached_spectrogram')
    with caplog.at_level(logging.DEBUG, logger='striqt.analysis'):
        log_spectrogram_snr(
            resources(calibration_nc),
            cache,
            CAPTURE,
            spectrogram_result(shape=(2, 4, 8)),
        )
    assert [r.levelno for r in caplog.records] == [logging.INFO, logging.INFO]


def test_log_spectrogram_snr_is_silent_for_another_cache(calibration_nc, caplog):
    cache = types.SimpleNamespace(name='ssb_iq_cache')
    with caplog.at_level(logging.DEBUG, logger='striqt.analysis'):
        log_spectrogram_snr(
            resources(calibration_nc),
            cache,
            CAPTURE,
            spectrogram_result(shape=(2, 4, 8)),
        )
    assert caplog.records == []


# %% confirm_labels


class _RecordingPrompt:
    """a `prompt` hook that appends each text to `texts` and pops `answers`"""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.texts = []

    def __call__(self, text):
        self.texts.append(text)
        return self.answers.pop(0)


ADJUSTED = tone_sweep((1e6,), adjust_captures={'defaults': {'snr': 5.0}})


def test_confirm_labels_lists_the_adjustments_until_answered():
    prompt = _RecordingPrompt('maybe', 'y')
    confirm_labels(ADJUSTED, 'beef', prompt)
    assert len(prompt.texts) == 2
    assert "'snr': (5.0,)" in prompt.texts[0]


def test_confirm_labels_rejection_interrupts():
    with pytest.raises(KeyboardInterrupt):
        confirm_labels(ADJUSTED, 'beef', _RecordingPrompt('n'))


def test_confirm_labels_is_silent_without_adjustments():
    # an empty answer list fails on the first prompt
    confirm_labels(tone_sweep((1e6,)), 'beef', _RecordingPrompt())


def test_confirm_labels_reads_stdin_without_a_hook(scripted_input):
    scripted_input['answers'] += ['y']
    confirm_labels(ADJUSTED, 'beef')
    assert len(scripted_input['prompts']) == 1


def test_run_binds_the_prompt_into_the_check_hook(monkeypatch, synthetic_spec_path):
    """`run` sets `prompt` on the callbacks of the manager and partially applies it
    to `check_func`, so that the hook keeps the ``(sweep, source_id)`` signature"""
    seen = {}

    class FakeManager:
        def set_callbacks(self, callbacks):
            seen['callbacks'] = callbacks
            raise _Stop

    monkeypatch.setattr(ss, 'open_resources', lambda spec, path: FakeManager())
    prompt = _RecordingPrompt('y')

    def check(sweep, source_id, prompt=None):
        seen['check'] = (sweep, source_id, prompt)

    with pytest.raises(_Stop):
        sensor_sweep.run(synthetic_spec_path, check_func=check, prompt=prompt)

    callbacks = seen['callbacks']
    assert callbacks.prompt is prompt
    callbacks.on_source_opened('sweep', 'beef')
    assert seen['check'] == ('sweep', 'beef', prompt)


class _Stop(Exception):
    pass


def test_run_rejects_another_suffix(tmp_path):
    with pytest.raises(click.ClickException, match=r'\.txt'):
        sensor_sweep.run(tmp_path / 'spec.txt')
