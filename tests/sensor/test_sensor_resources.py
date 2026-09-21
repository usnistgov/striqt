"""striqt.sensor.lib.resources: the open_resources lifecycle — working directory,
sink selection through extensions.sink, cleanup after a sink failure, the
calibration and log file resources, and the source-opened callback"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest
import yaml
from conftest import FAKE_SOAPY_SPEC, assert_source_released
from synthetic_sources import NO_SINK, tone_sweep

import striqt.analysis as sa
import striqt.sensor as ss

OFFSETS = (1e6, 2e6)


# %% working directory


def test_cwd_is_unchanged_without_a_spec_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with ss.open_resources(tone_sweep(OFFSETS), None):
        assert Path.cwd().resolve() == tmp_path.resolve()


def test_cwd_follows_the_spec_directory(tmp_path, monkeypatch):
    # open_resources does not change back; monkeypatch restores at teardown
    monkeypatch.chdir(os.getcwd())
    spec_dir = tmp_path / 'sub'
    spec_dir.mkdir()
    with ss.open_resources(tone_sweep(OFFSETS), spec_dir / 'sweep.yaml'):
        assert Path.cwd().resolve() == spec_dir.resolve()


# %% test_only


def test_test_only_omits_the_peripherals():
    """current behaviour, not a contract: the peripherals are not opened"""
    with ss.open_resources(tone_sweep(OFFSETS), None, test_only=True) as res:
        assert 'peripherals' not in res


@pytest.mark.xfail(
    strict=True,
    raises=KeyError,
    reason="open_resources(test_only=True) omits the 'peripherals' resource that "
    '_acquire_both indexes unconditionally (resources.py:146-151, execute.py:235)',
)
def test_test_only_resources_run_a_sweep():
    sweep = tone_sweep(OFFSETS)
    with ss.open_resources(sweep, None, test_only=True) as res:
        results = list(ss.iterate_sweep(res, sink=ss.sinks.NoSink(sweep)))
    assert len([r for r in results if r is not None]) == len(OFFSETS)


# %% sink selection and failure


def test_extensions_sink_dotted_path_selects_the_class():
    assert NO_SINK.sink == 'striqt.sensor.sinks.NoSink'
    with ss.open_resources(tone_sweep(OFFSETS), None) as res:
        assert type(res['sink']) is ss.sinks.NoSink


class FailingInitSink(ss.sinks.NoSink):
    def __init__(self, *args, **kws):
        raise RuntimeError('simulated sink failure')


class FailingOpenSink(ss.sinks.NoSink):
    def open(self):
        raise RuntimeError('simulated sink failure')


@pytest.mark.parametrize(
    'cls', [FailingInitSink, FailingOpenSink], ids=['init', 'open']
)
def test_sink_failure_surfaces_and_closes_the_source(cls, isolated_lookup):
    extension = ss.specs.Extension(sink=f'{__name__}.{cls.__name__}')
    sweep = tone_sweep(OFFSETS).replace(extensions=extension)

    with (
        pytest.raises(RuntimeError, match='simulated sink failure'),
        ss.open_resources(sweep, None),
    ):
        pass

    assert_source_released(isolated_lookup, sweep)


# %% calibration


def test_calibration_is_loaded_from_the_source_spec(fake_soapy_ext, calibration_nc):
    import xarray as xr

    spec = ss.read_yaml_spec(FAKE_SOAPY_SPEC)
    spec = spec.replace(
        source=spec.source.replace(calibration=calibration_nc), extensions=NO_SINK
    )
    with ss.open_resources(spec, None) as res:
        assert res['calibration'].equals(xr.load_dataset(calibration_nc))


# %% log file


def _run_with_log(tmp_path) -> str:
    log_path = tmp_path / 'logs' / 'sweep.log'
    sweep = tone_sweep(OFFSETS)
    sweep = sweep.replace(sink=sweep.sink.replace(log_path=str(log_path)))
    with ss.open_resources(sweep, None) as res:
        sa.util.get_logger('sweep').info('marker record')
        sa.util.get_logger('sweep').debug('below the log level')
        list(ss.iterate_sweep(res))
    # the file is terminated when the handler closes; restore_logging_state
    # would otherwise close it only at teardown
    logging.getLogger('striqt')._striqt_handler.close()
    return log_path.read_text()


def test_log_path_records_a_yaml_array_at_the_log_level(tmp_path):
    records = yaml.safe_load(_run_with_log(tmp_path))
    assert isinstance(records, list)
    by_message = {r['message']: r for r in records}
    assert by_message['marker record']['level'] == 'INFO'
    assert 'below the log level' not in by_message


@pytest.mark.xfail(
    strict=True,
    raises=json.JSONDecodeError,
    reason="_RotatingJSONFileHandler.emit appends ',\\n' after every record, so the "
    'array that close() terminates ends in a trailing comma (sensor/lib/util.py:466-473)',
)
def test_log_file_is_json(tmp_path):
    records = json.loads(_run_with_log(tmp_path))
    assert 'marker record' in {r['message'] for r in records}


# %% on_source_opened


def test_on_source_opened_is_called_once_with_the_spec_and_id():
    calls = []
    sweep = tone_sweep(OFFSETS)
    with ss.open_resources(sweep, None, on_source_opened=lambda *a: calls.append(a)):
        source_id = ss.lib.controller.lookup.id(sweep.source)
    assert calls == [(sweep, source_id)]
