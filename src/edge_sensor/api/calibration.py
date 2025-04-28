from __future__ import annotations

import functools
from pathlib import Path
import pickle
import typing

from . import store, structs, util, xarray_ops
from .captures import split_capture_channels

if typing.TYPE_CHECKING:
    import gzip
    import numpy as np
    import pandas as pd
    import xarray as xr
    import scipy
else:
    gzip = util.lazy_import('gzip')
    np = util.lazy_import('numpy')
    xr = util.lazy_import('xarray')
    scipy = util.lazy_import('scipy')
    iqwaveform = util.lazy_import('iqwaveform')


@functools.lru_cache
def read_calibration_corrections(path):
    if path is None:
        return None

    with gzip.GzipFile(path, 'rb') as fd:
        return pickle.load(fd)


def save_calibration_corrections(path, corrections: xr.Dataset):
    with gzip.GzipFile(path, 'wb') as fd:
        pickle.dump(corrections, fd)


def _y_factor_temperature(
    power: xr.DataArray, enr_dB: float, Tamb: float, Tref=290.0
) -> xr.Dataset:
    Toff = Tamb
    Ton = Tref * 10 ** (enr_dB / 10.0)

    # compute the Y-factor from measured power
    Pon = power.sel(noise_diode_enabled=True, drop=True)
    Poff = power.sel(noise_diode_enabled=False, drop=True)
    Y = Pon / Poff

    # compute receive noise temperature from the Y-factor
    T = (Ton - Y * Toff) / (Y - 1)
    T.name = 'T'
    T.attrs = {'units': 'K'}

    return T


def _limit_nyquist_bandwidth(data: xr.DataArray) -> xr.DataArray:
    """replace float('inf') analysis bandwidth with the Nyquist bandwidth"""

    # return bandwidth with same shape as dataset.channel_power_time_series
    bw = data.analysis_bandwidth.broadcast_like(data).copy().squeeze()
    sample_rate = data.sample_rate.broadcast_like(data).squeeze()
    where = bw.values == float('inf')
    bw.values[where] = sample_rate.values[where]
    return bw


def _y_factor_power_corrections(
    dataset: xr.Dataset, enr_dB: float, Tamb: float, Tref=290.0
) -> xr.Dataset:
    # TODO: check that this works for xr.DataArray inputs in (enr_dB, Tamb)

    kwargs = dict(list(locals().items())[1:])

    k = scipy.constants.Boltzmann * 1000  # scaled from W/K to mW/K
    enr = 10 ** (enr_dB / 10.0)

    power = (
        dataset.channel_power_time_series.sel(power_detector='rms', drop=True)
        .pipe(lambda x: 10 ** (x / 10.0))
        .mean(dim='time_elapsed')
    )
    power.name = 'RMS power'

    Pon = power.sel(noise_diode_enabled=True, drop=True)
    Poff = power.sel(noise_diode_enabled=False, drop=True)
    Y = Pon / Poff

    noise_figure = enr_dB - 10 * np.log10(Y - 1)
    noise_figure.name = 'Noise figure'
    noise_figure.attrs = {'units': 'dB'}

    T = Tref * (10 ** (noise_figure / 10) - 1)  # _y_factor_temperature(power, **kwargs)
    T.name = 'Noise temperature'
    T.attrs = {'units': 'K'}

    B = _limit_nyquist_bandwidth(T)

    power_correction = (k * (T + enr * Tref) * B) / Pon
    power_correction.name = 'Input power scaling correction'
    power_correction.attrs = {'units': 'mW/fs'}

    return xr.Dataset(
        {
            'temperature': T,
            'noise_figure': noise_figure,
            'power_correction': power_correction,
        }
    )


def _y_factor_frequency_response_correction(
    dataset: 'xr.DataArray',
    fc_temperatures: 'xr.DataArray',
    enr_dB: float,
    Tamb: float,
    Tref=290,
):
    spectrum = dataset.power_spectral_density.sel(
        frequency_statistic='mean', drop=True
    ).pipe(lambda x: 10 ** (x / 10.0))

    all_T = _y_factor_temperature(spectrum, enr_dB=20.87, Tamb=294.5389)

    # normalize the power correction at each center frequency, and then average the result across center frequency

    temp_norm = fc_temperatures.broadcast_like(all_T) / all_T
    frequency_response = temp_norm.median(dim='center_frequency')

    frequency_response.name = 'Baseband power scaling correction'
    frequency_response.attrs = {'units': 'unitless'}

    return frequency_response


def compute_y_factor_corrections(
    dataset: xr.Dataset, enr_dB: float, Tamb: float, Tref=290.0
) -> xr.Dataset:
    kwargs = locals()
    ret = _y_factor_power_corrections(**kwargs)
    # ret['baseband_frequency_response'] = _y_factor_frequency_response_correction(
    #     **kwargs, fc_temperatures=ret.temperature
    # )
    return ret


def summarize_calibration(corrections: xr.Dataset, **sel):
    nf_summary = _summarize_calibration_field(corrections, 'noise_figure', **sel)
    corr_summary = _summarize_calibration_field(corrections, 'power_correction', **sel)

    return pd.concat([nf_summary, corr_summary], axis=1)


def _summarize_calibration_field(
    corrections: xr.Dataset, field_name, **sel
) -> 'pd.DataFrame':
    max_gain = float(corrections.gain.max())
    corr = corrections[field_name].sel(gain=max_gain, **sel, drop=True).squeeze()
    stacked = corr.stack(condition=corr.dims).dropna('condition')
    return stacked.to_dataframe()[[field_name]]


def _describe_missing_data(corrections: xr.Dataset, exact_matches: dict):
    misses = []
    cal = corrections.power_correction.copy()

    invalid_matches = dict(exact_matches)
    # remove the valid matches
    for field, value in exact_matches.items():
        try:
            cal = cal.sel({field: value}, drop=True)
        except KeyError:
            pass
        else:
            del invalid_matches[field]

    # now note the remainder
    for field, value in invalid_matches.items():
        try:
            cal.sel({field: value})
        except KeyError:
            misses += [
                f'{repr(value)} in {repr(field)} (available: '
                + ', '.join([str(v) for v in cal[field].values])
                + ')'
            ]
    return '; '.join(misses)


@functools.lru_cache()
def lookup_power_correction(
    cal_data: Path | xr.Dataset | None, capture: structs.RadioCapture, xp
):
    if isinstance(cal_data, xr.Dataset):
        corrections = cal_data
    elif cal_data:
        corrections = read_calibration_corrections(cal_data)
    else:
        return None

    power_scale = []

    for capture_chan in split_capture_channels(capture):
        # these fields must match the calibration conditions exactly
        exact_matches = dict(
            channel=capture_chan.channel,
            gain=capture_chan.gain,
            lo_shift=capture_chan.lo_shift,
            sample_rate=capture_chan.sample_rate,
            analysis_bandwidth=capture_chan.analysis_bandwidth or np.inf,
            host_resample=capture_chan.host_resample,
        )

        try:
            sel = corrections.power_correction.sel(**exact_matches, drop=True)
        except KeyError:
            misses = _describe_missing_data(corrections, exact_matches)
            exc = KeyError(f'calibration is not available for this capture: {misses}')
        else:
            exc = None

        if exc is not None:
            raise exc

        for name in ('duration', 'radio_id', 'delay'):
            if name in sel.coords:
                sel = sel.drop(name)

        sel = sel.squeeze(drop=True).dropna('center_frequency')

        if sel.size == 0:
            raise ValueError(
                'no calibration data is available for this combination of sampling parameters'
            )
        elif capture_chan.center_frequency > sel.center_frequency.max():
            raise ValueError(
                f'center_frequency {capture_chan.center_frequency / 1e6} MHz exceeds calibration max {sel.center_frequency.max() / 1e6} MHz'
            )
        elif capture_chan.center_frequency < sel.center_frequency.min():
            raise ValueError(
                f'center_frequency {capture_chan.center_frequency / 1e6} MHz is below calibration min {sel.center_frequency.min() / 1e6} MHz'
            )

        # allow interpolation between sample points in these fields
        sel = sel.interp(center_frequency=capture_chan.center_frequency)

        power_scale.append(float(sel))

    return xp.asarray(power_scale, dtype='float32')


class CalibrationDataManager(store.SweepDataManager):
    _DROP_FIELDS = (
        'sweep_start_time',
        'start_time',
        'delay',
        'host_resample',
        'duration',
        'sample_rate',
    )

    def append(self, capture_data: xr.Dataset):
        if capture_data is None:
            return
        else:
            self.pending_data.append(capture_data)
        return

    def flush(self):
        # re-index by radio setting rather than capture
        sweep_start_time = self.data_captures[0].sweep_start_time[0]
        channel = int(self.data_captures[0].channel)

        data_captures = (
            xr.concat(self.data_captures, xarray_ops.CAPTURE_DIM)
            .assign_attrs({'sweep_start_time': float(sweep_start_time)})
            .drop_vars(self._DROP_FIELDS)
        )

        dataset = data_captures.set_xindex(list(data_captures.capture.coords)).unstack(
            'capture'
        )
        dataset['noise_diode_enabled'] = dataset.noise_diode_enabled.astype('bool')

        if self.output_path is None:
            radio_id = data_captures.radio_id.values[0]
            driver = data_captures.attrs['driver']
            output_path = f'cals/{driver}-{radio_id}.p'
        else:
            output_path = self.output_path

        # compute and merge corrections
        from edge_sensor import iq_corrections as cal

        corrections = cal.compute_y_factor_corrections(
            dataset,
            enr_dB=self.sweep_spec.radio_setup.enr,
            Tamb=self.sweep_spec.radio_setup.ambient_temperature,
        )

        if not self.force and Path(output_path).exists():
            print('merging results from previous file')
            prev_corrections = read_calibration_corrections(output_path)
            if channel in prev_corrections.channel:
                prev_corrections = prev_corrections.drop_sel(channel=channel)
            corrections = xr.concat([corrections, prev_corrections], dim='channel')

        print(f'Channel {channel} calibration results:')
        summary = cal.summarize_calibration(corrections, channel=channel)
        print(summary.sort_index(axis=1).sort_index(axis=0))

        cal.save_calibration_corrections(output_path, corrections)
