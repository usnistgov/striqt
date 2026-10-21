from __future__ import annotations as __

import dataclasses
import sys
from typing import cast, Any
from pathlib import Path

from .. import specs as specs
from . import compute, io
from .controller import Controller
from .typing import TypeVar, SP, PS, PC
import striqt.analysis as sa
from striqt.analysis.lib.util import np, pd, xr


SS = TypeVar('SS', bound='specs.SoapySource')


def compute_y_factor_corrections(
    dataset: 'xr.Dataset', Tref: float = 290.0
) -> 'xr.Dataset':
    return _y_factor_power_corrections(dataset, Tref=Tref)


def summarize_calibration(corrections: 'xr.Dataset', **sel) -> 'pd.DataFrame':
    summaries = {
        'NF (dB)': _summarize_calibration_field(corrections, 'noise_figure', **sel),
        'Power Corr (dB)': _summarize_calibration_field(
            corrections, 'power_correction', **sel
        ),
    }

    return pd.DataFrame(summaries, index=summaries['NF (dB)'].index)


@sa.util.lru_cache()
def lookup_power_correction(
    cal_data: 'str | Path | None',
    capture: specs.SoapyCapture,
    master_clock_rate: float,
    format_path: specs.paths.PathFormatter | None = None,
    *,
    xp=None,
):
    if cal_data is None:
        return None
    elif isinstance(cal_data, (str, Path)):
        corrections = io.read_calibration(cal_data, format_path)
    else:
        raise TypeError('invalid cal_data input type')

    if corrections is None:
        return None

    return _lookup_calibration_var(
        corrections.power_correction,
        capture=capture,
        master_clock_rate=master_clock_rate,
        xp=xp or np,
    )


@sa.util.lru_cache()
def lookup_system_noise_power(
    cal_data: 'Path | str | None',
    capture: specs.SoapyCapture,
    master_clock_rate: float,
    format_path: specs.paths.PathFormatter | None = None,
    *,
    T=290.0,
    B=1.0,
    xp=None,
):
    """return the power spectral density of the receive system noise"""
    from scipy.constants import Boltzmann

    if xp is None:
        xp = np

    if cal_data is None:
        return None
    elif isinstance(cal_data, (str, Path)):
        corrections = io.read_calibration(cal_data, format_path)
    else:
        raise TypeError('invalid cal_data input type')

    if corrections is None:
        return None

    noise_figure = _lookup_calibration_var(
        corrections.noise_figure,
        capture=capture,
        master_clock_rate=master_clock_rate,
        xp=xp,
    )

    k = Boltzmann * 1000  # W/K -> mW/K
    noise_psd = noise_figure + 10 * xp.log10(k * T * B)

    if B == 1:
        units = 'dBm/Hz'
    else:
        units = f'dBm/{round(B)} Hz'
    return xr.DataArray(
        data=noise_psd,
        dims=compute.CAPTURE_DIM,
        attrs={'name': 'Sensor system noise spectral density', 'units': units},
    )


def bind_manual_yfactor_calibration(
    name: str, ctrl_cls: 'type[Controller[SS, SP, Any, PS, PC]]'
) -> 'type[Controller[SS, SP, Any, PS, PC]]':
    """bind a manual Y-factor calibration of the sensor bound as `ctrl_cls`.

    The result is `bind_sensor` with `ctrl_cls`'s own `Sensor` and a `Schema` whose
    `action` is `specs.ManualYFactorCalibration`; `ManualYFactorRunner` runs it.
    Sweep files select it with ``sensor_binding: <name>``. The controller class
    records the caller's module.
    """
    from . import bindings

    schema = dataclasses.replace(ctrl_cls.schema, action=specs.ManualYFactorCalibration)

    return bindings.bind_sensor(
        name,
        ctrl_cls.sensor,
        cast(specs.Schema[SS, SP, Any, PS, PC], schema),
        module=sys._getframe(1).f_globals.get('__name__'),
    )


def _limit_nyquist_bandwidth(data: 'xr.DataArray') -> 'xr.DataArray':
    """replace float('inf') analysis bandwidth with the Nyquist bandwidth"""

    # return bandwidth with same shape as dataset.channel_power_time_series
    bw = data.analysis_bandwidth.broadcast_like(data).copy().squeeze()
    sample_rate = data.backend_sample_rate.broadcast_like(data).squeeze()
    where = ~np.isfinite(bw.values)
    bw.values[where] = sample_rate.values[where]
    return bw


def _y_factor_power_corrections(dataset: 'xr.Dataset', Tref=290.0) -> 'xr.Dataset':
    from scipy.constants import Boltzmann

    k = Boltzmann * 1000  # W/K -> mW/K
    enr_dB = dataset.enr.sel(noise_diode_enabled=True, drop=True)
    enr = 10 ** (enr_dB / 10.0)

    pvt = dataset.channel_power_time_series.drop_vars('capture_index')
    power = (
        pvt
        .sel(power_detector='rms', drop=True)
        .pipe(sa.dBtopow)
        .mean(dim='time_elapsed')
    )
    power.name = 'RMS power'

    peak = pvt.sel(power_detector='peak', drop=True).max('time_elapsed')
    avg = pvt.sel(power_detector='rms', drop=True).pipe(sa.dBlinmean, 'time_elapsed')
    papr = (peak - avg).assign_attrs(units='dB')

    Pon = power.sel(noise_diode_enabled=True, drop=True)
    Poff = power.sel(noise_diode_enabled=False, drop=True)
    Y = Pon / Poff

    noise_figure = enr_dB - 10 * np.log10(Y - 1)
    noise_figure.name = 'Noise figure'
    noise_figure.attrs = {'units': 'dB'}

    Te = Tref * (10 ** (noise_figure / 10) - 1)
    Te.name = 'Effective noise temperature'
    Te.attrs = {'units': 'K'}

    B = _limit_nyquist_bandwidth(Te)

    power_correction = (k * (enr * Tref) * B) / (Pon - Poff)
    power_correction.name = 'Power scaling correction'
    power_correction.attrs = {'units': 'mW/fs'}

    return xr.Dataset({
        'temperature': Te,
        'noise_figure': noise_figure,
        'power_correction': power_correction,
        'p_on': Pon,
        'p_off': Poff,
        'papr_on': papr.sel(noise_diode_enabled=True, drop=True),
        'papr_off': papr.sel(noise_diode_enabled=False, drop=True),
    })


def _summarize_calibration_field(
    corrections: 'xr.Dataset', field_name, **sel
) -> 'pd.Series':
    max_gain = float(corrections.gain.max())
    corr = corrections[field_name].sel(gain=max_gain, **sel, drop=True).squeeze()
    stacked = corr.stack(condition=corr.dims).dropna('condition')
    return stacked.to_dataframe()[field_name]


def _describe_missing_data(cal_data: 'xr.DataArray', exact_matches: dict):
    misses = []
    cal = cal_data.copy()

    invalid_matches = dict(exact_matches)
    # remove the valid matches
    for field, value in exact_matches.items():
        try:
            cal = cal.sel({field: value}, drop=True)
        except KeyError:
            pass
        else:
            del invalid_matches[field]

    # describe the remaining mismatches
    for field, value in invalid_matches.items():
        try:
            cal.sel({field: value})
        except KeyError:
            available = ', '.join([str(v) for v in cal[field].values])
            misses += [f'{repr(value)} in {repr(field)} (available: {available})']
    return '; '.join(misses)


def _lookup_calibration_var(
    cal_var: 'xr.DataArray',
    capture: specs.SoapyCapture,
    master_clock_rate: float,
    *,
    xp,
):
    from . import compute

    results = []

    for c in specs.captures.split_capture_ports(capture):
        assert not isinstance(c.center_frequency, tuple)

        fs = compute.design_resampler(c, master_clock_rate)['fs_sdr']
        port_key = _get_port_variable(cal_var)

        # these capture fields must match the calibration conditions exactly
        exact_matches: dict[str, Any] = {
            port_key: c.port,
            'gain': c.gain,
            'lo_shift': c.lo_shift,
            'backend_sample_rate': fs,
            'analysis_bandwidth': c.analysis_bandwidth or np.inf,
        }

        try:
            # there is still one more dim to drop
            sel = cal_var.sel(**exact_matches, drop=True)
        except KeyError:
            misses = _describe_missing_data(cal_var, exact_matches)
            exc = KeyError(f'calibration is not available for this capture: {misses}')
            raise exc
        else:
            exc = None

        if compute.PORT_DIM in sel.coords:
            sel = sel.dropna(compute.PORT_DIM).squeeze()

        if exc is not None:
            raise exc

        for name in ('duration', 'source_id', 'delay'):
            if name in sel.coords:
                sel = sel.drop_vars(name)

        sel = sel.squeeze(drop=True).dropna('center_frequency')

        if sel.size == 0:
            raise ValueError(
                'no calibration data is available for this combination of sampling parameters'
            )
        elif c.center_frequency > sel.center_frequency.max():
            raise ValueError(
                f'center_frequency {c.center_frequency / 1e6} MHz exceeds calibration max {sel.center_frequency.max() / 1e6} MHz'
            )
        elif c.center_frequency < sel.center_frequency.min():
            raise ValueError(
                f'center_frequency {c.center_frequency / 1e6} MHz is below calibration min {sel.center_frequency.min() / 1e6} MHz'
            )

        # allow interpolation between sample points in these fields
        try:
            sel = sel.sel(center_frequency=c.center_frequency)
        except BaseException:
            fc = [c.center_frequency]
            sel = sel.interp(center_frequency=fc).squeeze('center_frequency')

        results.append(float(sel))

    return xp.asarray(results, dtype='float32')


def _get_port_variable(ds: 'xr.DataArray|xr.Dataset') -> str:
    """return the name of the port coordinate variable, 'port' or 'channel'.

    Older calibration files name the coordinate 'channel'.
    """
    if ds is None:
        return compute.PORT_DIM
    if compute.PORT_DIM in ds.coords:
        return compute.PORT_DIM
    else:
        # compatibility
        return 'channel'
