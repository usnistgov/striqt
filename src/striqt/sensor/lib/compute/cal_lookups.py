"""lookups into saved Y-factor corrections for a measurement sweep.

The netCDF written by `actions.yfactor.save_yfactor_corrections` is indexed by the
radio settings of the calibration grid. Each lookup matches port, gain, LO shift,
backend sample rate, and analysis bandwidth exactly and interpolates along center
frequency.
"""

from __future__ import annotations as __

from pathlib import Path
from typing import Any

import striqt.analysis as sa
from striqt.analysis.lib.dataarrays import CAPTURE_DIM, PORT_DIM
from striqt.analysis.lib.util import np, pd, xr

from ... import specs
from .. import io
from . import corrections


def summarize_calibration(corrections: xr.Dataset, **sel) -> pd.DataFrame:
    summaries = {
        'NF (dB)': _summarize_calibration_field(corrections, 'noise_figure', **sel),
        'Power Corr (dB)': _summarize_calibration_field(
            corrections, 'power_correction', **sel
        ),
    }

    return pd.DataFrame(summaries, index=summaries['NF (dB)'].index)


@sa.util.lru_cache()
def lookup_power_correction(
    cal_data: str | Path | None,
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
    cal_data: Path | str | None,
    capture: specs.SoapyCapture,
    master_clock_rate: float,
    format_path: specs.paths.PathFormatter | None = None,
    *,
    T=290.0,
    B=1.0,
    xp=None,
):
    """return the system noise power (dBm) referred to the calibrated input, per port.

    The power is ``k * T * B * F``, the noise power that the sensor measures with the
    input terminated at temperature `T`. `F` is the linear noise figure from the
    calibration file at the port, gain, LO shift, backend sample rate, and analysis
    bandwidth of `capture`, interpolated in center frequency.

    Args:
        cal_data: the path of the calibration corrections file, or None for no
            calibration
        capture: the capture that selects the calibration row, one row per port
        master_clock_rate: the SDR master clock rate (S/s), which sets the backend
            sample rate of the row
        format_path: the formatter of the ``{field}`` templates in `cal_data`, or None
        T: the source temperature (K) that the noise figure is referred to
        B: the noise bandwidth (Hz); 1 Hz gives a power spectral density
        xp: the array namespace of the result, or None for numpy

    Returns:
        a `DataArray` on the capture dimension, one value per port, in dBm/Hz when
        `B` is 1 and in dBm with a `noise_bandwidth` attr otherwise; the
        `reference_temperature` attr is `T`. None when `cal_data` is None or the
        file has no corrections.

    Raises:
        TypeError: if `cal_data` is neither a path nor None
        KeyError: if the port, gain, LO shift, backend sample rate, or analysis
            bandwidth of a port is not in the calibration grid
        ValueError: if the center frequency of a port is outside the calibrated
            range, or no calibrated row remains for the port
    """
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

    attrs: dict[str, Any] = {
        'standard_name': 'System noise',
        'name': 'Sensor system noise spectral density',
        'reference_temperature': float(T),
    }
    if B == 1:
        attrs['units'] = 'dBm/Hz'
        attrs['long_name'] = (
            f'Sensor noise power spectral density referred to a {T:g} K source'
        )
    else:
        attrs['units'] = 'dBm'
        attrs['noise_bandwidth'] = float(B)
        attrs['long_name'] = (
            f'Sensor noise power referred to a {T:g} K source '
            f'(dBm/{sa.dataarrays.format_units(B, "Hz")})'
        )
    return xr.DataArray(data=noise_psd, dims=CAPTURE_DIM, attrs=attrs)


def _summarize_calibration_field(
    corrections: xr.Dataset, field_name, **sel
) -> pd.Series:
    max_gain = float(corrections.gain.max())
    corr = corrections[field_name].sel(gain=max_gain, **sel, drop=True).squeeze()
    stacked = corr.stack(condition=corr.dims).dropna('condition')
    return stacked.to_dataframe()[field_name]


def _describe_missing_data(cal_data: xr.DataArray, exact_matches: dict):
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

    for field, value in invalid_matches.items():
        try:
            cal.sel({field: value})
        except KeyError:
            available = ', '.join([str(v) for v in cal[field].values])
            misses += [f'{value!r} in {field!r} (available: {available})']
    return '; '.join(misses)


def _lookup_calibration_var(
    cal_var: xr.DataArray,
    capture: specs.SoapyCapture,
    master_clock_rate: float,
    *,
    xp,
):
    results = []

    for c in specs.captures.split_capture_ports(capture):
        assert not isinstance(c.center_frequency, tuple)

        fs = corrections.design_resampler(c, master_clock_rate)['fs_sdr']
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

        if PORT_DIM in sel.coords:
            sel = sel.dropna(PORT_DIM).squeeze()

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


def _get_port_variable(ds: xr.DataArray | xr.Dataset) -> str:
    """return the port coordinate name of `ds`.

    ``'port'`` is returned when present, otherwise ``'channel'``.
    """
    if ds is None:
        return PORT_DIM
    if PORT_DIM in ds.coords:
        return PORT_DIM
    else:
        # compatibility
        return 'channel'
