from __future__ import annotations as __

import typing
from pathlib import Path

from .. import specs as _specs
from . import sources as _sources
from . import compute as _compute
from . import io as _io
from . import peripherals as _peripherals
from . import sinks as _sinks
from . import util as _util
import msgspec as _msgspec

if typing.TYPE_CHECKING:
    import numpy as _np
    import pandas as _pd
    import xarray as _xr
    from . import bindings as _bindings
else:
    _np = _util.lazy_import('numpy')
    _pd = _util.lazy_import('pandas')
    _xr = _util.lazy_import('xarray')


_TC = typing.TypeVar('_TC', bound=_specs.SoapyCapture)
_TP = typing.TypeVar('_TP', bound=_specs.Peripherals)
_TS = typing.TypeVar('_TS', bound=_specs.SoapySource)
_PS = typing.ParamSpec('_PS')
_PC = typing.ParamSpec('_PC')


def _y_factor_temperature(
    power: '_xr.DataArray', enr_dB: float, Tamb: float, Tref=290.0
) -> '_xr.DataArray':
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


def _limit_nyquist_bandwidth(data: '_xr.DataArray') -> '_xr.DataArray':
    """replace float('inf') analysis bandwidth with the Nyquist bandwidth"""

    # return bandwidth with same shape as dataset.channel_power_time_series
    bw = data.analysis_bandwidth.broadcast_like(data).copy().squeeze()
    sample_rate = data.backend_sample_rate.broadcast_like(data).squeeze()
    where = bw.values == float('inf')
    bw.values[where] = sample_rate.values[where]
    return bw


def _y_factor_power_corrections(dataset: '_xr.Dataset', Tref=290.0) -> '_xr.Dataset':
    # TODO: check that this works for _xr.DataArray inputs in (enr_dB, Tamb)

    from scipy.constants import Boltzmann

    k = Boltzmann * 1000  # W/K -> mW/K
    enr_dB = dataset.enr_dB.sel(noise_diode_enabled=True, drop=True)
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

    noise_figure = enr_dB - 10 * _np.log10(Y - 1)
    noise_figure.name = 'Noise figure'
    noise_figure.attrs = {'units': 'dB'}

    T = Tref * (10 ** (noise_figure / 10) - 1)
    T.name = 'Noise temperature'
    T.attrs = {'units': 'K'}

    B = _limit_nyquist_bandwidth(T)

    power_correction = (k * (T + enr * Tref) * B) / Pon
    power_correction.name = 'Input power scaling correction'
    power_correction.attrs = {'units': 'mW/fs'}

    return _xr.Dataset(
        {
            'temperature': T,
            'noise_figure': noise_figure,
            'power_correction': power_correction,
        },
    )


def _y_factor_frequency_response_correction(
    dataset: '_xr.DataArray',
    fc_temperatures: '_xr.DataArray',
    enr_dB: float,
    Tamb: float,
    Tref=290,
):
    spectrum = dataset.power_spectral_density.sel(
        time_statistic='mean', drop=True
    ).pipe(lambda x: 10 ** (x / 10.0))

    all_T = _y_factor_temperature(spectrum, enr_dB=20.87, Tamb=294.5389)

    # normalize the power correction at each center frequency, and then average the result across center frequency

    temp_norm = fc_temperatures.broadcast_like(all_T) / all_T
    frequency_response = temp_norm.median(dim='center_frequency')

    frequency_response.name = 'Baseband power scaling correction'
    frequency_response.attrs = {'units': 'unitless'}

    return frequency_response


def _summarize_calibration_field(
    corrections: '_xr.Dataset', field_name, **sel
) -> '_pd.DataFrame':
    max_gain = float(corrections.gain.max())
    corr = corrections[field_name].sel(gain=max_gain, **sel, drop=True).squeeze()
    stacked = corr.stack(condition=corr.dims).dropna('condition')
    return stacked.to_dataframe()[[field_name]]


def _describe_missing_data(cal_data: '_xr.DataArray', exact_matches: dict):
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

    # now note the remainder
    for field, value in invalid_matches.items():
        try:
            cal.sel({field: value})
        except KeyError:
            available = ', '.join([str(v) for v in cal[field].values])
            misses += [f'{repr(value)} in {repr(field)} (available: {available})']
    return '; '.join(misses)


def _lookup_calibration_var(
    cal_var: '_xr.DataArray',
    capture: _specs.SoapyCapture,
    master_clock_rate: float,
    *,
    xp,
):
    results = []

    for capture_chan in _specs.helpers.split_capture_ports(capture):
        fs = _sources._base.design_capture_resampler(master_clock_rate, capture_chan)[
            'fs_sdr'
        ]
        port_key = _get_port_variable(cal_var)

        # these capture fields must match the calibration conditions exactly
        exact_matches = {
            port_key: capture_chan.port,
            'gain': capture_chan.gain,
            'lo_shift': capture_chan.lo_shift,
            'backend_sample_rate': fs,
            'analysis_bandwidth': capture_chan.analysis_bandwidth or _np.inf,
        }

        try:
            sel = (
                cal_var.sel(
                    **exact_matches, drop=True
                )  # there is still one more dim to drop
            )
        except KeyError:
            misses = _describe_missing_data(cal_var, exact_matches)
            exc = KeyError(f'calibration is not available for this capture: {misses}')
            raise exc
        else:
            exc = None

        if _compute.PORT_DIM in sel.coords:
            sel = sel.dropna(_compute.PORT_DIM).squeeze()

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
        elif capture_chan.center_frequency > sel.center_frequency.max():
            raise ValueError(
                f'center_frequency {capture_chan.center_frequency / 1e6} MHz exceeds calibration max {sel.center_frequency.max() / 1e6} MHz'
            )
        elif capture_chan.center_frequency < sel.center_frequency.min():
            raise ValueError(
                f'center_frequency {capture_chan.center_frequency / 1e6} MHz is below calibration min {sel.center_frequency.min() / 1e6} MHz'
            )

        # allow interpolation between sample points in these fields
        try:
            sel = sel.sel(center_frequency=capture_chan.center_frequency)
        except BaseException:
            fc = [capture_chan.center_frequency]
            sel = sel.interp(center_frequency=fc).squeeze('center_frequency')

        results.append(float(sel))

    return xp.asarray(results, dtype='float32')


def _get_port_variable(ds: '_xr.DataArray|_xr.Dataset') -> str:
    """return the appropriate name of the port coordinate variable.

    This is for backward-compatibility with prior versions that used 'channel'
    instead of 'port' nomenclature.
    """
    if ds is None:
        return _compute.PORT_DIM
    if _compute.PORT_DIM in ds.coords:
        return _compute.PORT_DIM
    else:
        # compatibility
        return 'channel'


def compute_y_factor_corrections(dataset: '_xr.Dataset', Tref=290.0) -> '_xr.Dataset':
    ret = _y_factor_power_corrections(dataset, Tref=Tref)
    # ret['baseband_frequency_response'] = _y_factor_frequency_response_correction(
    #     **kwargs, fc_temperatures=ret.temperature
    # )
    return ret


def summarize_calibration(corrections: '_xr.Dataset', **sel) -> '_pd.DataFrame':
    nf_summary = _summarize_calibration_field(corrections, 'noise_figure', **sel)
    corr_summary = _summarize_calibration_field(corrections, 'power_correction', **sel)

    return _pd.concat([nf_summary, corr_summary], axis=1)


def _ensure_loop_at_position(sweep: _specs.Sweep):
    loop_fields = [l.field for l in sweep.loops]

    try:
        idx = loop_fields.index('noise_diode_enabled')
    except ValueError:
        if len(loop_fields) > 0 and loop_fields[0] == 'port':
            idx = 1
        else:
            idx = 0

        toggle_diode = _specs.List(field='noise_diode_enabled', values=(False, True))
        loops = list(sweep.loops)
        loops.insert(idx, toggle_diode)

        _msgspec.structs.force_setattr(sweep, 'loops', tuple(loops))
    else:
        if idx == 0:
            pass
        elif idx == 1 and loop_fields[0] == 'port':
            pass
        else:
            raise TypeError('noise_diode_enabled must be the first specified loop')


@_util.lru_cache()
def lookup_power_correction(
    cal_data: 'str | Path | None',
    capture: _specs.SoapyCapture,
    master_clock_rate: float,
    alias_func: _specs.helpers.PathAliasFormatter | None = None,
    *,
    xp=None,
):
    if cal_data is None:
        return None
    elif isinstance(cal_data, (str, Path)):
        corrections = _io.read_calibration(cal_data, alias_func)
    else:
        raise TypeError('invalid cal_data input type')

    return _lookup_calibration_var(
        corrections.power_correction,
        capture=capture,
        master_clock_rate=master_clock_rate,
        xp=xp or _np,
    )


@_util.lru_cache()
def lookup_system_noise_power(
    cal_data: 'Path | str | None',
    capture: _specs.SoapyCapture,
    master_clock_rate: float,
    alias_func: _specs.helpers.PathAliasFormatter | None = None,
    *,
    T=290.0,
    B=1.0,
    xp=None,
):
    """return the calibrated system noise power, in dBm/Hz"""
    from scipy.constants import Boltzmann

    if xp is None:
        xp = _np

    if cal_data is None:
        return None
    elif isinstance(cal_data, (str, Path)):
        corrections = _io.read_calibration(cal_data, alias_func)
    else:
        raise TypeError('invalid cal_data input type')

    noise_figure = _lookup_calibration_var(
        corrections.noise_figure,
        capture=capture,
        master_clock_rate=master_clock_rate,
        xp=xp,
    )

    k = Boltzmann * 1000  # W/K -> mW/K
    noise_psd = (10 ** (noise_figure / 10) - 1) * k * T * B

    return _xr.DataArray(
        data=10 * xp.log10(noise_psd),
        dims=_compute.CAPTURE_DIM,
        attrs={'name': 'Sensor system noise PSD', 'units': 'dBm/Hz'},
    )


class YFactorSink(_sinks.SinkBase):
    _DROP_FIELDS = (
        'sweep_start_time',
        'start_time',
        'delay',
        'host_resample',
        'duration',
        'sample_rate',
    )

    def _get_path(self) -> Path | None:
        path = self._spec.path

        if self._alias_func is not None:
            path = self._alias_func(path)

        if path is not None:
            return Path(path)
        else:
            return None

    def open(self) -> None:
        path = self._get_path()

        if path is not None and not self.force and Path(path).exists():
            print('reading results from previous file')
            self.prev_corrections = _io.read_calibration(path)
        else:
            self.prev_corrections = None

        self.sweep_start_time = None

    def close(self, *exc_info):
        # pointedly, do not flush on close - only after a complete
        # dataset
        pass

    def append(self, capture_result):
        ret = super().append(capture_result)

        assert isinstance(capture_result.extra_coords, _specs.SoapyAcquisitionInfo)

        sweep_start_time = capture_result.extra_coords.sweep_start_time

        if self.sweep_start_time is None and sweep_start_time is not None:
            self.sweep_start_time = float(sweep_start_time)

        return ret

    def flush(self):
        data = self.pop()

        if len(data) == 0:
            return

        super().flush()

        # re-index by radio setting rather than capture
        port = int(data[0].port)

        fields = list(data[0].attrs['loops'])
        if 'sample_rate' in fields:
            i = fields.index('sample_rate')
            fields[i] = 'backend_sample_rate'

        attrs = {
            'sweep_start_time': self.sweep_start_time,
            'calibration_fields': fields,
        }
        capture_data = (
            _xr.concat(data, _compute.CAPTURE_DIM)
            .assign_attrs(attrs)
            .drop_vars(self._DROP_FIELDS)
            .set_xindex(fields)
        )

        # break out each remaining capture coordinate into its own dimension
        by_field = capture_data.unstack(_compute.CAPTURE_DIM)
        by_field['noise_diode_enabled'] = by_field.noise_diode_enabled.astype('bool')

        # compute and merge corrections
        corrections = compute_y_factor_corrections(by_field)

        path = self._get_path()

        if self.prev_corrections is None or path is None or self.force:
            pass
        elif Path(path).exists():
            prev_port_key = _get_port_variable(self.prev_corrections)

            print('merging results from previous file')
            if port in self.prev_corrections[prev_port_key]:
                self.prev_corrections = self.prev_corrections.drop_sel(
                    {prev_port_key: port}
                )

            corrections = _xr.concat(
                [corrections, self.prev_corrections], dim=_compute.PORT_DIM
            )

        print(f'calibration results on port {port} (shown for max gain)')
        summary = summarize_calibration(corrections, port=port)
        with _pd.option_context('display.max_rows', None):
            print(summary.sort_index(axis=1).sort_index(axis=0))

        _io.save_calibration(path, corrections)
        print(f'saved to {str(path)!r}')


def bind_manual_yfactor_calibration(
    name: str, sensor: '_bindings.SensorBinding[_TS, _TP, _TC, _PS, _PC]'
) -> '_bindings.SensorBinding[_TS, typing.Any, typing.Any, _PS, typing.Any]':
    """extend an existing binding with a y-factor calibration"""

    from . import bindings

    class capture_spec_cls(sensor.schema.capture, frozen=True, kw_only=True):
        noise_diode_enabled: _specs.types.NoiseDiodeEnabled = False

    class sweep_spec_cls(_specs.CalibrationSweep, frozen=True, kw_only=True):
        calibration: _specs.ManualYFactorPeripheral | None = None
        options = _specs.SweepOptions(
            reuse_iq=True, loop_only_nyquist=True, skip_warmup=True
        )

        def __post_init__(self):
            _ensure_loop_at_position(self)
            super().__post_init__()

    class peripherals_cls(_peripherals.CalibrationPeripheralsBase):
        _last_state = (None, None)

        def open(self):
            sensor.peripherals.open(self)  # type: ignore

        def close(self):
            sensor.peripherals.close(self)  # type: ignore

        def arm(self, capture):
            state = (capture.port, capture.noise_diode_enabled)

            if state != self._last_state:
                if capture.noise_diode_enabled:
                    input(f'enable noise diode at port {capture.port} and press enter')
                else:
                    input(f'disable noise diode at port {capture.port} and press enter')

            self._last_state = state

            sensor.peripherals.arm(capture)  # type: ignore

        def acquire(self, capture):
            assert self.calibration_spec is not None

            sensor_result = sensor.peripherals.acquire(self)  # type: ignore
            return sensor_result | self.calibration_spec.to_dict()

        def setup(
            self,
            captures: typing.Sequence[_TC],
            loops: typing.Sequence[_specs.LoopSpec],
        ):
            sensor.peripherals.setup(self, captures, loops)  # type: ignore

    return bindings.bind_sensor(
        name,
        bindings.Sensor(
            source=sensor.source,
            peripherals=peripherals_cls,
            sweep_spec=sweep_spec_cls,
            sink=YFactorSink,
        ),
        bindings.Schema(
            source=sensor.schema.source,
            capture=capture_spec_cls,
            peripherals=sensor.schema.peripherals,
            init_like=sensor.schema.init_like,
            arm_like=sensor.schema.arm_like,
        ),
    )
