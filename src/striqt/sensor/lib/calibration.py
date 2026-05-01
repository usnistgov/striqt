from __future__ import annotations as __

from typing import Any, Sequence, TYPE_CHECKING
from pathlib import Path

from .. import specs as specs
from . import compute, io, peripherals, sinks, sources, util
from .typing import Peripherals, TypeVar, SC, SP, SPC, PS, PC
import striqt.analysis as sa

import msgspec

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import xarray as xr
    from . import bindings
else:
    np = sa.util.lazy_import('numpy')
    pd = sa.util.lazy_import('pandas')
    xr = sa.util.lazy_import('xarray')


SS = TypeVar('SS', bound='specs.SoapySource')


def compute_y_factor_corrections(dataset: 'xr.Dataset', Tref=290.0) -> 'xr.Dataset':
    ret = _y_factor_power_corrections(dataset, Tref=Tref)
    # ret['baseband_frequency_response'] = _y_factor_frequency_response_correction(
    #     **kwargs, fc_temperatures=ret.temperature
    # )
    return ret


def summarize_calibration(corrections: 'xr.Dataset', **sel) -> 'pd.DataFrame':
    nf_summary = _summarize_calibration_field(corrections, 'noise_figure', **sel)
    corr_summary = _summarize_calibration_field(corrections, 'power_correction', **sel)

    return pd.concat([nf_summary, corr_summary], axis=1)


@sa.util.lru_cache()
def lookup_power_correction(
    cal_data: 'str | Path | None',
    capture: specs.SoapyCapture,
    master_clock_rate: float,
    alias_func: specs.helpers.PathAliasFormatter | None = None,
    *,
    xp=None,
):
    if cal_data is None:
        return None
    elif isinstance(cal_data, (str, Path)):
        corrections = io.read_calibration(cal_data, alias_func)
    else:
        raise TypeError('invalid cal_data input type')

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
    alias_func: specs.helpers.PathAliasFormatter | None = None,
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
        corrections = io.read_calibration(cal_data, alias_func)
    else:
        raise TypeError('invalid cal_data input type')

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


def set_iq_calibration(iq: sources.AcquiredIQ):
    assert isinstance(iq.capture, specs.SoapyCapture)

    xp = sources.buffers.get_array_namespace(iq.source_spec.array_backend)
    kwargs = {
        'cal_data': iq.source_spec.calibration,
        'capture': iq.capture,
        'master_clock_rate': iq.source_spec.master_clock_rate,
        'alias_func': iq.alias_func,
    }

    # calibration data
    power_scale = lookup_power_correction(**kwargs, xp=xp)
    if power_scale is not None:
        voltage_scale = sources.buffers.get_dtype_scale(iq.source_spec.transport_dtype)
        iq.voltage_scale = voltage_scale * (power_scale**0.5)
        iq.extra_data['system_noise'] = lookup_system_noise_power(**kwargs)


class YFactorSink(sinks.SinkBase):
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
        self.sweep_start_time = None

    def close(self, *exc_info):
        # pointedly, do not flush on close - only after a complete
        # dataset
        pass

    def append(self, capture_result):
        ret = super().append(capture_result)

        assert isinstance(capture_result.extra_coords, specs.SoapyAcquisitionInfo)

        if len(self._pending_data) == self._batch.size:
            self.flush()
            self._batch.next()

        return ret

    def flush(self):
        data = self.pop()

        if len(data) == 0:
            return

        super().flush()

        # re-index by radio setting rather than capture
        port = int(data[0].port)

        loops = data[0].attrs['loops']
        fields = [l['field'] for l in loops if l['field'] is not None]
        if 'sample_rate' in fields:
            i = fields.index('sample_rate')
            fields[i] = 'backend_sample_rate'

        attrs = {
            'sweep_start_time': self.sweep_start_time,
            'calibration_fields': fields,
        }
        capture_data = (
            xr.concat(data, compute.CAPTURE_DIM)
            .assign_attrs(attrs)
            .drop_vars(self._DROP_FIELDS, errors='ignore')
            .set_xindex(fields)
        )

        # break out each remaining capture coordinate into its own dimension
        by_field = capture_data.unstack(compute.CAPTURE_DIM)
        by_field['noise_diode_enabled'] = by_field.noise_diode_enabled.astype('bool')

        # compute and merge corrections
        corrections = compute_y_factor_corrections(by_field)

        path = self._get_path()

        for port, _ in corrections.groupby('port', squeeze=False):
            print(f'calibration results on port {port} (shown for max gain)')
            summary = summarize_calibration(corrections, port=port)
            with pd.option_context('display.max_rows', None):
                print(summary.sort_index(axis=1).sort_index(axis=0))

        io.save_calibration(path, corrections)
        print(f'saved to {str(path)!r}')


class ManualYFactorPeripheral(
    peripherals.CalibrationPeripheralsBase[Any, Any, specs.ManualYFactorPeripheral]
):
    _last_state = (None, None)

    def open(self):
        assert self.calibration_spec is not None

        enr = self.calibration_spec.enr
        prompt = f'Confirm that the noise diode ENR is {enr} dB (y/n): '
        while True:
            response = sa.util.blocking_input(prompt)
            if response.lower() == 'y':
                break
            elif response.lower() == 'n':
                raise RuntimeError('Set the proper noise diode ENR and run again')

    def close(self):
        pass

    def arm(self, capture):
        state = (capture.port, capture.noise_diode_enabled)
        if state != self._last_state:
            what = f'noise diode at port {capture.port}'
            if capture.noise_diode_enabled:
                sa.util.blocking_input(f'enable {what} and press enter: ')
            else:
                sa.util.blocking_input(f'disable {what} and press enter: ')

        self._last_state = state

    def acquire(self, capture):
        assert self.calibration_spec is not None
        return self.calibration_spec.to_dict()

    def setup(
        self,
        captures: Sequence[SC],
        loops: Sequence[specs.LoopSpec],
    ):
        pass


def _calibration_peripherals_cls(
    ext: type[Peripherals[SP, SC]],
    cal: type[peripherals.CalibrationPeripheralsBase[Any, Any, SPC]],
) -> type[peripherals.CalibrationPeripheralsBase[SP, SC, SPC]]:
    """return a peripheral type for external hardware and calibration"""

    class cls(peripherals.CalibrationPeripheralsBase):
        _last_state = (None, None)

        def __init__(self, spec):
            self.ext = ext(spec)
            self.cal = cal(spec)

        def open(self):
            futures = [
                util.threadpool.submit(self.ext.open),
                util.threadpool.submit(self.cal.open),
            ]
            util.await_and_ignore(futures)

        def setup(
            self,
            captures: Sequence[SC],
            loops: Sequence[specs.LoopSpec],
        ):
            self.ext.setup(captures, loops)
            self.cal.setup(captures, loops)

        def close(self):
            try:
                self.ext.close()
            finally:
                self.cal.close()

        def arm(self, capture):
            futs = [
                util.threadpool.submit(self.ext.arm, capture),
                util.threadpool.submit(self.cal.arm, capture),
            ]
            util.await_and_ignore(futs)

        def acquire(self, capture):
            ext_fut = util.threadpool.submit(self.ext.acquire, capture)
            cal_fut = util.threadpool.submit(self.cal.acquire, capture)

            with util.ExceptionStack() as exc:
                with exc.defer():
                    ext = ext_fut.result()
                with exc.defer():
                    cal = cal_fut.result()

            return ext | cal

    return cls


def bind_manual_yfactor_calibration(
    name: str, sensor: 'bindings.SensorBinding[SS, SP, SC, PS, PC]'
) -> 'bindings.SensorBinding[SS, SP, SC, PS, PC]':
    """extend an existing binding with a y-factor calibration"""

    from . import bindings

    class capture_spec_cls(sensor.schema.capture, frozen=True, kw_only=True):
        noise_diode_enabled: specs.types.NoiseDiodeEnabled = False

    class sweep_spec_cls(specs.CalibrationSweep, frozen=True, kw_only=True):
        calibration: specs.ManualYFactorPeripheral | None = None
        options = specs.SweepOptions(
            reuse_iq=True, loop_only_nyquist=True, skip_warmup=True
        )

        def __post_init__(self):
            _ensure_loop_at_position(self)
            super().__post_init__()

    peripherals_cls = _calibration_peripherals_cls(
        sensor.peripherals, ManualYFactorPeripheral
    )

    return bindings.bind_sensor(
        name,
        bindings.Sensor(
            source=sensor.source,
            peripherals=peripherals_cls,
            sweep_spec=sweep_spec_cls,
            sink=YFactorSink,
        ),  # pyright: ignore
        bindings.Schema(
            source=sensor.schema.source,
            capture=capture_spec_cls,
            peripherals=sensor.schema.peripherals,
            init_like=sensor.schema.init_like,
            arm_like=sensor.schema.arm_like,
        ),
    )


def _ensure_loop_at_position(sweep: specs.Sweep):
    loop_fields = [l.field for l in sweep.loops]

    try:
        idx = loop_fields.index('noise_diode_enabled')
    except ValueError:
        if len(loop_fields) > 0 and loop_fields[0] == 'port':
            idx = 1
        else:
            idx = 0

        toggle_diode = specs.List(field='noise_diode_enabled', values=(False, True))
        loops = list(sweep.loops)
        loops.insert(idx, toggle_diode)

        msgspec.structs.force_setattr(sweep, 'loops', tuple(loops))
    else:
        if idx == 0:
            pass
        elif idx == 1 and loop_fields[0] == 'port':
            pass
        else:
            raise TypeError('noise_diode_enabled must be the first specified loop')


def _y_factor_temperature(
    power: 'xr.DataArray', enr_dB: float, Tamb: float, Tref=290.0
) -> 'xr.DataArray':
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


def _limit_nyquist_bandwidth(data: 'xr.DataArray') -> 'xr.DataArray':
    """replace float('inf') analysis bandwidth with the Nyquist bandwidth"""

    # return bandwidth with same shape as dataset.channel_power_time_series
    bw = data.analysis_bandwidth.broadcast_like(data).copy().squeeze()
    sample_rate = data.backend_sample_rate.broadcast_like(data).squeeze()
    where = bw.data == float('inf')
    bw.data[where] = sample_rate.data[where]
    return bw


def _y_factor_power_corrections(dataset: 'xr.Dataset', Tref=290.0) -> 'xr.Dataset':
    # TODO: check that this works for xr.DataArray inputs in (enr_dB, Tamb)

    from scipy.constants import Boltzmann

    k = Boltzmann * 1000  # W/K -> mW/K
    enr_dB = dataset.enr.sel(noise_diode_enabled=True, drop=True)
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

    T = Tref * (10 ** (noise_figure / 10) - 1)
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
        },
    )


def _y_factor_frequency_response_correction(
    dataset: 'xr.DataArray',
    fc_temperatures: 'xr.DataArray',
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
    corrections: 'xr.Dataset', field_name, **sel
) -> 'pd.DataFrame':
    max_gain = float(corrections.gain.max())
    corr = corrections[field_name].sel(gain=max_gain, **sel, drop=True).squeeze()
    stacked = corr.stack(condition=corr.dims).dropna('condition')
    return stacked.to_dataframe()[[field_name]]


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

    # now note the remainder
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

    for c in specs.helpers.split_capture_ports(capture):
        assert not isinstance(c.center_frequency, tuple)

        fs = compute.design_resampler(c, master_clock_rate)['fs_sdr']
        port_key = _get_port_variable(cal_var)

        # these capture fields must match the calibration conditions exactly
        exact_matches = {
            port_key: c.port,
            'gain': c.gain,
            'lo_shift': c.lo_shift,
            'backend_sample_rate': fs,
            'analysis_bandwidth': c.analysis_bandwidth or np.inf,
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
    """return the appropriate name of the port coordinate variable.

    This is for backward-compatibility with prior versions that used 'channel'
    instead of 'port' nomenclature.
    """
    if ds is None:
        return compute.PORT_DIM
    if compute.PORT_DIM in ds.coords:
        return compute.PORT_DIM
    else:
        # compatibility
        return 'channel'
