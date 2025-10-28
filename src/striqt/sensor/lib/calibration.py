from __future__ import annotations

import itertools
from pathlib import Path
import typing

import msgspec

from . import datasets, peripherals, sinks, sources, specs, util
from .captures import split_capture_ports
from .specs import Annotated, meta

if typing.TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import scipy
    import xarray as xr
    import labbench as lb
else:
    np = util.lazy_import('numpy')
    pd = util.lazy_import('pandas')
    scipy = util.lazy_import('scipy')
    xr = util.lazy_import('xarray')
    lb = util.lazy_import('labbench')

NoiseDiodeEnabledType = Annotated[bool, meta(standard_name='Noise diode enabled')]


class ManualYFactorCapture(specs.RadioCapture, forbid_unknown_fields=True, frozen=True):
    """Specialize fields to add to the RadioCapture type"""

    # RadioCapture with added fields
    noise_diode_enabled: NoiseDiodeEnabledType = False


class ManualYFactorSetup(specs.SpecBase, forbid_unknown_fields=True, frozen=True):
    enr: Annotated[float, meta(standard_name='Excess noise ratio', units='dB')] = 20.87
    ambient_temperature: Annotated[
        float, meta(standard_name='Ambient temperature', units='K')
    ] = 294.5389


class CalibrationRadioSetup(specs.RadioSetup, forbid_unknown_fields=True, frozen=True):
    reuse_iq = True


class CalibrationVariables(
    specs.SpecBase, forbid_unknown_fields=True, kw_only=True, frozen=True
):
    noise_diode_enabled: tuple[NoiseDiodeEnabledType, ...] = (False, True)
    sample_rate: tuple[specs.BackendSampleRateType, ...]
    center_frequency: tuple[specs.CenterFrequencyType, ...] = (3700e6,)
    port: tuple[specs.PortType, ...] = (0,)
    gain: tuple[specs.GainType, ...] = (0,)

    # filtering and resampling
    analysis_bandwidth: tuple[specs.AnalysisBandwidthType, ...] = (float('inf'),)
    lo_shift: tuple[specs.LOShiftType, ...] = ('none',)


class ManualYFactorSweep(
    specs.Sweep, forbid_unknown_fields=True, kw_only=True, frozen=True
):
    """This specialized sweep is fed to the YAML file loader
    to specify the change in expected capture structure."""

    calibration_variables: CalibrationVariables
    defaults: ManualYFactorCapture = ManualYFactorCapture()
    calibration_setup: ManualYFactorSetup
    radio_setup: CalibrationRadioSetup

    def __post_init__(self):
        if self.radio_setup.calibration is not None:
            raise ValueError(
                'radio_setup.calibration must be None for a calibration sweep'
            )

    # the top here is just to set the annotation for msgspec
    captures: tuple[ManualYFactorCapture, ...] = tuple()

    def get_captures(self):
        variables = self.calibration_variables.validate()
        defaults = self.defaults.validate()
        return _cached_calibration_captures(variables, defaults)


@util.lru_cache()
def read_calibration(path):
    if path is None:
        return None

    return xr.open_dataset(path)


def save_calibration(path, corrections: 'xr.Dataset'):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    corrections.to_netcdf(path)


@util.lru_cache()
def _cached_calibration_captures(
    variables: CalibrationVariables, defaults: ManualYFactorCapture
):
    variables = variables.todict()

    # enforce ordering to place difficult-to-change variables in
    # the outermost loops, in case the ordering is changed by
    # subclasses
    analysis_bandwidths = variables.pop('analysis_bandwidth')
    variables = {
        datasets.PORT_DIM: variables[datasets.PORT_DIM],
        'noise_diode_enabled': variables['noise_diode_enabled'],
        **variables,
        'analysis_bandwidth': analysis_bandwidths,
    }

    # every combination of each variable
    combos = itertools.product(*variables.values())

    captures = []
    for values in combos:
        mapping = dict(zip(variables, values))
        if mapping['analysis_bandwidth'] == float('inf'):
            pass
        elif mapping['analysis_bandwidth'] > mapping['sample_rate']:
            # skip cases outside of 1st Nyquist zone
            continue
        capture = defaults.replace(**mapping)
        captures.append(capture)

    return tuple(captures)


def _y_factor_temperature(
    power: 'xr.DataArray', enr_dB: float, Tamb: float, Tref=290.0
) -> 'xr.Dataset':
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
    where = bw.values == float('inf')
    bw.values[where] = sample_rate.values[where]
    return bw


def _y_factor_power_corrections(dataset: 'xr.Dataset', Tref=290.0) -> 'xr.Dataset':
    # TODO: check that this works for xr.DataArray inputs in (enr_dB, Tamb)

    k = scipy.constants.Boltzmann * 1000  # scaled from W/K to mW/K
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


def compute_y_factor_corrections(dataset: 'xr.Dataset', Tref=290.0) -> 'xr.Dataset':
    ret = _y_factor_power_corrections(dataset, Tref=Tref)
    # ret['baseband_frequency_response'] = _y_factor_frequency_response_correction(
    #     **kwargs, fc_temperatures=ret.temperature
    # )
    return ret


def _summarize_calibration_field(
    corrections: 'xr.Dataset', field_name, **sel
) -> 'pd.DataFrame':
    max_gain = float(corrections.gain.max())
    corr = corrections[field_name].sel(gain=max_gain, **sel, drop=True).squeeze()
    stacked = corr.stack(condition=corr.dims).dropna('condition')
    return stacked.to_dataframe()[[field_name]]


def summarize_calibration(corrections: 'xr.Dataset', **sel) -> 'pd.DataFrame':
    nf_summary = _summarize_calibration_field(corrections, 'noise_figure', **sel)
    corr_summary = _summarize_calibration_field(corrections, 'power_correction', **sel)

    return pd.concat([nf_summary, corr_summary], axis=1)


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
    capture: specs.RadioCapture,
    base_clock_rate: float,
    *,
    xp,
):
    results = []

    for capture_chan in split_capture_ports(capture):
        fs = sources.design_capture_resampler(base_clock_rate, capture_chan)['fs_sdr']
        port_key = _get_port_variable(cal_var)

        # these capture fields must match the calibration conditions exactly
        exact_matches = {
            port_key: capture_chan.port,
            'gain': capture_chan.gain,
            'lo_shift': capture_chan.lo_shift,
            'backend_sample_rate': fs,
            'analysis_bandwidth': capture_chan.analysis_bandwidth or np.inf,
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

        if datasets.PORT_DIM in sel.coords:
            sel = sel.dropna(datasets.PORT_DIM).squeeze()

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
        try:
            sel = sel.sel(center_frequency=capture_chan.center_frequency)
        except BaseException:
            fc = [capture_chan.center_frequency]
            sel = sel.interp(center_frequency=fc).squeeze('center_frequency')

        results.append(float(sel))

    return xp.asarray(results, dtype='float32')


@util.lru_cache()
def lookup_power_correction(
    cal_data: Path | 'xr.Dataset' | None,
    capture: specs.RadioCapture,
    base_clock_rate: float,
    *,
    xp=np,
):
    if isinstance(cal_data, xr.Dataset):
        corrections = cal_data
    elif cal_data:
        corrections = read_calibration(cal_data)
    else:
        return None

    return _lookup_calibration_var(
        corrections.power_correction,
        capture=capture,
        base_clock_rate=base_clock_rate,
        xp=xp,
    )


@util.lru_cache()
def lookup_system_noise_power(
    cal_data: Path | 'xr.Dataset' | None,
    capture: specs.RadioCapture,
    base_clock_rate: float,
    *,
    T=290.0,
    B=1.0,
    xp=np,
):
    """return the calibrated system noise power, in dBm/Hz"""
    if isinstance(cal_data, xr.Dataset):
        corrections = cal_data
    elif cal_data:
        corrections = read_calibration(cal_data)
    else:
        return None

    noise_figure = _lookup_calibration_var(
        corrections.noise_figure,
        capture=capture,
        base_clock_rate=base_clock_rate,
        xp=xp,
    )

    k = scipy.constants.Boltzmann * 1000  # scaled from W/K to mW/K
    noise_psd = (10 ** (noise_figure / 10) - 1) * k * T * B

    return xr.DataArray(
        data=10 * xp.log10(noise_psd),
        dims=datasets.CAPTURE_DIM,
        attrs={'name': 'Sensor system noise PSD', 'units': 'dBm/Hz'},
    )


def _get_port_variable(ds: 'xr.Dataset') -> str:
    """return the appropriate name of the port coordinate variable.

    This is for backward-compatibility with prior versions that used 'channel'
    instead of 'port' nomenclature.
    """
    if ds is None:
        return datasets.PORT_DIM
    if datasets.PORT_DIM in ds.coords:
        return datasets.PORT_DIM
    else:
        # compatibility
        return 'channel'


class YFactorSink(sinks.SinkBase):
    sweep_spec: ManualYFactorSweep

    _DROP_FIELDS = (
        'sweep_start_time',
        'start_time',
        'delay',
        'host_resample',
        'duration',
        'sample_rate',
    )

    def open(self):
        if not self.force and Path(self.output_path).exists():
            print('reading results from previous file')
            self.prev_corrections = read_calibration(self.output_path)
        else:
            self.prev_corrections = None

        self.sweep_start_time = None

    def close(self, *exc_info):
        # pointedly, do not flush on close - only after a complete
        # dataset
        pass

    def append(self, capture_data: 'xr.Dataset', capture: ManualYFactorCapture):
        if capture_data is None:
            return

        super().append(capture_data, capture)

        if self.sweep_start_time is None:
            self.sweep_start_time = float(capture_data.sweep_start_time[0])

    def flush(self):
        data = self.pop()

        if len(data) == 0:
            return

        super().flush()

        # re-index by radio setting rather than capture
        port = int(data[0].port)

        fields = list(self.sweep_spec.calibration_variables.__struct_fields__)
        if 'sample_rate' in fields:
            fields.remove('sample_rate')
            fields.append('backend_sample_rate')

        attrs = {
            'sweep_start_time': self.sweep_start_time,
            'calibration_fields': fields,
        }
        capture_data = (
            xr.concat(data, datasets.CAPTURE_DIM)
            .assign_attrs(attrs)
            .drop_vars(self._DROP_FIELDS)
            .set_xindex(fields)
        )

        # break out each remaining capture coordinate into its own dimension
        by_field = capture_data.unstack(datasets.CAPTURE_DIM)
        by_field['noise_diode_enabled'] = by_field.noise_diode_enabled.astype('bool')

        # compute and merge corrections
        corrections = compute_y_factor_corrections(by_field)

        if not self.force and Path(self.output_path).exists():
            prev_port_key = _get_port_variable(self.prev_corrections)

            print('merging results from previous file')
            if port in self.prev_corrections[prev_port_key]:
                self.prev_corrections = self.prev_corrections.drop_sel(
                    {prev_port_key: port}
                )

            corrections = xr.concat(
                [corrections, self.prev_corrections], dim=datasets.PORT_DIM
            )

        print(f'calibration results on port {port} (shown for max gain)')
        summary = summarize_calibration(corrections, port=port)
        with pd.option_context('display.max_rows', None):
            print(summary.sort_index(axis=1).sort_index(axis=0))

        save_calibration(self.output_path, corrections)
        print(f'saved to {str(self.output_path)!r}')


class ManualYFactorPeripherals(peripherals.PeripheralsBase):
    """Human input "peripheral" that prompts noise diode connection changes"""

    sweep: ManualYFactorSweep

    _last_state = (None, None)

    def arm(self, capture: ManualYFactorCapture):
        """This is run before each capture"""
        state = (capture.port, capture.noise_diode_enabled)

        if state != self._last_state:
            if capture.noise_diode_enabled:
                input(f'enable noise diode at port {capture.port} and press enter')
            else:
                input(f'disable noise diode at port {capture.port} and press enter')

        self._last_state = state

        return capture

    def acquire(self, capture: ManualYFactorCapture):
        """This runs during each capture.

        It should return a dictionary of results keyed by name
        with (float, int, str, xr.DataArray, etc)
        """

        return {
            'enr_dB': self.sweep.calibration_setup.enr,
            'Tamb_K': self.sweep.calibration_setup.ambient_temperature,
        }


def _field_tuple(info: msgspec.structs.FieldInfo) -> tuple:
    return (
        info.name,
        info.type,
        msgspec.field(
            name=info.name, default=info.default, default_factory=info.default_factory
        ),
    )


def _merge_specs(
    name: str, module: str, cls_list: list[type[specs.SpecBase]]
) -> type[specs.SpecBase]:
    # for each field name, the definition that will be returned is the
    # the last defined in cls_list
    all_fields = sum((msgspec.structs.fields(c) for c in cls_list), tuple())
    fields = {info.name: _field_tuple(info) for info in all_fields}

    return msgspec.defstruct(name, fields.values(), bases=(cls_list[0],), module=module)


def specialize_cal_sweep(
    name: str,
    cal_cls: type[specs.Sweep],
    sensor_cls: type[specs.Sweep],
) -> type[specs.Sweep]:
    """build a type calibration sweep specification struct for the given sensor.

    The idea is to generate a `cal_cls` subclass that can be applied
    to a `sensor_cls` that has been extended with additional fields.

    As such, the `captures` and `radio_setup` fields in the returned class
    have been updated with fields and defaults from `sensor_cls`.
    """
    all_fields = {}
    module = sensor_cls.__module__

    for cls in (cal_cls, sensor_cls):
        for info in msgspec.structs.fields(cls):
            all_fields.setdefault(info.name, []).append(info.type)

    capture_cls = _merge_specs(
        'CalibrationRadioCapture', module, all_fields['defaults']
    )
    setup_cls = _merge_specs('CalibrationRadioSetup', module, all_fields['radio_setup'])

    sweep_cls = msgspec.defstruct(
        name,
        fields=(
            ('captures', tuple[capture_cls, ...], tuple()),
            ('defaults', capture_cls, capture_cls()),
            ('radio_setup', setup_cls, setup_cls()),
        ),
        bases=(cal_cls,),
        module=sensor_cls.__module__,
        forbid_unknown_fields=True,
        frozen=True,
        cache_hash=True,
    )

    return sweep_cls


def specialize_cal_peripherals(
    name: str,
    cal_cls: type[peripherals.PeripheralsBase],
    sensor_cls: type[peripherals.PeripheralsBase],
) -> type[peripherals.PeripheralsBase]:
    """extend a Peripherals class to perform calibrations.

    The idea is to generate a `cal_cls` subclass that can be applied
    to a custom `sensor_cls` peripheral class that has been extended with
    additional fields.
    """

    class peripheral_cls(sensor_cls, cal_cls):
        """A collection of peripheral devices (switches, thermometers, etc.) used in sensing.

        For remote control, devices connections should be accessible to the client PC.
        The `edge-sweep.py` script expects this object to have `setup` and `acquire` methods.
        """

        def open(self):
            lb.concurrently(
                integrations=lb.Call(sensor_cls.open, self),
                calibration=lb.Call(cal_cls.open, self),
            )

        def close(self):
            lb.concurrently(
                integrations=lb.Call(sensor_cls.close, self),
                calibration=lb.Call(cal_cls.close, self),
            )

        def arm(self, capture: specs.RadioCapture):
            """runs before each capture"""

            lb.concurrently(
                integrations=lb.Call(sensor_cls.arm, self, capture),
                calibration=lb.Call(cal_cls.arm, self, capture),
                flatten=True,
            )

        def acquire(self, capture) -> dict:
            """runs during each capture, and returns a dictionary of results keyed by name"""
            sensor_result = sensor_cls.acquire(self, capture) or {}
            cal_result = cal_cls.acquire(self, capture) or {}
            return sensor_result | cal_result

        def setup(self):
            sensor_cls.setup(self)
            cal_cls.setup(self)

    peripheral_cls.__name__ = name
    peripheral_cls.__module__ = sensor_cls.__module__
    peripheral_cls.__qualname__ = name

    return peripheral_cls
