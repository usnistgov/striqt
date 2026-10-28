"""strategies and factories for the sweep and capture spec tests."""

# ruff: noqa: UP045  msgspec resolves the QuickAnalyses annotations at runtime on py3.9
from __future__ import annotations

import dataclasses
import functools
from typing import Optional

import msgspec
import numpy as np
import xarray as xr
from hypothesis import strategies as st
from soapy_factories import MCR

import striqt.analysis as sa
import striqt.sensor as ss

SOURCE = ss.specs.FunctionSource(master_clock_rate=MCR, num_rx_ports=2)
CaptureCls = ss.bindings.single_tone.schema.capture
SweepCls = ss.bindings.single_tone.sensor.sweep_spec_cls

LO_SHIFTS = ('left', 'right', 'none')
LIST_LOOP_FIELDS = ('frequency_offset', 'snr', 'lo_shift')
SAMPLE_RATES = (1e6, 2e6, 4e6)


def make_capture_kws(**kws) -> dict:
    return {'port': 0, 'sample_rate': 1e6, 'duration': 1e-3, **kws}


def make_capture(**kws):
    return CaptureCls(**make_capture_kws(**kws))


def make_sweep_kws(captures=(), loops=(), adjust_captures=None, **kws) -> dict:
    return {
        'source': SOURCE,
        'captures': tuple(captures),
        'loops': tuple(loops),
        'adjust_captures': {} if adjust_captures is None else adjust_captures,
        **kws,
    }


def make_sweep(captures=(), loops=(), adjust_captures=None, **kws):
    return SweepCls(**make_sweep_kws(captures, loops, adjust_captures, **kws))


class QuickAnalyses(
    sa.specs.AnalysisGroup, frozen=True, kw_only=True, omit_defaults=True
):
    """an analysis group over 2 of the built-in measurements"""

    spectrogram: Optional[sa.specs.Spectrogram] = None
    channel_power_time_series: Optional[sa.specs.ChannelPowerTimeSeries] = None


@functools.lru_cache
def quick_binding(action=ss.specs.Sweep) -> type[ss.lib.controller.Controller]:
    """return the `single_tone` sensor bound for `action` over `QuickAnalyses`,
    unregistered"""
    schema = dataclasses.replace(
        ss.bindings.single_tone.schema, action=action, analysis=QuickAnalyses
    )
    return ss.bindings.bind_sensor(
        f'quick_{action.__name__}',
        ss.bindings.single_tone.sensor,
        schema,
        register=False,
        module=__name__,
    )


class ActionCls(
    ss.specs.ActionSpec[
        ss.specs.FunctionSource, ss.specs.Peripherals, ss.specs.BuiltinAnalyses
    ],
    frozen=True,
    kw_only=True,
):
    """an action spec over the type of `SOURCE` with no captures block"""


class PlanCls(
    ss.specs.CapturePlan[
        ss.specs.FunctionSource,
        ss.specs.Peripherals,
        CaptureCls,
        ss.specs.BuiltinAnalyses,
    ],
    frozen=True,
    kw_only=True,
):
    """a capture plan over the single_tone capture class with no analysis or options"""


def make_plan(captures=(), loops=(), adjust_captures=None, **kws):
    return PlanCls(**make_sweep_kws(captures, loops, adjust_captures, **kws))


# 1e4 Hz divides the 1e6 sample_rate of make_capture into 100 bins; 3e4 does not
SPG = ss.specs.BundledAnalysis.from_dict({
    'spectrogram': {'window': 'hann', 'frequency_resolution': 1e4}
})
RESOLUTION_MSG = 'sample_rate/resolution must be a counting number'


port_scalars = st.integers(min_value=0, max_value=3)
port_tuples = st.lists(port_scalars, min_size=1, max_size=3, unique=True).map(tuple)
port_values = st.one_of(port_scalars, port_tuples)
lo_frequencies = st.floats(min_value=1e6, max_value=1e10)


@st.composite
def ports_and_lo(draw):
    """draw a port tuple and an `external_lo_frequency` value: a tuple of the same
    length, a scalar, or None"""
    port = draw(port_tuples)
    per_port = st.tuples(*[lo_frequencies] * len(port))
    lo = draw(st.one_of(st.none(), lo_frequencies, per_port))
    return port, lo


@st.composite
def capture_kwargs(draw, ports=port_values):
    sample_rate = draw(st.sampled_from(SAMPLE_RATES))
    count = draw(st.integers(min_value=10, max_value=1000))
    return {
        'port': draw(ports),
        'sample_rate': sample_rate,
        # an integer sample count keeps Capture.__post_init__ satisfied
        'duration': count / sample_rate,
        # finite, since Sweep construction rejects an lo_shift loop over an infinite
        # bandwidth; 0.5 MHz clears the FIR transition band at every SAMPLE_RATES entry
        'analysis_bandwidth': 0.5e6,
        'frequency_offset': draw(st.floats(min_value=0, max_value=1e5)),
        'snr': draw(st.one_of(st.none(), st.floats(min_value=0, max_value=60))),
    }


def capture_tuples(min_size: int = 1, max_size: int = 3):
    return st.lists(capture_kwargs(), min_size=min_size, max_size=max_size).map(
        lambda kws: tuple(make_capture(**k) for k in kws)
    )


def _unique_floats(min_size: int = 0, max_size: int = 4):
    return st.lists(
        st.floats(min_value=0, max_value=1e5),
        min_size=min_size,
        max_size=max_size,
        unique=True,
    )


@st.composite
def list_loop(draw, field: str, min_size: int = 0):
    if field == 'lo_shift':
        lo_shifts = st.sampled_from(LO_SHIFTS)
        values = draw(st.lists(lo_shifts, min_size=min_size, max_size=3, unique=True))
    else:
        values = draw(_unique_floats(min_size))
    return ss.specs.List(field=field, values=tuple(values))


@st.composite
def range_loop(draw, field: str):
    start = draw(st.floats(min_value=0, max_value=1e4))
    step = draw(st.floats(min_value=1.0, max_value=1e3))
    count = draw(st.integers(min_value=1, max_value=4))
    stop = start + (count - 1) * step
    return ss.specs.Range(field=field, start=start, stop=stop, step=step)


@st.composite
def frequency_bin_range_loop(draw, field: str):
    step = draw(st.floats(min_value=1.0, max_value=1e3))
    half = draw(st.integers(min_value=0, max_value=2))
    return ss.specs.FrequencyBinRange(
        field=field, start=-half * step, stop=half * step, step=step
    )


@st.composite
def loop_sets(draw, allow_repeat: bool = True, allow_analysis: bool = True):
    fields = draw(st.lists(st.sampled_from(LIST_LOOP_FIELDS), max_size=3, unique=True))
    loops = []
    for field in fields:
        if field == 'frequency_offset':
            offset_loop = st.one_of(
                list_loop(field), range_loop(field), frequency_bin_range_loop(field)
            )
            loop = draw(offset_loop)
        else:
            loop = draw(list_loop(field))
        loops.append(loop)
    if allow_analysis and draw(st.booleans()):
        loops.append(
            ss.specs.List(field='window', isin='analysis', values=('hann', 'hamming'))
        )
    loops = draw(st.permutations(loops))
    if allow_repeat and draw(st.booleans()):
        # a Repeat is only valid as the outermost loop
        count = draw(st.integers(min_value=1, max_value=3))
        loops.insert(0, ss.specs.Repeat(count=count))
    return tuple(loops)


@st.composite
def sweeps(draw, min_captures: int = 1):
    captures = draw(capture_tuples(min_size=min_captures))
    loops = draw(loop_sets())

    # a loop point overwrites its field in every capture, so Sweep rejects captures
    # that disagree on a looped field; capture_kwargs draws those fields freely
    looped = {l.field for l in loops if l.isin == 'capture' and l.field is not None}
    shared = {f: getattr(captures[0], f) for f in looped} if captures else {}
    if shared:
        captures = tuple(c.replace(**shared) for c in captures)

    return make_sweep(captures=captures, loops=loops)


CAL = ss.bindings.air7101b_calibration
CalSweepCls = CAL.sensor.sweep_spec_cls
CalCaptureCls = CAL.schema.capture
CalSourceCls = CAL.schema.source

PORT_LOOP = ss.specs.List(field='port', values=(0, 1))
GAIN_LOOP = ss.specs.List(field='gain', values=(0, -10))
CAL_LOOP_VALUES = {'port': (0, 1), 'gain': (0, -10), 'center_frequency': (1e9, 2e9)}


def make_calibration_capture(**kws):
    kws = {
        'port': 0,
        'center_frequency': 1e9,
        'gain': 0,
        'duration': 1e-3,
        'sample_rate': 1e6,
        **kws,
    }
    return CalCaptureCls(**kws)


def make_calibration_sweep_kws(
    captures=None, loops=(), source=None, calibration=None, **kws
) -> dict:
    if captures is None:
        captures = (make_calibration_capture(),)
    if source is None:
        source = CalSourceCls()
    if calibration is None:
        calibration = ss.specs.ManualYFactorSpec(
            enr=YFACTOR_ENR_DB, ambient_temperature=290.0
        )
    return {
        'source': source,
        'captures': captures,
        'loops': loops,
        'calibration': calibration,
        **kws,
    }


def make_calibration_sweep(**kws):
    return CalSweepCls(**make_calibration_sweep_kws(**kws))


def as_sweep(spec):
    """return the `Sweep` that runs for `spec`: `spec` itself, or the sweep built by
    the runner of `spec`"""
    if isinstance(spec, ss.specs.Sweep):
        return spec
    return ss.lib.actions.registry.lookup(spec)(spec).sweep


def make_calibration_runner(**kws):
    """return the `ManualYFactorRunner` of ``make_calibration_sweep(**kws)``"""
    return ss.lib.actions.ManualYFactorRunner(make_calibration_sweep(**kws))


def loop_fields(sweep) -> list:
    return [loop.field for loop in sweep.loops]


# %% y-factor calibration model
#
# A receiver with power gain G after a noise source at temperature Tsource, with
# effective input noise temperature Te, has rms output power in full-scale units
#   P = G * k * (Tsource + Te) * B
# so the Y-factor method recovers noise_figure == nf_dB and
# power_correction == 1/G exactly.

BOLTZMANN_MW = 1.380649e-23 * 1e3
T_REF = 290.0
YFACTOR_ENR_DB = 20.87
YFACTOR_NF_DB = {0: 5.0, 1: 8.0}
YFACTOR_GRID = {
    'port': (0, 1),
    'backend_sample_rate': (125e6,),
    'center_frequency': (1e9, 2e9),
    'gain': (0.0, -10.0),
    'analysis_bandwidth': (40e6,),
    'lo_shift': ('none',),
}


def receiver_gain(gain_dB, front_end_gain_dB=50.0, fs_mW=1.0):
    """return the power gain from the receiver input (mW) to full-scale units"""
    return 10 ** ((front_end_gain_dB + gain_dB) / 10) / fs_mW


def noise_figure_lookup(nf_dB, port, center_frequency) -> float:
    """return the noise figure (dB) of `port` at `center_frequency` from `nf_dB`, keyed
    ``{port: value}`` or ``{port: {center_frequency: value}}``"""
    value = nf_dB[port]
    if isinstance(value, dict):
        return value[center_frequency]
    return value


def yfactor_rms_power(
    gain_dB, nf_dB, bandwidth, *, diode_on, enr_dB=YFACTOR_ENR_DB, T0=T_REF, **gain_kws
):
    """return the rms output power (full-scale units) of the modeled receiver in
    `bandwidth` (Hz)"""
    Te = T0 * (10 ** (nf_dB / 10) - 1)
    Tsource = T0 + np.where(diode_on, T0 * 10 ** (enr_dB / 10), 0.0)
    return (
        receiver_gain(gain_dB, **gain_kws) * BOLTZMANN_MW * (Tsource + Te) * bandwidth
    )


def _noise_figure_grid(grid, nf_dB):
    import xarray as xr

    dims = ('port', 'center_frequency')
    values = [
        [noise_figure_lookup(nf_dB, p, fc) for fc in grid['center_frequency']]
        for p in grid['port']
    ]
    return xr.DataArray(values, dims=dims, coords={d: list(grid[d]) for d in dims})


def make_yfactor_dataset(
    grid=YFACTOR_GRID,
    nf_dB=YFACTOR_NF_DB,
    *,
    enr_dB=YFACTOR_ENR_DB,
    papr_dB=10.0,
    time_count=4,
    **gain_kws,
):
    """return a calibration dataset with one dimension per key of `grid` plus
    `noise_diode_enabled`.

    The `channel_power_time_series` variable (dBfs) is constant over `time_count`
    bins at the receiver model power for `enr_dB`, with the peak detector `papr_dB`
    above the rms detector.
    """
    import xarray as xr

    coords = {'noise_diode_enabled': [False, True]}
    coords.update({k: list(v) for k, v in grid.items()})
    fields = xr.Dataset(coords=coords)
    template = xr.DataArray(
        np.zeros(tuple(fields.sizes.values())), coords=fields.coords
    )

    bw = fields.analysis_bandwidth
    bandwidth = bw.where(np.isfinite(bw), fields.backend_sample_rate)
    gain = fields.gain.broadcast_like(template)
    nf = _noise_figure_grid(grid, nf_dB).broadcast_like(template)
    bandwidth = bandwidth.broadcast_like(template)
    diode_on = fields.noise_diode_enabled.broadcast_like(template)
    rms_power = yfactor_rms_power(
        gain, nf, bandwidth, diode_on=diode_on, enr_dB=enr_dB, **gain_kws
    )
    rms_dB = 10 * np.log10(rms_power)

    pvt = xr.concat([rms_dB, rms_dB + papr_dB], dim='power_detector')
    pvt = pvt.assign_coords(power_detector=['rms', 'peak'])
    pvt = pvt.expand_dims(time_elapsed=np.arange(time_count) * 1e-3)
    pvt = pvt.transpose(*fields.dims, 'power_detector', 'time_elapsed')
    pvt.attrs = {'units': 'dBfs'}
    capture_index = template.copy(data=np.arange(template.size).reshape(template.shape))

    return xr.Dataset({
        'channel_power_time_series': pvt.assign_coords(capture_index=capture_index),
    })


def expected_yfactor(grid=YFACTOR_GRID, nf_dB=YFACTOR_NF_DB, *, T0=T_REF, **gain_kws):
    """return the closed-form `noise_figure` (dB), `temperature` (K), and
    `power_correction` of the receiver model over `grid`"""
    import xarray as xr

    nf = _noise_figure_grid(grid, nf_dB)
    gain = xr.DataArray(
        list(grid['gain']), dims='gain', coords={'gain': list(grid['gain'])}
    )
    return xr.Dataset({
        'noise_figure': nf,
        'temperature': T0 * (10 ** (nf / 10) - 1),
        'power_correction': 1 / receiver_gain(gain, **gain_kws),
    })


INF = float('inf')
CORRECTION_VARS = ('noise_figure', 'temperature', 'power_correction')
# float64 roundoff of the closed-form corrections, expressed in dB
EXACT_DB = 1e-8


def assert_corrections_match(actual, expected, tol_dB=EXACT_DB, names=CORRECTION_VARS):
    """assert that each `names` variable of `actual` matches `expected` within `tol_dB`.

    `expected` may span fewer dimensions than `actual`. `temperature` and
    `power_correction` are compared in dB.
    """
    for name in names:
        result = actual[name].reset_coords(drop=True)
        model = expected[name].reset_coords(drop=True).broadcast_like(result)
        if name != 'noise_figure':
            result, model = 10 * np.log10(result), 10 * np.log10(model)
        xr.testing.assert_allclose(result, model, rtol=0, atol=tol_dB)


def save_yfactor_calibration(
    path, grid=YFACTOR_GRID, nf_dB=YFACTOR_NF_DB, **kws
) -> str:
    """write the Y-factor corrections of ``make_yfactor_dataset(grid, nf_dB, **kws)`` to
    a netCDF file at `path`.

    Returns:
        `path` as a string
    """
    from striqt.sensor.lib import io
    from striqt.sensor.lib.actions import yfactor

    corrections = yfactor.compute_y_factor_corrections(
        make_yfactor_dataset(grid, nf_dB, **kws), kws.get('enr_dB', YFACTOR_ENR_DB)
    )
    io.save_calibration(path, corrections)
    return str(path)


# %% cellular sweeps
#
# The 3750 MHz channel of sweeps/site/site-cellular.yaml: 100 MHz at 30 kHz spacing,
# so nr_sample_rate derives 107.52 MS/s (4 x 896 x 30 kHz).

CELL_CHANNEL = ss.specs.CellChannel(
    center_frequency=3750e6,
    channel_bandwidth=100e6,
    ssb_frequency_offset=-40.08e6,
    channel_guard_bandwidths=(875e3, 575e3),
    tdd_frame_slots='dddsuudddddddsuudddd',
    tdd_special_symbols='ddddddfffffffu',
)
CELL_LABEL = '3750 MHz'


def make_cell_channel(**kws):
    """return `CELL_CHANNEL` with fields replaced by `kws`, validated"""
    return ss.specs.CellChannel(**{**CELL_CHANNEL.to_dict(), **kws})


def cellular_sweep_cls():
    """return the bound `site_cellular` sweep spec class.

    The import of `site_strategies` registers the site extension module.
    """
    from site_strategies import EXT

    return EXT.site_cellular.sensor.sweep_spec_cls


def make_cellular_capture(**kws):
    """return a site capture at the `CELL_CHANNEL` center frequency with `sample_rate`
    and `analysis_bandwidth` left at the defaults"""
    kws = {
        'port': 0,
        'center_frequency': CELL_CHANNEL.center_frequency,
        'gain': -10,
        'duration': 1e-3,
        # the antenna_index coordinate is integer, so from_delayed rejects None
        'antenna_index': 0,
        **kws,
    }
    return cellular_sweep_cls().schema.capture(**kws)


def make_cellular_sweep_kws(
    captures=None, loops=(), cell_channels=None, source=SOURCE, **kws
) -> dict:
    if captures is None:
        captures = (make_cellular_capture(),)
    if cell_channels is None:
        cell_channels = {CELL_LABEL: CELL_CHANNEL}
    return {
        'source': source,
        'captures': tuple(captures),
        'loops': tuple(loops),
        'cell_channels': cell_channels,
        'sink': ss.specs.ZarrSink(path=None),
        **kws,
    }


def make_cellular_sweep(**kws):
    return cellular_sweep_cls()(**make_cellular_sweep_kws(**kws))


def make_cellular_runner(**kws):
    """return the `Cellular5GNRSweepRunner` of ``make_cellular_sweep(**kws)``"""
    return ss.lib.actions.Cellular5GNRSweepRunner(make_cellular_sweep(**kws))


# the built-in measurements that a Cellular5GNRSweep generates, and no others
CellularAnalyses = msgspec.defstruct(
    'CellularAnalyses',
    [
        (f.name, f.type, None)
        for f in msgspec.structs.fields(sa.specs.BuiltinAnalyses)
        if f.name in ss.specs.Cellular5GNRAnalysis.__struct_fields__
    ],
    bases=(sa.specs.AnalysisGroup,),
    frozen=True,
    kw_only=True,
    omit_defaults=True,
    module=__name__,
)


def restricted_cellular_binding() -> type[ss.lib.controller.Controller]:
    """return the `site_cellular` sensor bound over `CellularAnalyses`, unregistered"""
    from site_strategies import EXT

    return _restricted_cellular_binding(EXT.site_cellular)


@functools.lru_cache
def _restricted_cellular_binding(origin):
    schema = dataclasses.replace(origin.schema, analysis=CellularAnalyses)
    return ss.bindings.bind_sensor(
        'cellular_restricted', origin.sensor, schema, register=False, module=__name__
    )
