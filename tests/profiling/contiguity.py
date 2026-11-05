#!/usr/bin/env python3
"""Profile the record-sized copies, the resampling stage, and the per-capture layers of
the signal path at the ADA shapes.

Each row times one library call as the current code makes it, on the capture and source
of the aggregate-directivity-acquisition fragments ``denver-3750`` and ``7201b``
(2 ports, 0.1 s at 107.52 MS/s from the 125 MHz master clock, a 2 ms transient
holdoff, and a 20 ms trigger strobe). Nothing under ``src/`` is modified. The same run
before and after a change to the library is the measurement of that change.

Rows of the stage table:

- ``cast_iq column slice``: `cast_iq` over the ``[:, span]`` column slice of the host
  buffer, the copy that `Controller.read_iq` makes with ``start`` at the end of the
  holdoff. ``resident`` is the size of the device buffer that the result keeps alive.
- ``oaresample, out=x``: `sw.oaresample` with the arguments of the resampling stage,
  with the input consumed in place and the output written into the buffer of the
  input, as the sensor stage runs it. The peak is the stacked STFT frames, the FFT
  work area, and the low-pass response.
- ``stft, stage block size``: the analysis half of the resampler alone, so the peak
  of the row above splits into the frames and the rest.
- ``D2H column slice``: ``.get()`` of the resampled record without the lead and tail
  pads, the array that `iq_waveform` returns for a 2-port capture. cupy copies a
  non-contiguous array into a contiguous device temporary before the transfer.
- ``get_5g_ssb_spans``: the span extraction of the cellular measurements over the
  same record, for the 30 kHz PSS at the ADA analysis offset of -40.08 MHz.
- ``correlate pss`` and ``correlate sss``: `correlate_sync_spans` at 15.36 MS/s, 30 kHz
  subcarrier spacing (SSB case C), 2 ports, and 5 synchronization blocks, for the 3
  PSS templates and for ``--cells`` of the 1008 SSS templates. The kernel of the 8 GB
  Tegra X2 kills the process at the full 1008 cells, so the default is 252; the same
  count before and after a change is the comparison.
- ``histogram ravel``: `xp.histogram` over one port of a spectrogram at the ADA
  shape.

The layer table times `channel_power_time_series` as the pipeline reaches it, on a
conditioned acquisition of the ADA capture: the resampling stage, the aligned stage
the measurement selects on top of the resampled waveform, the measurement body on
that stage, and the registry call with the stages memoized and from cold. Two flags
reproduce the conditions of the pipeline. ``--trigger`` conditions the acquisition
with the ``cellular_5g_pss_sync`` trigger of the production source; the aligned stage
then runs the trigger measurement and shifts each port to the trigger.
``--overwrite-x`` lets the resampling stage consume the raw waveform in place; each
fresh conditioning then first refills the raw waveform with ones, and the ``refill
raw.x`` row times that refill alone.

Usage:
    pixi run -e cupy python tests/profiling/contiguity.py
    pixi run -e cupy python tests/profiling/contiguity.py --cells 504 --repeat 3
    pixi run -e cupy python tests/profiling/contiguity.py --trigger --overwrite-x
    pixi run -e test39 python tests/profiling/contiguity.py --backend numpy --duration 0.02

Reading the output:
    ``host`` is the wall time of the call, and ``device`` the CUDA event span around it
    (equal to ``host`` on numpy). ``MB`` is the size of the array the call returns,
    and the last column the quantity named in the row description; ``pool peak`` is
    the high-water mark of the cupy pool during one further call above the bytes in
    use before the call. Rows are the median of ``--repeat`` calls after ``--warmup``
    calls. In the layer table, a registry call far above the body is time in the
    wrapper or the stage it selects, and a stage row names which. With
    ``--overwrite-x`` the ``resample stage`` and ``registry call, cold`` rows include
    the ``refill raw.x`` time; the ``aligned stage`` row does not.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import statistics
from collections.abc import Callable
from fractions import Fraction
from types import ModuleType
from typing import Any

import numpy as np
from profiling_tools import (
    Setup,
    build_raw,
    build_setup,
    device_synchronize,
    free_pool,
    oaresample_kws,
    peak_pool_bytes,
    print_row,
    print_table_header,
    repeat_call,
    select_backend,
    time_call,
)

import striqt.analysis as sa
import striqt.sensor as ss
import striqt.waveform as sw
from striqt.analysis.measurements import power
from striqt.sensor.lib.sources import buffers

# the ADA capture, as `profiling_tools.CAPTURE` sets it
PORTS = 2
SAMPLE_RATE = 107.52e6
ANALYSIS_BANDWIDTH = 100e6
DURATION = 0.1
SSB_FREQUENCY_OFFSET = -40.08e6

# fragments/analysis/cellular.yaml spectrogram and spectrogram_histogram
FREQUENCY_RESOLUTION = 15000
FRACTIONAL_OVERLAP = Fraction(13, 28)
INTEGRATION_BANDWIDTH = 360e3
HISTOGRAM = {'power_low': -150, 'power_high': 0, 'power_resolution': 0.1}

SYNC_SAMPLE_RATE = 15.36e6
SYNC_SCS = 30e3
SYNC_CASE = 'c'
SYNC_BLOCKS = 5
SSS_CELL_COUNT = 1008
DEFAULT_SSS_CELLS = 252


def host_empty(shape: tuple[int, ...], dtype: Any, xp: ModuleType) -> np.ndarray:
    """return an uninitialized host array, pinned when `xp` is cupy."""
    if xp is np:
        return np.empty(shape, dtype=dtype)
    from cupyx import empty_pinned  # ty: ignore[unresolved-import]

    return empty_pinned(shape, dtype=dtype)


def pool_label(peak: int, xp: ModuleType) -> str:
    """return the ``pool peak`` cell of a row, ``-`` on numpy."""
    return 'pool peak -' if xp is np else f'pool peak {peak / 2**20:.0f} MB'


# %% Stage table
def profile_cast_iq(setup: Setup, xp: ModuleType, repeat: int, warmup: int) -> None:
    capture, source = setup.capture, setup.source
    lead, tail = ss.lib.compute.get_correction_overlaps(capture, source, None)
    overlap = lead + tail
    output_count = buffers.get_read_count(capture, source, overlap=overlap)
    buffer_count = buffers.get_read_count(
        capture, source, include_holdoff=True, overlap=overlap
    )
    holdoff = buffer_count - output_count
    span = slice(holdoff, holdoff + output_count)

    buffer = host_empty((PORTS, buffer_count), np.complex64, xp)
    buffer[:] = 0
    print(
        f'receive buffer: {PORTS} ports x {buffer_count} samples, '
        f'holdoff {holdoff} samples ({holdoff / buffer_count:.1%})'
    )

    def column_slice() -> Any:
        return buffers.cast_iq(source, buffer[:, span], output_count)

    host, device, out = repeat_call(column_slice, repeat, warmup)
    base = out if out.base is None else out.base
    print_row(
        'cast_iq column slice',
        host,
        device,
        out.nbytes,
        f'resident {base.nbytes / 2**20:.1f} MB',
    )


def profile_resampler(
    raw: ss.specs.AcquiredIQ, xp: ModuleType, repeat: int, warmup: int
) -> Any:
    """print the resampler rows and return the resampled record, a view of `raw.x`."""
    design: Any = raw.resampler
    kws = oaresample_kws(raw)
    nfft = design['nfft']
    print(
        f'input: {raw.x.shape} complex64, {raw.x.nbytes / 2**20:.0f} MB; '
        f'resampler nfft {nfft} -> {design["nfft_out"]}'
    )

    def into_x() -> Any:
        raw.x[:] = 1
        return sw.oaresample(raw.x, overwrite_x=True, out=raw.x, **kws)

    def stft_only() -> Any:
        return sw.stft(
            raw.x,
            fs=design['fs_sdr'],
            window=design['window'],
            nperseg=nfft,
            noverlap=nfft // 2,
            axis=1,
            truncate=True,
        )[2]

    for name, func in [
        ('oaresample, out=x', into_x),
        ('stft, stage block size', stft_only),
    ]:
        free_pool()
        peak = peak_pool_bytes(func)
        host, device, out = repeat_call(func, repeat, warmup)
        print_row(name, host, device, out.nbytes, pool_label(peak, xp))
        del out
    free_pool()
    return into_x()


def profile_record(
    resampled: Any, capture: Any, xp: ModuleType, repeat: int, warmup: int
) -> None:
    """print the D2H and span extraction rows over the record without the pads."""
    width = resampled.shape[1]
    record = round(capture.duration * capture.sample_rate)
    pad = (width - record) // 2
    sliced = resampled[:, pad : pad + record]
    print(
        f'resampled record: {resampled.shape} complex64; '
        f'iq_waveform slice {sliced.shape} ({sliced.nbytes / 2**20:.0f} MB)'
    )

    def to_host() -> Any:
        return np.array(sliced) if xp is np else sliced.get()

    params = sw.comms_5gnr.pss_params(
        sample_rate=SYNC_SAMPLE_RATE,
        subcarrier_spacing=SYNC_SCS,
        symbol_indexes=SYNC_CASE,
    )

    def ssb_spans() -> Any:
        return sw.comms_5gnr.get_5g_ssb_spans(
            sliced,
            params=params,
            fs_in=capture.sample_rate,
            frequency_offset=SSB_FREQUENCY_OFFSET,
        )

    for name, func, extra in [
        ('D2H column slice', to_host, 'host copy'),
        ('get_5g_ssb_spans', ssb_spans, f'{SSB_FREQUENCY_OFFSET / 1e6:+g} MHz'),
    ]:
        free_pool()
        peak = peak_pool_bytes(func)
        host, device, out = repeat_call(func, repeat, warmup)
        print_row(name, host, device, out.nbytes, f'{extra}, {pool_label(peak, xp)}')
        del out
    free_pool()


def profile_correlator(xp: ModuleType, cells: int, repeat: int, warmup: int) -> None:
    rng = np.random.default_rng(0)

    rows = [
        (
            'correlate pss',
            sw.comms_5gnr.pss_params(
                sample_rate=SYNC_SAMPLE_RATE,
                subcarrier_spacing=SYNC_SCS,
                symbol_indexes=SYNC_CASE,
            ),
            sw.comms_5gnr.pss_5g_nr(SYNC_SAMPLE_RATE, SYNC_SCS, xp=xp),
        ),
        (
            'correlate sss',
            sw.comms_5gnr.sss_params(
                sample_rate=SYNC_SAMPLE_RATE,
                subcarrier_spacing=SYNC_SCS,
                symbol_indexes=SYNC_CASE,
            ),
            sw.comms_5gnr.sss_5g_nr(SYNC_SAMPLE_RATE, SYNC_SCS, xp=xp)[:cells],
        ),
    ]

    for name, params, templates in rows:
        shape = (PORTS, SYNC_BLOCKS, params.corr_size)
        spans = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        spans = xp.asarray(spans.astype('complex64'))

        def run(
            spans: Any = spans, templates: Any = templates, params: Any = params
        ) -> Any:
            return sw.comms_5gnr.correlate_sync_spans(spans, templates, params=params)

        free_pool()
        try:
            peak = peak_pool_bytes(run)
            host, device, out = repeat_call(run, repeat, warmup)
        except Exception as ex:
            if 'OutOfMemory' not in type(ex).__name__:
                raise
            print(f'{name:<26} {"-":>9} {"-":>9} {"-":>9}  out of device memory')
            free_pool()
            continue
        label = f'{templates.shape[0]} cells, {pool_label(peak, xp)}'
        print_row(name, host, device, out.nbytes, label)
        del out, spans
        free_pool()


def profile_histogram(xp: ModuleType, repeat: int, warmup: int) -> None:
    nperseg = round(SAMPLE_RATE / FREQUENCY_RESOLUTION)
    hop = round(nperseg * (1 - FRACTIONAL_OVERLAP))
    frames = math.floor(DURATION * SAMPLE_RATE / hop)
    bins = round(ANALYSIS_BANDWIDTH / INTEGRATION_BANDWIDTH)
    spg = xp.asarray(
        np.random
        .default_rng(0)
        .uniform(-120, -60, (PORTS, frames, bins))
        .astype('float32')
    )
    edges = power.make_power_histogram_bin_edges(**HISTOGRAM, xp=xp)
    print(f'spectrogram: {spg.shape} float32')

    def from_ravel() -> Any:
        return xp.histogram(spg[0].ravel(), edges)[0]

    host, device, out = repeat_call(from_ravel, repeat, warmup)
    print_row('histogram ravel', host, device, out.nbytes, f'{edges.size - 1} bins')


def profile_stages(
    setup: Setup, xp: ModuleType, cells: int, repeat: int, warmup: int
) -> None:
    """print the stage table."""
    print_table_header()
    profile_cast_iq(setup, xp, repeat, warmup)
    raw = build_raw(setup, xp)
    resampled = profile_resampler(raw, xp, repeat, warmup)
    profile_record(resampled, setup.capture, xp, repeat, warmup)
    del raw, resampled
    free_pool()
    profile_correlator(xp, cells, repeat, warmup)
    profile_histogram(xp, repeat, warmup)


# %% Layer table
@dataclasses.dataclass
class Layer:
    label: str
    """the row label of the layer table"""
    func: Callable[[], Any]
    """the timed call"""
    prepare: Callable[[], None] | None = None
    """untimed setup that runs before each call of `func`; None for no setup"""


def profile_measurement(
    duration: float,
    backend: str,
    xp: ModuleType,
    repeat: int,
    trigger: bool,
    overwrite_x: bool,
) -> None:
    """print the time of the channel power measurement layer by layer.

    Args:
        duration: the capture duration (in s)
        backend: the array backend of the source spec, ``'numpy'`` or ``'cupy'``
        xp: the array namespace of the raw waveform
        repeat: the number of timed calls per row; the table prints the median
        trigger: True conditions the acquisition with the ``cellular_5g_pss_sync``
            trigger of the production source
        overwrite_x: True lets the resampling stage consume the raw waveform in
            place, as the pipeline does

    With `overwrite_x`, each fresh conditioning refills the raw waveform with ones
    before the stage consumes it. The ``refill raw.x`` row times that refill alone.
    The rows that condition a fresh acquisition inside the timed call (``resample
    stage``, ``registry call, cold``) include the refill. The ``aligned stage`` row
    conditions and resamples before the timed call and excludes the refill.
    """
    setup = build_setup(duration, backend, trigger)
    raw = build_raw(setup, xp)
    name = 'channel_power_time_series'
    spec = getattr(setup.analysis, name)
    info = sa.registry[type(spec)]
    kws = setup.analysis.to_dict()[name]
    capture = setup.capture

    def refill() -> None:
        raw.x[:] = 1

    def conditioned() -> Any:
        if overwrite_x:
            refill()
        return ss.lib.compute.attach_corrections(
            raw, signal_trigger=setup.trigger, overwrite_x=overwrite_x
        )

    # the preprocessor holds the acquisition weakly, so each fresh acquisition stays
    # bound to a local, or to `pending`, for the duration of the call
    def resample_stage() -> Any:
        fresh = conditioned()
        return fresh.preprocess(prefer_filtered=False, prefer_aligned=False)

    def registry_cold() -> Any:
        fresh = conditioned()
        return info.func(iq=fresh, capture=capture, as_xarray=False, **kws)

    pending: list[Any] = []

    def prepare_aligned() -> None:
        fresh = conditioned()
        fresh.preprocess(prefer_filtered=False, prefer_aligned=False)
        pending[:] = [fresh]

    def aligned_stage() -> Any:
        return pending[0].preprocess(prefer_filtered=True, prefer_aligned=True)

    iq = conditioned()
    layers: list[Layer] = []
    if overwrite_x:
        layers.append(Layer('refill raw.x', refill))
    layers.append(Layer('resample stage', resample_stage))
    iq.preprocess(prefer_filtered=False, prefer_aligned=False)
    x = iq.preprocess(prefer_filtered=True, prefer_aligned=True)

    layers += [
        Layer('aligned stage', aligned_stage, prepare_aligned),
        Layer(
            'body on the stage',
            lambda: power.evaluate_channel_power_time_series(x, capture, spec),
        ),
        Layer(
            'registry call, stages memoized',
            lambda: info.func(iq=iq, capture=capture, as_xarray=False, **kws),
        ),
        Layer('registry call, cold', registry_cold),
    ]

    def timed(layer: Layer) -> tuple[float, float]:
        if layer.prepare is not None:
            layer.prepare()
            device_synchronize()
        host, device, _ = time_call(layer.func)
        return host, device

    flags = ', '.join(
        f'{label} {"on" if value else "off"}'
        for label, value in (('trigger', trigger), ('overwrite_x', overwrite_x))
    )
    print(
        f'\nchannel_power_time_series on the ADA capture ({duration} s, '
        f'{raw.x.shape[0]} ports), median of {repeat}; {flags}'
    )
    print(f'{"layer":<32} {"host ms":>8} {"dev ms":>8}')
    for layer in layers:
        timed(layer)
        times = [timed(layer) for _ in range(repeat)]
        host = statistics.median(t[0] for t in times)
        dev = statistics.median(t[1] for t in times)
        print(f'{layer.label:<32} {host * 1e3:8.2f} {dev * 1e3:8.2f}', flush=True)
    pending.clear()
    free_pool()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--backend', choices=('cupy', 'numpy'), default='cupy')
    parser.add_argument('--duration', type=float, default=DURATION)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument(
        '--cells',
        type=int,
        default=DEFAULT_SSS_CELLS,
        help=f'number of SSS templates to correlate ({SSS_CELL_COUNT} is the full set)',
    )
    parser.add_argument(
        '--trigger',
        action='store_true',
        help='condition the acquisition of the layer table with the '
        'cellular_5g_pss_sync trigger',
    )
    parser.add_argument(
        '--overwrite-x',
        action='store_true',
        help='let the resampling stage of the layer table consume the raw waveform '
        'in place',
    )
    args = parser.parse_args()

    xp, _ = select_backend(args.backend)
    print(f'median of {args.repeat} calls after {args.warmup} warmup\n')

    setup = build_setup(args.duration, args.backend, trigger=False)
    profile_stages(setup, xp, args.cells, args.repeat, args.warmup)
    free_pool()
    profile_measurement(
        args.duration, args.backend, xp, args.repeat, args.trigger, args.overwrite_x
    )


if __name__ == '__main__':
    main()
