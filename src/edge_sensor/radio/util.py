from __future__ import annotations
from functools import lru_cache
from iqwaveform import fourier
from iqwaveform.power_analysis import isroundmod
from typing import Type
import contextlib

import numpy as np

from .. import structs
from ..util import import_cupy_with_fallback_warning
from .base import RadioDevice


TRANSIENT_HOLDOFF_WINDOWS = 1


@lru_cache(30000)
def design_capture_filter(
    master_clock_rate: float, capture: structs.RadioCapture
) -> tuple[float, float, dict]:
    """design a filter specified by the capture for a radio with the specified MCR.

    For the return value, see `iqwaveform.fourier.design_cola_resampler`
    """
    if str(capture.lo_shift).lower() == 'none':
        lo_shift = False
    else:
        lo_shift = capture.lo_shift

    if capture.gpu_resample:
        # use GPU DSP to resample from integer divisor of the MCR
        return fourier.design_cola_resampler(
            fs_base=max(capture.sample_rate, master_clock_rate),
            fs_target=capture.sample_rate,
            bw=capture.analysis_bandwidth,
            bw_lo=0.75e6,
            shift=lo_shift,
        )
    elif lo_shift:
        raise ValueError('lo_shift requires gpu_resample=True')
    elif master_clock_rate < capture.sample_rate:
        raise ValueError(
            f'upsampling above {master_clock_rate/1e6:f} MHz requires gpu_resample=True'
        )
    else:
        # use the SDR firmware to set the desired sample rate
        return fourier.design_cola_resampler(
            fs_base=capture.sample_rate,
            fs_target=capture.sample_rate,
            bw=capture.analysis_bandwidth,
            shift=False,
        )


def warm_resampler_design_cache(
    radio: RadioDevice, captures: list[structs.RadioCapture]
):
    """warm up the cache of resampler designs"""

    for c in captures:
        design_capture_filter(radio._master_clock_rate, c)


@lru_cache(30000)
def _get_capture_buffer_sizes_cached(master_clock_rate: float, periodic_trigger:float|None, capture: structs.RadioCapture, include_holdoff: bool=False):
    if isroundmod(capture.duration * capture.sample_rate, 1):
        Nout = round(capture.duration * capture.sample_rate)
    else:
        msg = f'duration must be an integer multiple of the sample period (1/{capture.sample_rate} s)'
        raise ValueError(msg)

    _, _, analysis_filter = design_capture_filter(master_clock_rate, capture)

    Nin = round(
        np.ceil(Nout * analysis_filter['fft_size'] / analysis_filter['fft_size_out'])
    )

    if include_holdoff and periodic_trigger is not None:
        # account for maximum holdoff needed for the periodic trigger
        Nin += np.rint(np.ceil(analysis_filter['fs']*periodic_trigger))
        print(Nin)

    if analysis_filter:
        Nin += TRANSIENT_HOLDOFF_WINDOWS * analysis_filter['fft_size']
        Nout = fourier._istft_buffer_size(
            Nin,
            window=analysis_filter['window'],
            fft_size_out=analysis_filter['fft_size_out'],
            fft_size=analysis_filter['fft_size'],
            extend=True,
        )

    return Nin, Nout


def get_capture_buffer_sizes(radio: RadioDevice, capture=None, include_holdoff=False) -> tuple[int, int]:
    if capture is None:
        capture = radio.get_capture_struct()

    return _get_capture_buffer_sizes_cached(
        master_clock_rate=radio._master_clock_rate,
        periodic_trigger = radio.periodic_trigger,
        capture=capture,
        include_holdoff=include_holdoff
    )



def find_largest_capture(radio, captures):
    from edge_sensor.radio import soapy

    sizes = [
        sum(soapy.get_capture_buffer_sizes(radio, c))
        for c in captures
    ]

    return captures[sizes.index(max(sizes))]


def empty_capture(radio: RadioDevice, capture: structs.RadioCapture):
    """evaluate a capture on an empty buffer to warm up a GPU"""
    from .. import iq_corrections

    xp = import_cupy_with_fallback_warning()

    nin, _ = get_capture_buffer_sizes(radio, capture, include_holdoff=True)
    radio._prepare_buffer(capture)
    iq = xp.array(radio._inbuf, copy=False).view('complex64')[:nin]
    ret = iq_corrections.resampling_correction(iq, capture, radio)

    return ret


def radio_subclasses(subclass=RadioDevice):
    """returns a list of radio subclasses that have been imported"""

    subs = {c.__name__: c for c in subclass.__subclasses__()}

    for sub in list(subs.values()):
        subs.update(radio_subclasses(sub))

    subs = {name: c for name, c in subs.items() if not name.startswith('_')}

    return subs


def find_radio_cls_by_name(
    name, parent_cls: Type[RadioDevice] = RadioDevice
) -> RadioDevice:
    """returns a list of radio subclasses that have been imported"""

    mapping = radio_subclasses(parent_cls)

    if name in mapping:
        return mapping[name]
    else:
        raise AttributeError(
            f'invalid driver {repr(name)}. valid names: {tuple(mapping.keys())}'
        )


def is_same_resource(r1: str | dict, r2: str | dict):
    if isinstance(r1, str):
        return r1 == r2
    else:
        return set(r1.items()) == set(r2.items())


@contextlib.contextmanager
def prepare_gpu(radio, captures, spec, swept_fields):
    """perform analysis imports and warm up the gpu evaluation graph"""

    try:
        import cupy
    except ModuleNotFoundError:
        # skip priming if a gpu is unavailable
        yield None
        return

    from edge_sensor.radio import util
    from edge_sensor import actions
    import labbench as lb

    analyzer = actions._RadioCaptureAnalyzer(
        radio, analysis_spec=spec, remove_attrs=swept_fields
    )

    with lb.stopwatch('priming gpu'):
        # select the capture with the largest size
        capture = util.find_largest_capture(radio, captures)
        iq = util.empty_capture(radio, capture)
        analyzer(iq, timestamp=None, capture=capture)
        # soapy.free_cuda_memory()

    yield None
