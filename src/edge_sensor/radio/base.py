from __future__ import annotations
import labbench as lb
from .. import diagnostic_data, structs
from functools import lru_cache
from iqwaveform import fourier
from iqwaveform.power_analysis import isroundmod
import numpy as np


TRANSIENT_HOLDOFF_WINDOWS = 1


class RadioBase(lb.Device):
    _inbuf = None
    _outbuf = None

    def build_index_variables(self):
        return diagnostic_data.index_variables()

    def build_metadata(self):
        return dict(
            super().build_metadata(), **diagnostic_data.package_host_resources()
        )

    def _prepare_buffer(self, input_size, output_size):
        raise NotImplementedError


@lru_cache(30000)
def design_capture_filter(
    master_clock_rate: float, capture: structs.RadioCapture
) -> tuple[float, float, dict]:
    if str(capture.lo_shift).lower() == 'none':
        lo_shift = False
    else:
        lo_shift = capture.lo_shift

    # fs_backend, lo_offset, self.analysis_filter
    return fourier.design_cola_resampler(
        fs_base=master_clock_rate,
        fs_target=capture.sample_rate,
        bw=capture.analysis_bandwidth,
        bw_lo=0.75e6,
        shift=lo_shift,
    )


@lru_cache(30000)
def get_capture_buffer_sizes(
    master_clock_rate: float, capture: structs.RadioCapture
) -> tuple[int, int]:
    if isroundmod(capture.duration * capture.sample_rate, 1):
        Nout = round(capture.duration * capture.sample_rate)
    else:
        msg = f'duration must be an integer multiple of the sample period (1/{capture.sample_rate} s)'
        raise ValueError(msg)

    _, _, analysis_filter = design_capture_filter(master_clock_rate, capture)

    Nin = round(
        np.ceil(Nout * analysis_filter['fft_size'] / analysis_filter['fft_size_out'])
    )

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


def empty_capture(radio: RadioBase, capture: structs.RadioCapture):
    """evaluate a capture on an empty buffer to warm up a GPU"""

    import cupy as cp
    from edge_sensor import iq_corrections

    nin, _ = get_capture_buffer_sizes(radio._master_clock_rate, capture)
    radio._prepare_buffer(capture)
    iq = cp.array(radio._inbuf, copy=False).view('complex64')[:nin]
    ret = iq_corrections.resampling_correction(iq, capture, radio)

    return ret
