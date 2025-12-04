"""wrap lower-level striqt.waveform DSP calls to accept physical inputs and return xarray.DataArray"""

from __future__ import annotations as __

import typing

from . import dataarrays, specs, util

if typing.TYPE_CHECKING:
    import array_api_compat
    import numpy as np
    import scipy

    import striqt.waveform as iqwaveform
    from striqt.waveform._typing import ArrayType
else:
    np = util.lazy_import('numpy')
    scipy = util.lazy_import('scipy')
    array_api_compat = util.lazy_import('array_api_compat')
    iqwaveform = util.lazy_import('striqt.waveform')


def select_parameter_kws(locals_: dict, omit=(dataarrays.CAPTURE_DIM, 'out')) -> dict:
    """return the analysis parameters from the locals() evaluated at the beginning of analysis function"""

    items = list(locals_.items())
    return {k: v for k, v in items[1:] if k not in omit}


@util.lru_cache(8)
def _generate_iir_lpf(
    capture: specs.Capture,
    *,
    passband_ripple: float | int,
    stopband_attenuation: float | int,
    transition_bandwidth: float | int,
) -> np.ndarray:
    """
    Generate an elliptic IIR low pass filter for complex-valued waveforms.

    Args:
        passband_ripple:
            Maximum amplitude ripple in the passband of the frequency-response below unity gain (dB)
        stopband_attenuation:
            Maximum amplitude ripple in the passband of the frequency-response below unity gain (dB).
        transition_bandwidth:
            Passband-to-stopband transition width (Hz)

    Returns:
        Second-order sections (sos) representation of the IIR filter.
    """

    order, wn = scipy.signal.ellipord(
        capture.analysis_bandwidth / 2,
        capture.analysis_bandwidth / 2 + transition_bandwidth,
        passband_ripple,
        stopband_attenuation,
        False,
        capture.sample_rate,
    )

    sos = scipy.signal.ellip(
        order,
        passband_ripple,
        stopband_attenuation,
        wn,
        'lowpass',
        False,
        'sos',
        capture.sample_rate,
    )

    return sos


def iir_filter(
    iq: ArrayType,
    capture: specs.Capture,
    *,
    passband_ripple: float | int,
    stopband_attenuation: float | int,
    transition_bandwidth: float | int,
    axis=0,
    out=None,
):
    filter_kws = select_parameter_kws(locals())
    sos = _generate_iir_lpf(capture, **filter_kws)

    xp = iqwaveform.util.array_namespace(iq)

    if util.is_cupy_array(iq):
        from . import cuda_kernels

        sos = xp.asarray(sos)
        return cuda_kernels.sosfilt(sos.astype('float32'), iq, axis=axis)

    else:
        return scipy.signal.sosfilt(sos.astype('float32'), iq, axis=axis)


def ola_filter(
    iq: ArrayType,
    capture: specs.Capture,
    *,
    nfft: int,
    window: typing.Any = 'hamming',
    out=None,
    cache=None,
):
    kwargs = select_parameter_kws(locals())

    return iqwaveform.fourier.ola_filter(
        iq,
        fs=capture.sample_rate,
        passband=(-capture.analysis_bandwidth / 2, capture.analysis_bandwidth / 2),
        **kwargs,
    )
