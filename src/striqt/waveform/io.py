# routines for reading spectrum monitoring data files

import numpy as np
import json
from pathlib import Path
from .util import lazy_import
from . import _typing

signal = lazy_import('scipy.signal')
pd = lazy_import('pandas')


def extract_ntia_calibration_metadata(metadata: dict) -> dict:
    temp_K = None
    noise_fig_dB = None
    gain_dB = None

    # Look for calibration annotation
    for a in metadata['annotations']:
        if a['ntia-core:annotation_type'] == 'CalibrationAnnotation':
            temp_K = a['ntia-sensor:temperature'] + 273.15  # C to K
            noise_fig_dB = a['ntia-sensor:noise_figure_sensor']
            gain_dB = a['ntia-sensor:gain_preselector']
            break
    else:
        gain_dB = None

    return {
        'ambient temperature (K)': temp_K,
        'noise figure (dB)': noise_fig_dB,
        'gain (dB)': gain_dB,
    }


def read_sigmf_metadata(metadata_fn, ntia=False) -> tuple[_typing.DataFrameType, float]:
    with open(metadata_fn, 'r') as fd:
        metadata = json.load(fd)

    df = pd.DataFrame(metadata['captures'])

    df.columns = [n.replace('core:', '') for n in df.columns]

    if ntia:
        cal = extract_ntia_calibration_metadata(metadata)
    else:
        cal = {}

    return (
        dict(df.set_index('sample_start').frequency),
        dict(df.set_index('sample_start').datetime),
        metadata['global']['core:sample_rate'],
        cal,
    )


def read_sigmf(
    metadata_path: str,
    force_sample_rate: float = None,
    sigmf_data_ext='.npy',
    stack=False,
    ntia_extensions=False,
    z0=50,
):
    metadata_path = Path(metadata_path)

    """pack a DataFrame with data read from a SigMF modified for npy file format"""

    center_freqs, timestamps, sample_rate, cal = read_sigmf_metadata(
        metadata_path, ntia=ntia_extensions
    )

    if force_sample_rate is not None:
        sample_rate = force_sample_rate

    if sigmf_data_ext == '.npy':
        data_fn = metadata_path.with_suffix('.sigmf-data.npy')

        x = np.load(data_fn)
    else:
        raise TypeError(f'SIGMF data extension {sigmf_data_ext} not supported')

    x_split = np.array_split(x, list(center_freqs.keys())[1:])

    if stack:
        x_split = np.vstack(x_split).T

    if cal.get('gain (dB)', None) is not None:
        print('gain dB: ', cal['gain (dB)'])
        gain = 10 ** (cal['gain (dB)'] / 10.0)
        x_split = x_split / np.sqrt(gain * 2 / z0)
    elif ntia_extensions:
        raise LookupError('no calibration data is available in NTIA extensions')

    return (x_split, np.array(list(center_freqs.values())), 1.0 / sample_rate, cal)


def read_sigmf_to_df(
    metadata_path: str, force_sample_rate: float = None, sigmf_data_ext='.npy'
) -> np.array:
    x_split, center_freqs, Ts = read_sigmf(**locals())

    return waveform_to_frame(
        x_split, Ts, columns=pd.Index(center_freqs / 1e9), name='Frequency (Hz)'
    )


def waveform_to_frame(
    waveform: np.array,
    Ts: float,
    columns: _typing.IndexType = None,
    column_name=None,
) -> tuple[_typing.SeriesType, _typing.DataFrameType]:
    """packs IQ data into a pandas Series or DataFrame object.

    The input waveform `iq` may have shape (N,) or (N,M), representing a single
    waveform or M different waveforms, respectively.

    Args:
        iq: Complex-valued time series representing an IQ waveform.
        Ts: The sample period of the IQ waveform.
        columns: The list of column names to use if `iq` has 2-dimensions

    Returns:
        If iq.ndim == 1, then pd.Series, otherwise pd.DataFrame, with a time index
    """

    if waveform.ndim == 2:
        if columns is None:
            columns = np.arange(waveform.shape[1])
        obj = pd.DataFrame(waveform, columns=columns)

        if column_name is not None:
            obj.columns.name = column_name

    elif waveform.ndim == 1:
        obj = pd.Series(waveform)
    else:
        raise TypeError('iq must have 1 or 2 dimensions')

    obj.index = pd.Index(
        np.linspace(0, Ts * waveform.shape[0], waveform.shape[0], endpoint=False),
        name='Time elapsed (s)',
    )

    return obj


def resample_iq(iq: np.array, Ts, scale, axis=0):
    N = int(np.round(iq.shape[0] * scale))
    return signal.resample(iq, num=N, axis=axis), Ts / scale
