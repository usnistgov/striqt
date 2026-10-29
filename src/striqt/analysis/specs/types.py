import fractions
from typing import Annotated, Literal, Union

import striqt.waveform as sw

from .helpers import Meta

AnalysisBandwidth = Annotated[float, Meta('Analysis bandwidth', 'Hz')]
AsXArray = Literal['delayed', True, False]
CellSSBSymbolIndexes = Annotated[
    sw.typing.CellSSBIndexes,
    Meta(
        'index locations in the SSB, or a cell search case from 3GPP TS 38.213 Sec. 4.1)'
    ),
]
CellularFrame = Annotated[
    Union[str, None],
    Meta(
        "string composed of ('d', 'u', 's') specifying the sequence of slots in 1 TDD cellular frame, or None to fill with downlink"
    ),
]
CellularSpecialSymbols = Annotated[
    Union[str, None],
    Meta(
        "string composed of the characters ('d', 'u', 'f') indicating the sequence of symbol types (when 's' is in frame_slots)"
    ),
]
CellularSubcarrierSpacing = Annotated[
    float, Meta('Subcarrier spacing (15e3, 30e3, 60e3, etc)', units='Hz')
]
CellularSubcarrierSpacingTuple = Annotated[
    Union[float, tuple[float, ...]],
    Meta('One or more subcarrier spacings (15e3, 30e3, 60e3, etc)', units='Hz'),
]
CellularCyclicPrefix = Annotated[
    Literal['normal', 'extended'],
    Meta('cyclic prefix length of the 3GPP numerology (TS 38.211 Table 4.2-1)'),
]
CellularAverageSlots = Annotated[
    bool, Meta('True to coarsen spectrogram bins by averaging 1-symbol time resolution')
]
CellularAverageRBs = Annotated[
    Union[bool, Literal['half']],
    Meta(
        'frequency resolution of the spectrogram: False for 1 subcarrier, True for 1 '
        "resource block (12 subcarriers), or 'half' for 6 subcarriers"
    ),
]
CenterFrequencyScalar = Annotated[float, Meta('RF center frequency', 'Hz', gt=0)]
CenterFrequency = Annotated[
    Union[CenterFrequencyScalar, tuple[CenterFrequencyScalar, ...]],
    Meta('Center frequency for each port', 'Hz'),
]
CyclicPeriod = Annotated[
    float,
    Meta(
        'Cycle period of the channel power statistics, a whole multiple of '
        '`detector_period` and a common multiple of the expected signal periods',
        's',
    ),
]
CyclicStatistics = Annotated[
    tuple[Union[str, float], ...],
    Meta(
        "Statistics evaluated across cycles at each cycle lag: 'min', 'mean', 'max', 'median', or a quantile in (0, 1)"
    ),
]
DetectorPeriod = Annotated[
    fractions.Fraction,
    Meta('Power detector bin duration, a whole multiple of the sample period', 's'),
]
Duration = Annotated[float, Meta('Duration of the analysis waveform', 's')]
GuardBandwidths = Annotated[
    tuple[float, float],
    Meta(
        'Bandwidths excluded at the (low, high) edges of the analysis band',
        units='Hz',
    ),
]
LOBandstop = Annotated[
    float,
    Meta(
        'mask with float("nan") at this bandwidth centered at baseband DC', units='Hz'
    ),
]
MaxBeams = Annotated[int, Meta('limit the beam count in 5G sync evaluation', gt=0)]
MaxLagSymbols = Annotated[
    int,
    Meta('limit the number of symbols of lag in the 5G sync correlator', ge=1, le=6),
]
PerPort = Annotated[
    bool, Meta('whether to evaluate signal synchronization separately on each port')
]
PowerDetectors = Annotated[
    tuple[str, ...],
    Meta("Power detectors evaluated in each bin: 'rms', 'peak', or both"),
]
PowerBinMin = Annotated[float, Meta('Minimum power bin', units='dB power')]
PowerBinMax = Annotated[float, Meta('Maximum power bin', units='dB power')]
PowerBinStep = Annotated[float, Meta('Power bin resolution', units='dB')]
SampleRate = Annotated[
    float, Meta('Sample rate of the analyzed IQ waveform', 'S/s', gt=0)
]
WindowFill = Annotated[
    float, Meta('Fraction of a symbol to fill with weighting function', gt=0, le=1)
]
WindowType = Annotated[
    Union[str, tuple[str, float]],
    Meta('window function specification following `scipy.signal.get_window`'),
]
