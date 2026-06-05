from typing import Annotated, Literal, Union
from .helpers import Meta
import striqt.waveform as sw

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
    Union[Literal['normal'], Literal['extended']], Meta('the 3GPP cyclic prefix type')
]
CellularAverageSlots = Annotated[
    bool, Meta('True to coarsen spectrogram bins by averaging 1-symbol time resolution')
]
CellularAverageRBs = Annotated[
    Union[bool, Literal['half']],
    Meta(
        'True (or "half") to coarsen spectrogram bins by integrating 1-subcarrier frequency resolution into 1- (or 1/2)-resource block'
    ),
]
Duration = Annotated[float, Meta('Duration of the analysis waveform', 's')]
GuardBandwidths = Annotated[
    tuple[float, float],
    Meta('Channel guard bandwidths to ignore on the left and right sides', units='Hz'),
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
PowerBinMin = Annotated[float, Meta('Minimum power bin', units='dB power')]
PowerBinMax = Annotated[float, Meta('Maximum power bin', units='dB power')]
PowerBinStep = Annotated[float, Meta('Power bin resolution', units='dB')]
SampleRate = Annotated[float, Meta('Analysis sample rate', 'S/s')]
WindowFill = Annotated[
    float, Meta('Fraction of a symbol to fill with weighting function', gt=0, le=1)
]
WindowType = Annotated[
    Union[str, tuple[str, float]],
    Meta('window function specification following `scipy.signal.get_window`'),
]
