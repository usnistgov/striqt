from __future__ import annotations as __

from pathlib import Path
from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Annotated, Any, Literal, Optional, Union
from striqt.analysis.lib.specs import Meta

if _TYPE_CHECKING:
    import pandas as pd
else:
    from ..lib import util as _util
    pd = _util.lazy_import('pandas')


AliasCandidateMatches = Annotated[
    tuple[dict[str, Any], ...],
    Meta('one or more dictionaries of valid match sets to "or"'),
]
AliasMatch = Annotated[
    dict[str, AliasCandidateMatches],
    Meta('key: alias field value, value: a list of match conditions for that value'),
]
AmbientTemperature = Annotated[
    float, Meta(standard_name='Ambient temperature', units='K')
]
AnalysisBandwidth = Annotated[
    float, Meta('Bandwidth of the analysis filter (or inf to disable)', 'Hz', gt=0)
]
ArrayBackend = Annotated[
    Literal['numpy', 'cupy'],
    Meta('array module to use to set compute device: numpy = cpu, cupy = gpu'),
]
BackendSampleRate = Annotated[float, Meta('Source sample rate', 'Hz', gt=0)]
BaseClockRate = Annotated[
    float, Meta('Base sample rate used inside the source', 'Hz', gt=0)
]
CenterFrequency = Annotated[float, Meta('RF center frequency', 'Hz', gt=0)]
ClockSource = Annotated[
    Literal['internal', 'external', 'gps'],
    Meta('Hardware source for the frequency reference'),
]
ContinuousTrigger = Annotated[
    bool,
    Meta('Whether to trigger immediately after each call to acquire() when armed'),
]
StartDelay = Annotated[float, Meta('Delay in acquisition start time', 's', gt=0)]
ENR = Annotated[float, Meta(standard_name='Excess noise ratio', units='dB')]
ExtensionPath = Annotated[
    str,
    Meta('path to append to sys.path before extension imports'),
]
FileMetadata = Annotated[dict, Meta('any capture fields not included in the file')]
FileLoop = Annotated[
    bool, Meta('whether to loop the file to create longer IQ waveforms')
]
Format = Annotated[
    Literal['auto', 'mat', 'tdms'], Meta('data format or auto to guess by extension'),
]
FrequencyOffset = Annotated[float, Meta('Baseband frequency offset', 'Hz')]
GainScalar = Annotated[float, Meta('Gain setting', 'dB')]
Gain = Annotated[
    Union[GainScalar, tuple[GainScalar, ...]],
    Meta('Gain setting for each channel', 'dB'),
]
GaplessRepeat = Annotated[
    bool,
    Meta('whether to raise an exception on overflows between identical captures'),
]
LOShift = Annotated[Literal['left', 'right', 'none'], Meta('LO shift direction')]
MockSensor = Annotated[
    Optional[str],
    Meta('replace the bound sensor with one from this binding name'),
]
ModuleName = Annotated[
    Union[str, None],
    Meta('name of the extension module that calls bind_sensor'),
]
NoiseDiodeEnabled = Annotated[bool, Meta(standard_name='Noise diode enabled')]
OnOverflowType = Literal['ignore', 'except', 'log']
PortScalar = Annotated[int, Meta('Input port index', ge=0)]
Port = Annotated[
    Union[PortScalar, tuple[PortScalar, ...]],
    Meta('Input port indices'),
]
PSD = Annotated[float, Meta('noise total channel power', 'mW/Hz', ge=0)]
Power = Annotated[float, Meta('peak power level', 'dB', gt=0)]
Period = Annotated[float, Meta('waveform period', 's', ge=0)]
ReceiveRetries = Annotated[
    int, Meta('number of acquisition retry attempts on stream error', ge=0)
]
SinkClass = Annotated[
    Union[str, Literal['striqt.sensor.writers.CaptureAppender']],
    Meta('data sink class to import and use'),
]
SNR = Annotated[float, Meta('SNR with added noise ', 'dB')]
SourceID = Annotated[str, Meta('Source UUID string')]
StartTime = Annotated[
    'pd.Timestamp', Meta('Acquisition start time of the first capture')
]
StoreFormat = Annotated[
    Literal['zip', 'directory'], Meta('serialization output format of the sink')
]
SweepStartTime = Annotated['pd.Timestamp', Meta('Capture acquisition start time')]
SyncSource = Annotated[
    str, Meta('name of a registered waveform alignment function'),
]
TimeOffset = Annotated[float, Meta('start time offset', 's')]
TimeSource = Annotated[
    Literal['host', 'internal', 'external', 'gps'],
    Meta('Hardware source for timestamps'),
]
TimeSyncEveryCapture = Annotated[
    bool, Meta('whether to sync to PPS before each capture in a sweep')
]
TransportDType = Annotated[
    Literal['int16', 'float32'], Meta('data transfer type to use inside the source')
]
OverloadDetectFlag = Annotated[
    Union[bool, Literal['auto']], Meta('whether to detect overload')
]
WarmupSweep = Annotated[
    bool, Meta('if True, run empty buffers through the GPU before sweeping'),
]
WaveformInputPath = Annotated[Path, Meta('path to the waveform data file')]
ZarrSelect = Annotated[
    dict, Meta('dictionary to select in the data as .sel(**select)')
]