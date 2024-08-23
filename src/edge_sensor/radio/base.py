from __future__ import annotations
from functools import lru_cache
from math import ceil
import typing

import labbench as lb
from labbench import paramattr as attr
import numpy as np
from .. import structs
from channel_analysis import type_stubs

iqwaveform = lb.util.lazy_import('iqwaveform')
pd = lb.util.lazy_import('pandas')


TRANSIENT_HOLDOFF_WINDOWS = 1


class RadioDevice(lb.Device):
    analysis_bandwidth = attr.value.float(
        None,
        min=1e-6,
        label='Hz',
        help='bandwidth of the digital bandpass filter (or None to bypass)',
    )

    calibration = attr.value.Path(
        None,
        help='path to a calibration file, or None to skip calibration',
    )

    duration = attr.value.float(
        10e-3, min=0, label='s', help='receive waveform capture duration'
    )

    periodic_trigger = attr.value.float(
        None,
        allow_none=True,
        help='if specified, acquisition start times will begin at even multiples of this',
    )

    _downsample = attr.value.float(1.0, min=0, help='backend_sample_rate/sample_rate')

    @attr.property.str(sets=False, cache=True, help='unique radio hardware identifier')
    def id(self):
        raise NotImplementedError

    @property
    def master_clock_rate(self):
        return type(self).backend_sample_rate.max

    def _prepare_buffer(self, capture: structs.RadioCapture):
        samples_in, _ = get_capture_buffer_sizes(self, capture, include_holdoff=True)

        # total buffer size for 2 values per IQ sample
        size_in = 2 * samples_in

        if self._inbuf is None or self._inbuf.size < size_in:
            self._logger.debug(
                f'allocating input sample buffer ({size_in * 2 /1e6:0.2f} MB)'
            )
            self._inbuf = np.empty((size_in,), dtype=np.float32)
            self._logger.debug('done')

    def acquire(
        self,
        capture: structs.RadioCapture,
        next_capture: typing.Union[structs.RadioCapture, None] = None,
        correction: bool = True,
    ) -> tuple[np.array, type_stubs.TimestampType]:
        from .. import iq_corrections

        count, _ = get_capture_buffer_sizes(self, capture)

        with lb.stopwatch('acquire', logger_level='debug'):
            self.arm(capture)
            self.channel_enabled(True)
            timestamp = pd.Timestamp('now')
            self._prepare_buffer(capture)
            iq = self._read_stream(count)
            self.channel_enabled(False)

            if next_capture is not None:
                self.arm(next_capture)

            if correction:
                iq = iq_corrections.resampling_correction(iq, capture, self)

            return iq, timestamp

    def setup(self, radio_config: structs.RadioSetup):
        # TODO: the other parameters too
        self.calibration = radio_config.calibration
        self.periodic_trigger = radio_config.periodic_trigger
        if radio_config.preselect_if_frequency is not None:
            raise IOError('external frequency conversion is not yet supported')

    def arm(self, capture: structs.RadioCapture):
        """to be implemented in subclasses"""
        raise NotImplementedError

    def _read_stream(self, samples: int) -> np.ndarray:
        """to be implemented in subclasses"""
        raise NotImplementedError

    def get_capture_struct(self) -> structs.RadioCapture:
        """generate the currently armed capture configuration for the specified channel"""
        if self.lo_offset == 0:
            lo_shift = 'none'
        elif self.lo_offset < 0:
            lo_shift = 'left'
        elif self.lo_offset > 0:
            lo_shift = 'right'

        return structs.RadioCapture(
            # RF and leveling
            center_frequency=self.center_frequency(),
            channel=self.channel(),
            gain=self.gain(),
            # acquisition
            duration=self.duration,
            sample_rate=self.sample_rate(),
            # filtering and resampling
            analysis_bandwidth=self.analysis_bandwidth,
            lo_shift=lo_shift,
            # future: external frequency conversion support
            # if_frequency=None,
            # lo_gain=0,
            # rf_gain=0,
        )

    def close(self):
        self._inbuf = None


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
        fs_sdr, lo_offset, kws = iqwaveform.fourier.design_cola_resampler(
            fs_base=master_clock_rate,
            fs_target=capture.sample_rate,
            bw=capture.analysis_bandwidth,
            bw_lo=0.75e6,
            shift=lo_shift,
        )

        return fs_sdr, lo_offset, kws

    elif lo_shift:
        raise ValueError('lo_shift requires gpu_resample=True')
    elif master_clock_rate < capture.sample_rate:
        raise ValueError(
            f'upsampling above {master_clock_rate/1e6:f} MHz requires gpu_resample=True'
        )
    else:
        # use the SDR firmware to set the desired sample rate
        return iqwaveform.fourier.design_cola_resampler(
            fs_base=capture.sample_rate,
            fs_target=capture.sample_rate,
            bw=capture.analysis_bandwidth,
            shift=False,
        )


@lru_cache(30000)
def _get_capture_buffer_sizes_cached(
    master_clock_rate: float,
    periodic_trigger: float | None,
    capture: structs.RadioCapture,
    include_holdoff: bool = False,
):
    if iqwaveform.power_analysis.isroundmod(capture.duration * capture.sample_rate, 1):
        samples_out = round(capture.duration * capture.sample_rate)
    else:
        msg = f'duration must be an integer multiple of the sample period (1/{capture.sample_rate} s)'
        raise ValueError(msg)

    _, _, analysis_filter = design_capture_filter(master_clock_rate, capture)

    samples_in = ceil(samples_out * analysis_filter['nfft'] / analysis_filter['nfft_out'])

    if include_holdoff and periodic_trigger is not None:
        # add holdoff samples needed for the periodic trigger
        samples_in += ceil(analysis_filter['fs'] * periodic_trigger)

    if analysis_filter and capture.gpu_resample:
        samples_in += TRANSIENT_HOLDOFF_WINDOWS * analysis_filter['nfft']
        samples_out = 1.5*iqwaveform.fourier._istft_buffer_size(
            samples_in,
            window=analysis_filter['window'],
            nfft_out=analysis_filter['nfft_out'],
            nfft=analysis_filter['nfft'],
            extend=True,
        )

    return samples_in, samples_out


def get_capture_buffer_sizes(
    radio: RadioDevice, capture=None, include_holdoff=False
) -> tuple[int, int]:
    if capture is None:
        capture = radio.get_capture_struct()

    return _get_capture_buffer_sizes_cached(
        master_clock_rate=radio.master_clock_rate,
        periodic_trigger=radio.periodic_trigger,
        capture=capture,
        include_holdoff=include_holdoff,
    )


def radio_subclasses(subclass=RadioDevice):
    """returns a list of radio subclasses that have been imported"""

    subs = {c.__name__: c for c in subclass.__subclasses__()}

    for sub in list(subs.values()):
        subs.update(radio_subclasses(sub))

    subs = {name: c for name, c in subs.items() if not name.startswith('_')}

    return subs


def find_radio_cls_by_name(
    name, parent_cls: type[RadioDevice] = RadioDevice
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
