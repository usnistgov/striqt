"""generate reference synthetic waveforms for tests and the synthetic sensor sources.

Each generator returns an array of shape ``(ports, count)`` in the array namespace
`xp` and is *windowable*: `start_index` is an absolute sample index, so a call with
`start_index` and `count` is exactly the slice ``[:, start_index:start_index + count]``
of the whole-capture call. Index 0 is the first sample of a capture, and a negative
`start_index` extends the waveform backward in time for the pre-roll that the
sensor's resampler and filter overlaps consume.
"""

from __future__ import annotations as __

import typing

from .util import np

if typing.TYPE_CHECKING:
    from types import ModuleType

    from striqt.waveform.lib.typing import DTypeLike

    from .typing import Array


def _resolve_count(
    duration: float | None,
    sample_rate: float,
    start_index: int,
    count: int | None,
    ports: int,
) -> int:
    if ports < 1:
        raise ValueError(f'ports must be at least 1, not {ports}')

    if count is None:
        if duration is None:
            raise ValueError('either duration or count must be given')
        count = round(duration * sample_rate) - start_index

    if count < 0:
        raise ValueError(f'count must be non-negative, not {count}')

    return int(count)


def _sample_indices(start_index: int, count: int, xp: ModuleType) -> Array:
    return xp.arange(start_index, start_index + count, dtype='int64')


def _expand_ports(x: Array, ports: int, xp: ModuleType) -> Array:
    # a broadcast view would be read-only, which is surprising for a waveform
    return xp.broadcast_to(x, (ports, x.shape[-1])).copy()


# offset of the pre-roll seed from the user's seed; both fit RandomState's
# 32-bit range when the user's seed is below 2**31
PREROLL_SEED_OFFSET = 2**31


def _complex_normal_draws(*, ports: int, size: int, seed: int, xp: ModuleType) -> Array:
    """generate complex64 draws whose real and imaginary parts are standard normal.

    The draws are interleaved port-by-port within each sample, so the first `size`
    samples of every port are a prefix of the stream from ``RandomState(seed)``.

    Returns:
        an array of shape ``(ports, size)`` and dtype complex64 in the namespace `xp`
    """
    # RandomState rather than default_rng: it is the generator API that numpy and
    # cupy share
    gen = xp.random.RandomState(seed=seed)
    values = gen.standard_normal(size=2 * size * ports).astype('float32')
    return values.view('complex64').reshape(size, ports).T


def _windowed_normal_draws(
    *, ports: int, start_index: int, count: int, seed: int, xp: ModuleType
) -> Array:
    """generate complex64 standard normal draws at absolute indices from `start_index`.

    Indices ``i >= 0`` are the stream from ``RandomState(seed)``; index ``i < 0`` is
    draw ``-i - 1`` of the stream from ``RandomState(seed + PREROLL_SEED_OFFSET)``,
    so the pre-roll runs backward from index -1 and is independent of the forward
    stream.

    Returns:
        an array of shape ``(ports, count)`` and dtype complex64 in the namespace
        `xp`, covering indices ``[start_index, start_index + count)``
    """
    stop = start_index + count
    forward = _complex_normal_draws(ports=ports, size=max(stop, 0), seed=seed, xp=xp)
    forward = forward[:, max(start_index, 0) :]

    if start_index >= 0:
        return forward

    preroll = _complex_normal_draws(
        ports=ports, size=-start_index, seed=seed + PREROLL_SEED_OFFSET, xp=xp
    )
    preroll = preroll[:, ::-1][:, : min(count, -start_index)]

    return xp.concatenate([preroll, forward], axis=1)


def tone(
    duration: float | None,
    sample_rate: float,
    *,
    frequency: float = 0.0,
    ports: int = 1,
    start_index: int = 0,
    count: int | None = None,
    xp: ModuleType = np,
    dtype: DTypeLike = 'complex64',
) -> Array:
    """generate a unit-amplitude complex tone at `frequency`.

    The phase is zero at absolute sample index 0 regardless of `start_index`, and
    advances `frequency` cycles per second. Mean power is 1, which the power
    measurements read as 0 dBm, since they treat ``|iq|**2`` as mW.

    Args:
        duration: capture duration, in s, used only when `count` is None
        sample_rate: IQ sample rate, in Hz
        frequency: tone frequency, in Hz relative to the center frequency
        ports: number of receive ports, which all get the same samples
        start_index: absolute index of the first returned sample
        count: number of samples, or None for `duration` less `start_index`
        xp: the array namespace to generate with
        dtype: the returned complex dtype

    Returns:
        an array of shape ``(ports, count)`` and dtype `dtype` in the namespace `xp`

    Raises:
        ValueError: when `ports` is below 1, when `duration` and `count` are both
            None, or when the resolved `count` is negative
    """
    count = _resolve_count(duration, sample_rate, start_index, count, ports)
    i = _sample_indices(start_index, count, xp)
    phase_scale = (2j * np.pi * frequency) / sample_rate
    x = xp.exp(phase_scale * i).astype(dtype)
    return _expand_ports(x, ports, xp)


def single_tone(
    duration: float | None,
    sample_rate: float,
    *,
    frequency_offset: float = 0.0,
    snr: float | None = None,
    lo_offset: float = 0.0,
    ports: int = 1,
    start_index: int = 0,
    count: int | None = None,
    seed: int = 0,
    xp: ModuleType = np,
    dtype: DTypeLike = 'complex64',
) -> Array:
    """generate a unit-amplitude tone, LO-shifted, with optional additive noise.

    The tone is `tone` at the sum of `frequency_offset` and `lo_offset`, with unit
    amplitude and zero phase at absolute sample index 0. When `snr` is set,
    `circular_awgn` is added at a total power `snr` dB below the tone's mean power
    of 1. Each port gets the same tone, and independent noise.

    Args:
        duration: capture duration, in s, used only when `count` is None
        sample_rate: IQ sample rate, in Hz
        frequency_offset: tone frequency, in Hz relative to the center frequency
        snr: ratio of the tone power to the noise power, in dB, or None for no noise
        lo_offset: additional frequency shift, in Hz, as applied by an LO shift
        ports: number of receive ports
        start_index: absolute index of the first returned sample
        count: number of samples, or None for `duration` less `start_index`
        seed: seed of the noise generator, in ``[0, 2**31)``
        xp: the array namespace to generate with
        dtype: the returned complex dtype

    Returns:
        an array of shape ``(ports, count)`` and dtype `dtype` in the namespace `xp`

    Raises:
        ValueError: when `ports` is below 1, when `duration` and `count` are both
            None, or when the resolved `count` is negative
    """
    kws = {'start_index': start_index, 'count': count, 'xp': xp, 'dtype': dtype}
    x = tone(duration, sample_rate, frequency=frequency_offset, ports=ports, **kws)

    if lo_offset:
        x *= tone(duration, sample_rate, frequency=lo_offset, ports=1, **kws)

    if snr is not None:
        x += circular_awgn(
            duration,
            sample_rate,
            power=10 ** (-snr / 10),
            ports=ports,
            seed=seed,
            **kws,
        )

    return x


def circular_awgn(
    duration: float | None,
    sample_rate: float,
    *,
    power: float = 1.0,
    ports: int = 1,
    start_index: int = 0,
    count: int | None = None,
    seed: int = 0,
    xp: ModuleType = np,
    dtype: DTypeLike = 'complex64',
) -> Array:
    """generate circularly-symmetric complex gaussian noise of total power `power`.

    The real and imaginary parts are independent and each has variance
    ``power / 2``, so the mean of ``|iq|**2`` is `power`. Each port is an
    independent draw. White noise of power spectral density ``noise_psd`` mW/Hz
    spread over the full sample rate is ``power=noise_psd * sample_rate``, which is
    how the sensor's `noise` source calls this.

    The draws are interleaved port-by-port within each sample so that the first
    ``start_index + count`` samples of every port are a prefix of the stream from
    ``RandomState(seed)``, which is what makes the result windowable. Samples at
    negative indices come from a second stream seeded ``2**31`` above `seed`,
    running backward from index -1: they are independent noise of the same power,
    and a window that crosses index 0 is still the corresponding slice of any
    longer window. Both seeds must fit the 32-bit range of ``RandomState``, so
    `seed` must be below ``2**31``.

    Args:
        duration: capture duration, in s, used only when `count` is None
        sample_rate: IQ sample rate, in Hz, unused except to size `duration`
        power: total power of each port, in linear units
        ports: number of receive ports, which get independent noise
        start_index: absolute index of the first returned sample
        count: number of samples, or None for `duration` less `start_index`
        seed: seed of the noise generator, in ``[0, 2**31)``
        xp: the array namespace to generate with
        dtype: the returned complex dtype

    Returns:
        an array of shape ``(ports, count)`` and dtype `dtype` in the namespace `xp`

    Raises:
        ValueError: when `ports` is below 1, when `duration` and `count` are both
            None, when the resolved `count` is negative, or when `start_index` is
            negative and `seed` is outside ``[0, 2**31)``
    """
    count = _resolve_count(duration, sample_rate, start_index, count, ports)
    values = _windowed_normal_draws(
        ports=ports, start_index=start_index, count=count, seed=seed, xp=xp
    )

    # a python float scalar; a numpy scalar would upcast the samples to complex128
    scale = float(np.sqrt(power / 2))
    return (values * scale).astype(dtype, copy=False)


def sawtooth(
    duration: float | None,
    sample_rate: float,
    *,
    period: float = 0.01,
    power: float = 0.0,
    ports: int = 1,
    start_index: int = 0,
    count: int | None = None,
    xp: ModuleType = np,
    dtype: DTypeLike = 'complex64',
) -> Array:
    """generate a real-valued sawtooth ramp that repeats every `period` s.

    Each ramp rises linearly from 0 toward the amplitude whose power is `power` dB
    (1 at 0 dB), resetting at absolute sample index 0 and every `period` s before
    and after it; the peak itself is never reached. The imaginary part is 0. Each
    port gets the same samples.

    Args:
        duration: capture duration, in s, used only when `count` is None
        sample_rate: IQ sample rate, in Hz
        period: duration of one ramp, in s; must be positive
        power: peak amplitude of the ramp expressed as a power, in dB
        ports: number of receive ports
        start_index: absolute index of the first returned sample
        count: number of samples, or None for `duration` less `start_index`
        xp: the array namespace to generate with
        dtype: the returned complex dtype

    Returns:
        an array of shape ``(ports, count)`` and dtype `dtype` in the namespace `xp`

    Raises:
        ValueError: when `period` is not positive, when `ports` is below 1, when
            `duration` and `count` are both None, or when the resolved `count` is
            negative
    """
    if period <= 0:
        raise ValueError(f'period must be positive, not {period}')

    count = _resolve_count(duration, sample_rate, start_index, count, ports)
    t = _sample_indices(start_index, count, xp) / sample_rate
    amplitude = 10 ** (power / 20)
    x = ((t % period) * (amplitude / period)).astype(dtype)
    return _expand_ports(x, ports, xp)


def dirac_delta(
    duration: float | None,
    sample_rate: float,
    *,
    time: float = 0.0,
    power: float = 0.0,
    ports: int = 1,
    start_index: int = 0,
    count: int | None = None,
    xp: ModuleType = np,
    dtype: DTypeLike = 'complex64',
) -> Array:
    """generate a single impulse at `time` in a field of zeros.

    The impulse is real, with the amplitude whose power is `power` dB (1 at 0 dB),
    and sits at the absolute sample index nearest `time` s from the start of the
    capture; the result is all zeros when that index falls outside the returned
    window. Each port gets the same samples.

    Args:
        duration: capture duration, in s, used only when `count` is None
        sample_rate: IQ sample rate, in Hz
        time: time of the impulse, in s from the start of the capture
        power: impulse amplitude expressed as a power, in dB
        ports: number of receive ports
        start_index: absolute index of the first returned sample
        count: number of samples, or None for `duration` less `start_index`
        xp: the array namespace to generate with
        dtype: the returned complex dtype

    Returns:
        an array of shape ``(ports, count)`` and dtype `dtype` in the namespace `xp`

    Raises:
        ValueError: when `ports` is below 1, when `duration` and `count` are both
            None, or when the resolved `count` is negative
    """
    count = _resolve_count(duration, sample_rate, start_index, count, ports)
    x = xp.zeros(count, dtype=dtype)
    index = round(time * sample_rate) - start_index

    if 0 <= index < count:
        x[index] = 10 ** (power / 20)

    return _expand_ports(x, ports, xp)
