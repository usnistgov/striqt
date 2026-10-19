"""Shared builders for the comms tests: the 3GPP PHY at the analysis sample rate,
cyclic-prefix cases on a synthetic OFDM waveform, and silent synchronization blocks
carrying embedded PSS symbols.
"""

from __future__ import annotations

import numpy as np

from striqt.waveform.lib import comms

# the analysis default synchronization block sample rate (15.36e6 / 2)
FS = 7.68e6
SCS_5G = (15e3, 30e3, 60e3)


def corr_atol_for(x, n_inds, norm, n_impl=1):
    """comms.corr_atol with the dtype and scale taken from the waveform `x`"""
    scale = float(np.abs(x).max() ** 2)
    return comms.corr_atol(x.dtype, n_inds, norm, scale=scale, n_impl=n_impl)


def khz_id(scs):
    return f'{scs / 1e3:g}kHz'


def msps_id(fs):
    return f'{fs / 1e6:g}MSps'


def phy_5g(scs, fs=FS, xp=None):
    return comms.get_3gpp_phy(
        1, subcarrier_spacing=scs, sample_rate=fs, generation='5G', xp=xp
    )


def sync_params(scs=30e3, fs=FS, **kws):
    """the parameters of the sensor sweep configurations: 30 kHz shared spectrum"""
    kws.setdefault('shared_spectrum', round(scs) == 30_000)
    return comms.pss_params(sample_rate=fs, subcarrier_spacing=scs, **kws)


def ofdm_slots(phy, n_slots, seed=0, dtype=np.complex64):
    """`n_slots` contiguous slots of unit-power QPSK OFDM symbols with cyclic prefixes"""
    rng = np.random.default_rng(seed)
    nfft = int(phy.nfft)
    qpsk = np.array([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]) / np.sqrt(2)

    parts = []
    for _ in range(n_slots):
        for cp in phy.cp_sizes:
            body = np.fft.ifft(rng.choice(qpsk, size=nfft)) * np.sqrt(nfft)
            parts += [body[-int(cp) :], body]
    return np.concatenate(parts).astype(dtype)


def cp_case(n_slots=None):
    """(phy, waveform, cyclic prefix indexes) at 30 kHz over `n_slots` slots, or over
    every slot of a frame when `n_slots` is None"""
    phy = phy_5g(30e3)
    slots = 'all' if n_slots is None else tuple(range(n_slots))
    n_slots = n_slots or phy.SCS_TO_SLOTS_PER_FRAME[30e3]
    return phy, ofdm_slots(phy, n_slots), phy.index_cyclic_prefix(slots=slots)


def embed_symbol(iq, phy, body, symbol_index, delay, port=0, scale=1.0):
    """write an OFDM symbol (with its cyclic prefix) into `iq` at the nominal position
    of `symbol_index` in the frame structure, offset by `delay` samples"""
    slot, symbol = divmod(symbol_index, phy.FFT_PER_SLOT)
    start = slot * phy.contiguous_size + phy.cp_start_idx[symbol] + delay
    cp = int(phy.cp_sizes[symbol])
    wave = scale * np.concatenate([body[-cp:], body])
    iq[port, start : start + wave.size] = wave
    return start


def pss_setup(fs=FS):
    """(params, phy, pss, bodies) for 30 kHz PSS detection at `fs`, where `bodies`
    are the PSS symbols without their cyclic prefix"""
    params = sync_params(fs=fs)
    phy = phy_5g(30e3, fs)
    pss = comms.pss_5g_nr(fs, 30e3)
    return params, phy, pss, pss[:, int(phy.cp_sizes.min()) :]


def silent_block(params, n_blocks=1, ports=1):
    size = params.frame_size * params.frames_per_sync * n_blocks
    return np.zeros((ports, size), dtype=np.complex64)


def embed_pss_ports(embeds, scales=None, n_blocks=1):
    """(params, pss, iq) for a silent synchronization block carrying one PSS symbol
    per port, each given as an (n_id2, beam, delay) triple"""
    params, phy, pss, bodies = pss_setup()
    scales = scales or [1.0] * len(embeds)

    iq = silent_block(params, n_blocks=n_blocks, ports=len(embeds))
    for port, ((n_id2, beam, delay), scale) in enumerate(zip(embeds, scales)):
        symbol = params.symbol_indexes[beam]
        embed_symbol(iq, phy, bodies[n_id2], symbol, delay, port=port, scale=scale)
    return params, pss, iq


def get_ssb_iq(iq, fs_in=FS, **kws):
    """get_5g_ssb_iq with the 30 kHz, 20 ms discovery defaults of the measurements"""
    kws = {
        'discovery_periodicity': 20e-3,
        'fs_out': FS,
        'fs_in': fs_in,
        'subcarrier_spacing': 30e3,
        **kws,
    }
    return comms.get_5g_ssb_iq(iq, **kws)
