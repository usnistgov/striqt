"""striqt.waveform.lib.comms.cp: the cyclic-prefix correlation against a float64
numpy reference and on a synthetic OFDM waveform, on numpy and cupy.

The cupy cases skip when cupy is not available.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import as_xp, gaussian_iq, iq_waveforms
from hypothesis import given, strategies as st
from numeric_checks import assert_close, numpy_and_cupy, reference_power
from waveform_strategies import corr_atol_for, cp_case

from striqt.waveform.lib.comms import cp


def corr_reference(inds, x, nfft, ncp, norm):
    """return the float64 cyclic-prefix correlation of `cp.correlate_cyclic_shift`.

    Index pairs that run past the end of `x` are dropped.
    """
    xc = np.asarray(x).astype(np.complex128)
    inds = np.asarray(inds)
    out = np.empty(nfft + ncp, dtype=np.complex128)

    for j in range(nfft + ncp):
        ix = inds + j
        valid = ix + nfft < xc.shape[0]
        a = xc[ix[valid]]
        b = xc[ix[valid] + nfft]
        corr = np.sum(a * b.conj())
        if norm:
            corr /= np.sqrt(np.sum(np.abs(a) ** 2) * np.sum(np.abs(b) ** 2))
        else:
            corr /= inds.size
        out[j] = corr

    return out


def corr_cases(
    min_inds=1, max_inds=8, dtype=None, ncp_from_inds=False, sorted_inds=False
):
    """draw (inds, x, nfft, ncp, norm) arguments with arbitrary index sets.

    The waveform length is drawn so that some examples run index pairs past the end of
    `x`. The indices are in arbitrary order unless `sorted_inds`, so an out-of-range
    pair may precede a valid one. With `ncp_from_inds`, `ncp` is the index count, as
    `cp.correlate_cyclic_shift` infers from the last axis of `inds`.
    """

    @st.composite
    def _case(draw):
        nfft = draw(st.sampled_from([32, 64, 128]))
        n_inds = draw(st.integers(min_value=min_inds, max_value=max_inds))
        if ncp_from_inds:
            ncp = n_inds
        else:
            ncp = draw(st.integers(min_value=4, max_value=nfft // 4))
        positions = st.integers(min_value=0, max_value=2 * nfft)
        inds = draw(st.lists(positions, min_size=n_inds, max_size=n_inds, unique=True))
        inds = np.asarray(sorted(inds) if sorted_inds else inds, dtype=np.int64)
        # every lag keeps at least one valid pair (from the smallest index) so the
        # correlation is defined. Larger indexes may still run past the end of x
        min_size = int(inds.min()) + 2 * nfft + ncp
        full_size = int(inds.max()) + 2 * nfft + ncp
        size = draw(st.integers(min_value=min_size, max_value=full_size))
        x = draw(iq_waveforms(min_size=size, max_size=size, dtype=dtype))
        norm = draw(st.booleans())
        return inds, x, nfft, ncp, norm

    return _case()


def runs_cases():
    """draw (inds, x, nfft, ncp, norm) arguments with 3GPP-style index sets.

    `inds` has shape ``(n_symbols, ncp)``, with each row the ``ncp`` consecutive
    samples from a symbol start. The starts are sorted and at least ``nfft + ncp``
    apart. The waveform length is drawn as in `corr_cases`: some examples run the
    pairs of the last symbols past the end of `x`.
    """

    @st.composite
    def _case(draw):
        nfft = draw(st.sampled_from([32, 64, 128]))
        ncp = draw(st.integers(min_value=4, max_value=nfft // 4))
        n_symbols = draw(st.integers(min_value=1, max_value=6))
        first = draw(st.integers(min_value=0, max_value=nfft))
        gaps = st.integers(min_value=nfft + ncp, max_value=2 * nfft)
        steps = draw(st.lists(gaps, min_size=n_symbols - 1, max_size=n_symbols - 1))
        starts = first + np.cumsum([0, *steps], dtype=np.int64)
        inds = starts[:, np.newaxis] + np.arange(ncp, dtype=np.int64)
        min_size = int(starts.min()) + 2 * nfft + ncp
        full_size = int(starts.max()) + 2 * nfft + ncp
        size = draw(st.integers(min_value=min_size, max_value=full_size))
        x = draw(iq_waveforms(min_size=size, max_size=size))
        norm = draw(st.booleans())
        return inds, x, nfft, ncp, norm

    return _case()


def assert_matches_reference(out, inds, x, nfft, ncp, norm):
    expected = corr_reference(inds, x, nfft, ncp, norm)
    assert_close(out, expected, atol=corr_atol_for(x, inds.size, norm))


class TestCorrelateCyclicShiftDispatcher:
    """cp.correlate_cyclic_shift on each array namespace against the numpy reference."""

    @given(case=corr_cases(ncp_from_inds=True, sorted_inds=True))
    def test_sizes_out_from_inds(self, xp, case):
        inds, x, nfft, ncp, norm = case
        inds_2d, x_xp = as_xp(xp, inds[np.newaxis, :], x)

        out = cp.correlate_cyclic_shift(inds_2d, x_xp, nfft, norm=norm)
        assert isinstance(out, xp.ndarray)
        assert out.shape == (nfft + ncp,)
        assert out.dtype == x.dtype
        assert_matches_reference(out, inds, x, nfft, ncp, norm)

        buffer = xp.empty(nfft + ncp, dtype=x.dtype)
        ret = cp.correlate_cyclic_shift(inds_2d, x_xp, nfft, norm=norm, out=buffer)
        assert ret is buffer
        assert_matches_reference(ret, inds, x, nfft, ncp, norm)

    @given(case=corr_cases(ncp_from_inds=True))
    def test_arbitrary_index_order(self, xp, case):
        inds, x, nfft, ncp, norm = case
        inds_xp, x_xp = as_xp(xp, inds[np.newaxis, :], x)
        out = cp.correlate_cyclic_shift(inds_xp, x_xp, nfft, norm=norm)
        assert_matches_reference(out, inds, x, nfft, ncp, norm)

    @given(case=runs_cases())
    def test_consecutive_prefix_runs(self, xp, case):
        inds, x, nfft, ncp, norm = case
        inds_xp, x_xp = as_xp(xp, inds, x)
        out = cp.correlate_cyclic_shift(inds_xp, x_xp, nfft, norm=norm)
        assert out.shape == (nfft + ncp,)
        assert_matches_reference(out, inds, x, nfft, ncp, norm)

    @pytest.mark.parametrize('norm', [False, True])
    def test_valid_index_after_out_of_range(self, xp, norm):
        nfft, ncp = 32, 4
        x = gaussian_iq(3 * nfft, seed=0)
        # the first prefix runs past the end of x at every lag; the second never does
        inds = np.array([2 * nfft, 0], dtype=np.int64)[:, np.newaxis] + np.arange(ncp)
        inds_xp, x_xp = as_xp(xp, inds, x)
        out = cp.correlate_cyclic_shift(inds_xp, x_xp, nfft, norm=norm)
        assert_matches_reference(out, inds, x, nfft, ncp, norm)


class TestCorrelateCyclicShift:
    """cyclic prefix correlation on a synthetic OFDM waveform, indexed as the
    cellular cyclic autocorrelation measurement does"""

    def test_normalized_correlation_peaks_at_zero_lag(self):
        phy, x, inds = cp_case()
        ncp = int(phy.cp_sizes[1])

        R = np.abs(cp.correlate_cyclic_shift(inds, x, phy.nfft, norm=True))
        assert R.shape == (phy.nfft + ncp,)

        # the cyclic prefix is an exact copy of the symbol tail
        assert R[0] == pytest.approx(1, abs=corr_atol_for(x, inds.size, norm=True))

        # partial overlap decays linearly over the prefix length. The remaining
        # pairs are independent QPSK products that average out
        lags = np.arange(ncp)
        assert_close(R[:ncp], 1 - lags / ncp, atol=0.1)
        assert R[ncp : phy.nfft].max() < 0.1

    def test_unnormalized_zero_lag_is_prefix_power(self):
        phy, x, inds = cp_case()

        R = cp.correlate_cyclic_shift(inds, x, phy.nfft, norm=False)
        power = reference_power(x[inds.ravel()]).mean()
        assert R[0] == pytest.approx(power, abs=corr_atol_for(x, inds.size, norm=False))
        assert R.dtype == x.dtype


class TestCupy:
    """the dispatcher on cupy against the numpy result"""

    def test_correlate_cyclic_shift(self, cupy_available):
        phy, x, inds = cp_case(4)
        R_np, R_cp = numpy_and_cupy(
            cupy_available, cp.correlate_cyclic_shift, inds, x, phy.nfft, norm=False
        )
        assert_close(R_cp, R_np, atol=corr_atol_for(x, inds.size, norm=False, n_impl=2))
