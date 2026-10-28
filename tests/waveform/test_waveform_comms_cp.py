"""striqt.waveform.lib.comms.cp and the numba kernels in
striqt.waveform.lib.comms.cp_cpu_jit and striqt.waveform.lib.comms.cp_cuda_jit: the
cyclic-prefix correlation against a numpy reference and on a synthetic OFDM waveform.

The CUDA tests skip when cupy is not available. The `cupy.fuse` dB kernels are
defined in `lib/power/cuda_jit.py` and are covered through the public dispatcher in
test_waveform_power_convert.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import gaussian_iq, iq_waveforms
from hypothesis import given
from hypothesis import strategies as st
from numeric_checks import assert_close, numpy_and_cupy, reference_power
from waveform_strategies import corr_atol_for, cp_case

from striqt.waveform.lib.comms import cp

# thread block sizing from cp.correlate_cyclic_shift
THREADS_PER_BLOCK = 32


def corr_reference(inds, x, nfft, ncp, norm):
    """return the float64 cyclic-prefix correlation that `_corr_at_indices` computes.

    Index pairs that run past the end of `x` are dropped, as the zero-fill of the CPU
    kernel does for any index order.
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
            corr /= inds.shape[0]
        out[j] = corr

    return out


def corr_cases(
    min_inds=1, max_inds=8, dtype=None, ncp_from_inds=False, sorted_inds=False
):
    """draw (inds, x, nfft, ncp, norm) kernel arguments.

    The waveform length is drawn so that some examples run index pairs past the end of
    `x` and reach the out-of-range branches of the kernels. The indices are in
    arbitrary order unless `sorted_inds`, so an out-of-range pair may precede a valid
    one. With `ncp_from_inds`, `ncp` is the index count, as `cp.correlate_cyclic_shift`
    infers from the last axis of `inds`.
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


def assert_matches_reference(out, inds, x, nfft, ncp, norm):
    expected = corr_reference(inds, x, nfft, ncp, norm)
    assert_close(out, expected, atol=corr_atol_for(x, inds.size, norm))


class TestCorrAtIndicesKernels:
    """the numba kernels called directly, against the numpy reference"""

    @given(case=corr_cases())
    def test_cpu(self, case):
        from striqt.waveform.lib.comms.cp_cpu_jit import _corr_at_indices

        inds, x, nfft, ncp, norm = case
        out = np.empty(nfft + ncp, dtype=x.dtype)
        _corr_at_indices(inds, x, nfft, ncp, norm, out)
        assert_matches_reference(out, inds, x, nfft, ncp, norm)

    @staticmethod
    def _run_cuda(cp, inds, x, nfft, ncp, norm):
        from striqt.waveform.lib.comms.cp_cuda_jit import _corr_at_indices

        x_cp = cp.asarray(x)
        out = cp.empty(nfft + ncp, dtype=x.dtype)
        bpg = max((x_cp.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK, 1)
        _corr_at_indices[bpg, THREADS_PER_BLOCK](
            cp.asarray(inds), x_cp, nfft, ncp, norm, out
        )
        cp.cuda.Device().synchronize()
        return out

    # the CUDA kernel stops at the first out-of-range pair, which only matches the
    # reference for sorted indices. test_cuda_unsorted_indices covers the rest
    @given(case=corr_cases(sorted_inds=True))
    def test_cuda(self, cupy_available, case):
        inds, x, nfft, ncp, norm = case
        out = self._run_cuda(cupy_available, inds, x, nfft, ncp, norm)
        assert_matches_reference(out, inds, x, nfft, ncp, norm)

    @pytest.mark.xfail(
        strict=True,
        reason='the CUDA _corr_at_indices breaks out of the index loop at the first '
        'out-of-range pair (comms/cp_cuda_jit.py:28-29) instead of skipping it like the CPU '
        'kernel, so a valid index after an out-of-range one is dropped',
    )
    def test_cuda_unsorted_indices(self, cupy_available):
        nfft, ncp = 32, 4
        x = gaussian_iq(3 * nfft, seed=0)
        # the first index runs past the end of x at every lag; the second never does
        inds = np.array([2 * nfft, 0], dtype=np.int64)
        out = self._run_cuda(cupy_available, inds, x, nfft, ncp, False)
        assert_matches_reference(out, inds, x, nfft, ncp, False)


class TestCorrelateCyclicShiftDispatcher:
    """cp.correlate_cyclic_shift selects the kernel for the array namespace."""

    @given(case=corr_cases(ncp_from_inds=True, sorted_inds=True))
    def test_sizes_out_from_inds(self, xp, case):
        inds, x, nfft, ncp, norm = case
        inds_2d = xp.asarray(inds[np.newaxis, :])
        x_xp = xp.asarray(x)

        out = cp.correlate_cyclic_shift(inds_2d, x_xp, nfft, norm=norm)
        assert isinstance(out, xp.ndarray)
        assert out.shape == (nfft + ncp,)
        assert out.dtype == x.dtype
        assert_matches_reference(out, inds, x, nfft, ncp, norm)

        buffer = xp.empty(nfft + ncp, dtype=x.dtype)
        ret = cp.correlate_cyclic_shift(inds_2d, x_xp, nfft, norm=norm, out=buffer)
        assert ret is buffer
        assert_matches_reference(ret, inds, x, nfft, ncp, norm)


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
