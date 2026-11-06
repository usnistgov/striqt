"""Power scale conversions and time-binned power detectors, with roundoff models."""

from .convert import (
    dB_tolerance,
    dBlinmean,
    dBlinsum,
    dBtopow,
    envelope_power_rtol,
    envtodB,
    envtopow,
    level_tolerance_dB,
    linear_stat_tol,
    linear_tolerance_dB,
    log_conversion_tol,
    off_peak_dB_tolerance,
    pow_conversion_rtol,
    powtodB,
    roundtrip_dB_tol,
    roundtrip_power_rtol,
)
from .detect import (
    bin_power_rms,
    bin_power_rtol,
    iq_to_bin_power,
    iq_to_cyclic_power,
    sample_ccdf,
    stat_rtol,
    stat_ufunc_from_shorthand,
)
