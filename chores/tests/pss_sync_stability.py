#!/usr/bin/env python3
"""Tabulate the PSS synchronization delay of a sweep store by capture group.

A capture on a `trigger_strobe` grid of a PPS-locked clock sees the PSS of one GPS-synced
cell at one delay in every segment. The table below shows, for each group of rows, the
distinct delay levels and the differences between them against the candidate quanta of
an acquisition timing fault, so a shift can be attributed to a mechanism:

    sample          one source sample at --sdr-rate (8 ns at 125 MS/s)
    tick            the 16 ns AirStack time stamp tick
    multiframe      one JESD204B multiframe, 32 frames at --sdr-rate (256 ns)
    page            one 4096-byte xdma page of complex64 samples at --sdr-rate
    descriptor      2**19 samples at --sdr-rate (4.19 ms)
    symbol          one OFDM symbol with cyclic prefix at 30 kHz subcarrier spacing
    half-frame      5 ms of 3GPP frame structure

Usage:
    pixi run -e test39 python chores/tests/pss_sync_stability.py <store.zarr>
        [--variable cellular_5g_pss_sync] [--group channel_name port burst_index]
        [--resolution 6.5e-8] [--sdr-rate 125e6]

Rows are grouped by the coordinates of --group that the store has. Levels are the
clusters of delay values that are closer than --resolution (one sample of the
measurement `sample_rate`, 15.36 MS/s by default) to a neighbor.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

DEFAULT_GROUP = ('channel_name', 'center_frequency', 'port', 'burst_index')


def quanta(sdr_rate: float) -> dict[str, float]:
    """return the candidate timing quanta (in s) at the source rate `sdr_rate`."""
    return {
        'sample': 1 / sdr_rate,
        'tick': 16e-9,
        'multiframe': 32 / sdr_rate,
        'page': 4096 / 8 / sdr_rate,
        'descriptor': 2**19 / sdr_rate,
        'symbol': 1 / 14e3 / 2,
        'half-frame': 5e-3,
    }


def cluster_levels(
    values: Iterable[float], resolution: float
) -> list[tuple[float, int]]:
    """return ``(mean, count)`` of each cluster of `values` separated by more than
    `resolution` from its neighbors, in ascending order."""
    v = np.sort(np.asarray([x for x in values if math.isfinite(x)], dtype='float64'))
    if v.size == 0:
        return []
    splits = np.flatnonzero(np.diff(v) > resolution) + 1
    return [(float(c.mean()), int(c.size)) for c in np.split(v, splits)]


def nearest_quantum(delta: float, table: dict[str, float]) -> str:
    """return the quantum of `table` with the ratio to `delta` closest to a whole
    number of at most 64, as ``name x n (error %)``."""
    best = ('', math.inf, 0.0)
    for name, q in table.items():
        n = round(delta / q)
        if n < 1 or n > 64:
            continue
        err = abs(delta - n * q) / delta
        if err < best[1]:
            best = (name, err, n)
    if not best[0]:
        return 'none within 64 quanta'
    return f'{best[0]} x {best[2]} ({100 * best[1]:.1f} % off)'


def tabulate(
    frame: pd.DataFrame,
    variable: str,
    group: Sequence[str],
    resolution: float,
    table: dict[str, float],
) -> None:
    keys = [k for k in group if k in frame.columns]
    print(f'{variable}: {len(frame)} rows grouped by {keys}')
    grouped = frame.groupby(keys, dropna=False) if keys else [((), frame)]
    for key, rows in grouped:
        values = key if isinstance(key, tuple) else (key,)
        label = ', '.join(f'{k}={v}' for k, v in zip(keys, values))
        levels = cluster_levels(rows[variable].to_numpy(), resolution)
        if not levels:
            print(f'\n{label}: {len(rows)} rows, no finite value')
            continue
        spread_us = 1e6 * (levels[-1][0] - levels[0][0])
        print(
            f'\n{label}: {len(rows)} rows, {len(levels)} levels, spread {spread_us:.3f} us'
        )
        for mean, count in levels:
            print(f'    {1e6 * mean:12.3f} us  x{count}')
        if len(levels) > 1:
            print('  level differences:')
            for i in range(1, len(levels)):
                delta = levels[i][0] - levels[0][0]
                print(f'    {1e6 * delta:12.3f} us  ~ {nearest_quantum(delta, table)}')


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('store', help='the zarr store of a sweep')
    parser.add_argument('--variable', default='cellular_5g_pss_sync')
    parser.add_argument('--group', nargs='*', default=list(DEFAULT_GROUP))
    parser.add_argument('--resolution', type=float, default=1 / 15.36e6)
    parser.add_argument('--sdr-rate', type=float, default=125e6)
    args = parser.parse_args(argv)

    import striqt.analysis as sa

    ds = sa.load(args.store)
    if args.variable not in ds.data_vars:
        parser.error(
            f'{args.variable!r} is not in the store; its data variables are '
            f'{sorted(map(str, ds.data_vars))}'
        )
    da = ds[args.variable]
    frame = da.to_dataframe().reset_index()
    tabulate(frame, args.variable, args.group, args.resolution, quanta(args.sdr_rate))


if __name__ == '__main__':
    main()
