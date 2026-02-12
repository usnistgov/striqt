#!/usr/bin/env python

from . import click_capture_plotter
import typing


def try_submit(executor, func: typing.Callable, plotter, data, *args, **kws) -> list:
    if func.__name__ in data.data_vars:
        return [executor.submit(func, plotter, data, *args, **kws)]
    else:
        return []


@click_capture_plotter()
def run(dataset, output_path: str, interactive: bool, style: str):
    """generic plots"""

    from striqt import figures as sf
    from striqt import sensor as ss
    from concurrent import futures
    import numpy as np
    import os
    from pathlib import Path

    if 'center_frequency' in dataset.coords and np.isfinite(
        dataset.center_frequency.data[0]
    ):
        suptitle_fmt = '{center_frequency}'
        filename_fmt = '{name} {center_frequency}.svg'
    else:
        suptitle_fmt = '{path}'
        filename_fmt = '{name} {path}.svg'

    plotter = sf.CapturePlotter(
        unstack=['sweep_start_time', 'start_time', 'port'],
        interactive=interactive,
        output_dir=Path(output_path),
        col='port',
        col_wrap=2,
        col_label_format='{antenna_name}',
        suptitle_fmt=suptitle_fmt,
        filename_fmt=filename_fmt,
        ignore_missing=True,
        style=f'striqt.figures.{style}',
    )

    executor = futures.ProcessPoolExecutor(
        max_workers=os.process_cpu_count(), initializer=plotter.process_setup
    )
    exc = ss.util.ExceptionStack('plots')

    # with executor, exc:
    pending = []
    if 'start_time' in dataset.coords:
        groups = dataset.groupby('start_time')
    else:
        groups = [(None, dataset)]

    for _, data in groups:
        pending += try_submit(
            executor,
            sf.analysis.spectrogram,
            plotter,
            data,
            spectrogram_time=slice(0, 20e-3),
        )
        pending += try_submit(
            executor,
            sf.analysis.cellular_5g_pss_correlation,
            plotter,
            data,
            dB=True,
        )
        pending += try_submit(
            executor, sf.analysis.cellular_5g_ssb_spectrogram, plotter, data
        )
        pending += try_submit(
            executor,
            sf.analysis.cellular_cyclic_autocorrelation,
            plotter,
            data,
            dB=True,
        )
        pending += try_submit(
            executor,
            sf.analysis.cellular_resource_power_histogram,
            plotter,
            data,
            yscale='log',
        )
        pending += try_submit(
            executor,
            sf.analysis.channel_power_histogram,
            plotter,
            data,
            channel_power_bin=slice(-95, -15),
        )
        pending += try_submit(
            executor, sf.analysis.channel_power_time_series, plotter, data
        )
        pending += try_submit(executor, sf.analysis.cyclic_channel_power, plotter, data)
        pending += try_submit(
            executor,
            sf.analysis.power_spectral_density,
            plotter,
            data,
        )
        pending += try_submit(
            executor,
            sf.analysis.spectrogram_histogram,
            plotter,
            data,
            yscale='log',
            spectrogram_power_bin=slice(-130, -50),
        )
        # pending += try_submit(
        #     executor, sf.analysis.spectrogram_ratio_histogram, plotter, data
        # )

    with exc:
        for future in futures.as_completed(pending):
            with exc.defer():
                future.result()


if __name__ == '__main__':
    run()  # type: ignore
