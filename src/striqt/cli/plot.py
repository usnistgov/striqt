#!/usr/bin/env python

from . import click_capture_plotter


def _submit_if_available(executor, func: callable, data, *args, **kws) -> list:
    if func.__name__ in data.data_vars:
        return [executor.submit(func, data, *args, **kws)]
    else:
        return []


@click_capture_plotter()
def run(dataset, output_path: str, interactive: bool, style: str):
    """generic plots"""
    from striqt import figures
    from concurrent import futures
    import numpy as np

    if 'center_frequency' in dataset.coords and np.isfinite(
        dataset.center_frequency.data[0]
    ):
        suptitle_fmt = '{center_frequency}'
        filename_fmt = '{name} {center_frequency}.svg'
    else:
        suptitle_fmt = '{path}'
        filename_fmt = '{name} {path}.svg'

    plotter = figures.CapturePlotter(
        interactive=interactive,
        output_dir=output_path,
        subplot_by_port=True,
        col_wrap=2,
        title_fmt='Port {port}',
        suptitle_fmt=suptitle_fmt,
        filename_fmt=filename_fmt,
        ignore_missing=True,
        style=f'striqt.figures.{style}',
    )

    executor = futures.ProcessPoolExecutor(max_workers=6)

    ex = None

    with executor:
        pending = []
        if 'start_time' in dataset.coords:
            groups = dataset.groupby('start_time')
        else:
            groups = [(None, dataset)]
        for _, data in groups:
            # 1 start time per (maybe multi-channel) capture

            # channel power representations
            pending += _submit_if_available(
                executor, plotter.spectrogram, data, spectrogram_time=slice(0, 20e-3)
            )

            pending += _submit_if_available(
                executor, plotter.cellular_5g_pss_correlation, data, dB=True
            )
            pending += _submit_if_available(
                executor, plotter.cellular_5g_ssb_spectrogram, data
            )
            pending += _submit_if_available(
                executor, plotter.cellular_cyclic_autocorrelation, data, dB=True
            )
            pending += _submit_if_available(
                executor, plotter.channel_power_time_series, data
            )
            pending += _submit_if_available(
                executor,
                plotter.channel_power_histogram,
                data,
                channel_power_bin=slice(-95, -15),
            )
            pending += _submit_if_available(
                executor, plotter.cyclic_channel_power, data
            )
            pending += _submit_if_available(
                executor,
                plotter.power_spectral_density,
                data,
            )
            pending += _submit_if_available(
                executor,
                plotter.spectrogram_histogram,
                data,
                spectrogram_power_bin=slice(-130, -50),
            )
            pending += _submit_if_available(
                executor, plotter.spectrogram_ratio_histogram, data
            )
        for future in futures.as_completed(pending):
            try:
                future.result()
            except Exception as exc:
                print(f'generated an exception: {exc}')
                if ex is None:
                    ex = exc

    if ex is not None:
        raise ex


if __name__ == '__main__':
    run()
