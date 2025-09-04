#!/usr/bin/env python

from . import click_capture_plotter


def _submit_if_available(executor, func: callable, data, *args, **kws) -> list:
    if func.__name__ in data.data_vars:
        return [executor.submit(func, data, *args, **kws)]
    else:
        return []


@click_capture_plotter()
def run(dataset, output_path: str, interactive: bool):
    """generic plots"""
    from striqt.analysis import figures
    from concurrent import futures
    import iqwaveform  # needed for plt.style.use()
    from matplotlib import pyplot as plt
    import labbench as lb
    lb.util.force_full_traceback(True)

    plt.style.use('iqwaveform.ieee_double_column')

    plotter = figures.CapturePlotter(
        interactive=interactive,
        output_dir=output_path,
        subplot_by_channel=True,
        col_wrap=2,
        title_fmt='Port {port}',
        suptitle_fmt='{center_frequency}',
        filename_fmt='{name} {center_frequency}.svg',
        ignore_missing=True,
    )

    executor = futures.ProcessPoolExecutor(max_workers=6)

    ex = None

    with executor:
        pending = []
        for start_time, data in dataset.groupby('start_time'):
            # 1 start time per (maybe multi-channel) capture

            # channel power representations
            pending += _submit_if_available(
                executor, plotter.channel_power_time_series, data
            )
            pending += _submit_if_available(
                executor, plotter.cyclic_channel_power, data
            )
            pending += _submit_if_available(
                executor,
                plotter.channel_power_histogram,
                data,
                channel_power_bin=slice(-95, -15),
            )

            pending += _submit_if_available(
                executor, plotter.spectrogram, data, spectrogram_time=slice(0, 20e-3)
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
            pending += _submit_if_available(
                executor,
                plotter.power_spectral_density,
                data,
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
