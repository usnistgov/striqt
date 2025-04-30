#!/usr/bin/env python

from flex_spectrum_sensor_scripts import click_capture_plotter


@click_capture_plotter()
def run(dataset, output_path: str, interactive: bool):
    """generic plots"""
    from channel_analysis import figures
    from concurrent import futures
    import multiprocessing
    import iqwaveform  # needed for plt.style.use()
    from matplotlib import pyplot as plt

    plt.style.use('iqwaveform.ieee_double_column')

    plotter = figures.CapturePlotter(
        interactive=interactive,
        output_dir=output_path,
        subplot_by_channel=True,
        col_wrap=2,
        title_fmt='Channel {channel}',
        suptitle_fmt='{center_frequency}',
        filename_fmt='{name} {center_frequency}.svg',
        ignore_missing=True,
    )

    executor = futures.ProcessPoolExecutor(max_workers=6)

    with executor:
        pending = []
        for start_time, data in dataset.groupby('start_time'):
            # 1 start time per (maybe multi-channel) capture

            # channel power representations
            pending += [executor.submit(plotter.channel_power_time_series, data)]
            pending += [executor.submit(plotter.cyclic_channel_power, data)]
            pending += [
                executor.submit(
                    plotter.channel_power_histogram,
                    data,
                    channel_power_bin=slice(-95, -15),
                )
            ]

            # spectrogram representations
            pending += [
                executor.submit(
                    plotter.spectrogram, data, spectrogram_time=slice(0, 20e-3)
                )
            ]
            pending += [
                executor.submit(
                    plotter.spectrogram_histogram,
                    data,
                    spectrogram_power_bin=slice(-130, -50),
                )
            ]
            pending += [executor.submit(plotter.spectrogram_ratio_histogram, data)]
            pending += [
                executor.submit(
                    plotter.power_spectral_density,
                    data,
                    frequency_statistic=['mean', 0.9, 'max'],
                )
            ]

        for future in futures.as_completed(pending):
            try:
                future.result()
            except Exception as exc:
                print(f'generated an exception: {exc}')


if __name__ == '__main__':
    run()
