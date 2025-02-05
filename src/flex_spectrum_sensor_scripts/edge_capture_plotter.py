#!/usr/bin/env python

from flex_spectrum_sensor_scripts import click_capture_plotter


@click_capture_plotter()
def run(dataset, output_path: str, interactive: bool):
    """generic plots"""
    from channel_analysis import figures

    plotter = figures.CapturePlotter(
        interactive=interactive,
        output_dir=output_path,
        subplot_by_channel = True,
        col_wrap=2,
        title_fmt='Channel {channel}',
        suptitle_fmt='{center_frequency}',
        filename_fmt='{name} {center_frequency}.svg',
    )

    for start_time, data in dataset.groupby('start_time'):
        # 1 start time per (maybe multi-channel) capture

        # channel power representations
        plotter.channel_power_time_series(data)
        plotter.cyclic_channel_power(data) 
        plotter.channel_power_histogram(data)

        # spectrogram representations
        plotter.spectrogram(data)
        plotter.spectrogram_histogram(data)
        plotter.spectrogram_ratio_histogram(data)
        plotter.power_spectral_density(data)


if __name__ == "__main__":
    run()
