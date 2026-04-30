from __future__ import annotations as __
from typing import cast, Sequence
import striqt.analysis as sa
from . import datasets
from ... import specs
from ..typing import SC, SS, SP


def log_info(dd: datasets.DelayedDataset):
    # log overload messages
    overload_msgs = []
    adc_ol_info = _adc_overload_message(dd.extra_data, dd.capture)
    if adc_ol_info is not None:
        overload_msgs.append(adc_ol_info)
    if_ol_info = _if_overload_message(dd.extra_data, dd.capture, dd.config.sweep_spec)
    if if_ol_info is not None:
        overload_msgs.append(if_ol_info)
    if len(overload_msgs) > 0:
        sa.util.get_logger('analysis').warning(', '.join(overload_msgs))


def _adc_overload_message(
    extra_data: dict[str, Sequence[float]], capture: specs.SensorCapture
) -> str | None:
    if 'adc_headroom' in extra_data and isinstance(capture, specs.SoapyCapture):
        headroom = extra_data['adc_headroom']
        caps = specs.helpers.split_capture_ports(capture)
    else:
        return None

    overload_ports = []
    for c, hr in zip(caps, headroom):
        if hr > 0:
            continue
        else:
            assert not isinstance(c.center_frequency, tuple)
        msg = f'port {c.port} ({c.center_frequency / 1e6:0.0f} MHz)'
        overload_ports.append(msg)

    if len(overload_ports) > 0:
        return 'adc overload on ' + ', '.join(overload_ports)
    else:
        return None


def _if_overload_message(
    extra_data: dict[str, Sequence[float]],
    capture: SC,
    sweep_spec: specs.Sweep[SS, SP, SC],
) -> str | None:
    if 'if_headroom' in extra_data:
        if_headroom = extra_data['if_headroom']
    else:
        return None

    if not isinstance(capture, specs.SoapyCapture):
        return None
    else:
        captures = cast(tuple[specs.SoapyCapture, ...], sweep_spec.captures)

    gains = specs.helpers.max_by_frequency('gain', captures, sweep_spec.loops)
    caps = specs.helpers.split_capture_ports(capture)

    ol_cases = {}
    for c, hr in zip(caps, if_headroom):
        # estimate IM3 levels in other channels
        ol_cases.setdefault(c.port, set())

        for fc, gain in gains[c.port].items():
            im3_headroom = hr + (2 / 3 * (c.gain - gain))

            if im3_headroom > 0:
                continue

            ol_cases[c.port].add(fc)

    ol_labels = []
    for port, freqs in ol_cases.items():
        if len(freqs) > 0:
            freqs_MHz = ', '.join([f'{f / 1e6:0.0f}' for f in sorted(freqs)])
            ol_labels.append(f'port {port} (onto {freqs_MHz} MHz)')

    if len(ol_labels) > 0:
        return 'if overload at ' + ' and '.join(ol_labels)
    else:
        return None
