from edge_sensor.radios import Air7201B
import edge_sensor
import labbench as lb

lb.show_messages('debug')

radio = Air7201B()

radio_setup = edge_sensor.RadioSetup(
    time_source='internal',
    warmup_sweep=False,
    array_backend='numpy'
)

capture = edge_sensor.RadioCapture(
    center_frequency=3750e6,
    duration=50e-3,
    sample_rate=125e6,
    host_resample=False,
    channel=(0,1),
    gain=-10
)

with radio:
    radio.setup(radio_setup)
    radio.arm(capture)
    iq, _ = radio.acquire(capture)

# iq is a 2-D numpy array with dimensions (channel index, IQ sample index).
# can save or transfer data here