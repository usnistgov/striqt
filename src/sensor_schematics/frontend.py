import schemdraw
from schemdraw import dsp
from .rx_blocks import RxBlock, SDR, Mixer
from iqwaveform import powtodB, dBtopow
from .schematics import copy_element

def summarize_sensor_performance(frontend: RxBlock, *, sensor: RxBlock, channel_bandwidth, dnr_max_dB: float=-6., ):
    """Summarize the predicted performance of (1) the RX front-end and (2) the front-end connected to the SDR.

    Args:
        frontend: the cascaded front-end
        sensor: the fully connected receive chain in cascade (frontend->SDR)
        dnr_max_dB: max distortion-to-noise ratio of adjacent-channel leakage,
            e.g., dnr_max_dB = -6 is similar to "1 dB increase in noise"
    """
    noise_power = -174 + powtodB(channel_bandwidth) + sensor.NF_dB
    imd3_power_max = noise_power + dnr_max_dB # -6 dB ~= "1 dB increase in noise"
    acp_max = (1/3)*imd3_power_max + 2/3*(frontend.iip3_dBm)

    print('Outside IF BW:')
    print(f'\tIIP3: {frontend.iip3_dBm:0.1f} dBm')
    print(f'\tOIP3: {frontend.oip3_dBm:0.1f} dBm')
    print(f'\tIM3 overload level: {acp_max:0.1f} dBm')
    print(f'\tAdjacent-channel rejection ratio (ACRR) @ overload: {acp_max-imd3_power_max:0.1f} dB')    
    print('Inside IF BW:')
    print(f'\tGain: {frontend.G_dB:0.1f} dB')
    print(f'\tNoise figure: {sensor.NF_dB:0.1f} dB')

def cascade(*blocks: RxBlock) -> RxBlock:
    """return an RxBlock representing the design estimate for the cascaded performance"""
    if len(blocks) == 1:
        return blocks[0]
    elif len(blocks) > 1:
        # recursively cascade 2 blocks at a time
        b0, b1 = blocks[:2]

        # see Pozar, "Microwave Engineering", 3rd. ed., p. 508, eq (10.54)
        iip3_dBm = -powtodB(dBtopow(-b0.iip3_dBm) + dBtopow(b0.G_dB - b1.iip3_dBm))

        # Friis noise equation
        NF_dB = powtodB(dBtopow(b0.NF_dB) + (dBtopow(b1.NF_dB)-1)/dBtopow(b0.G_dB))

        if None in (b0.G_dB, b1.G_dB):
            G_dB = None
        else:
            G_dB = b0.G_dB + b1.G_dB

        b0b1 = RxBlock(G_dB=G_dB, NF_dB=NF_dB, iip3_dBm=iip3_dBm)
        return cascade(b0b1, *blocks[2:])
    elif len(blocks) == 0:
        raise TypeError('must pass at least one block')


def draw(*blocks: RxBlock, sdr_TX_as_LO: bool = True):
    """draw the frontend consisting of the cascaded chain of system blocks"""
    # ref: https://schemdraw.readthedocs.io/en/0.6.0/gallery/gallery.html#superheterodyne-receiver
    with schemdraw.Drawing() as d:
        d.config(fontsize=10)
        L = d.unit/3

        antenna = dsp.Antenna()

        symbols = {}

        for block in blocks:
            dsp.Line(arrow='->').right(L)

            # confusion results if we don't make fresh copies of these objects
            symbol = copy_element(block.symbol)
            symbols.setdefault(type(block), []).append(symbol)

            if block.name is not None:
                symbol = symbol.label(block.name, loc='top')
            if isinstance(block, SDR):
                symbol.anchor('RX1')

            d.add(symbol)

        if sdr_TX_as_LO and SDR in symbols and Mixer in symbols:
            mixer_symbol = symbols[Mixer][0]
            sdr_symbol = symbols[SDR][0]

            dsp.Line(xy=sdr_symbol.TXRX2).tox(mixer_symbol.S)
            dsp.Line(arrow='->').toy(mixer_symbol.S)

    return d

