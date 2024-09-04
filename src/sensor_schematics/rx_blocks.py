from dataclasses import dataclass
from schemdraw import dsp
from . import schematics


@dataclass
class RxBlock:
    NF_dB: float = None  # noise figure
    G_dB: float = None  # available power gain
    oip3_dBm: float = float('inf')  # output-referred IP3
    iip3_dBm: float = float('inf')  # input-referred IP3
    name: str = None
    symbol = None

    def __post_init__(self):
        # calculate the missing ip3 value from the input or output
        if self.oip3_dBm == float('inf') and self.iip3_dBm == float('inf'):
            return
        elif self.iip3_dBm != float('inf') and self.G_dB is not None:
            self.oip3_dBm = self.iip3_dBm + self.G_dB
        elif self.iip3_dBm != float('inf') and self.G_dB is not None:
            self.iip3_dBm = self.oip3_dBm - self.G_dB


@dataclass
class Amplifier(RxBlock):
    symbol = dsp.Amp(fill='lightblue')


@dataclass
class Mixer(RxBlock):
    symbol = dsp.Mixer(fill='navajowhite')


@dataclass
class PassiveBlock(RxBlock):
    NF_dB = None

    def __post_init__(self):
        super().__post_init__()
        self.NF_dB = -self.G_dB


@dataclass
class Filter(PassiveBlock):
    symbol = dsp.Filter(response='bp', lblofst=0.2, fill='thistle')


@dataclass
class Attenuator(PassiveBlock):
    symbol = schematics.Attenuator(l=0)

    def __post_init__(self):
        super().__post_init__()

        if self.name is None:
            self.name = f'{-self.G_dB} dB'


@dataclass
class VariableAttenuator(PassiveBlock):
    symbol = schematics.AttenuatorVarIEEE(l=0)

    def __post_init__(self):
        super().__post_init__()

        if self.name is None:
            self.name = f'{-self.G_dB} dB'


@dataclass
class SDR(RxBlock):
    symbol = dsp.Ic(
        pins=[
            dsp.IcPin(name='TX2', side='left'),
            dsp.IcPin(name='RX2', side='left'),
            dsp.IcPin(name='TX1', side='left'),
            dsp.IcPin(name='RX1', side='left'),
        ],
        leadlen=0,
        size=(2, 3),
    )


# %% Amplifiers
minicircuits_zx60_83ln_s = Amplifier(
    # https://www.minicircuits.com/pdfs/ZX60-83MP-S+.pdf
    name='Minicircuits\nZX60-83MP-S+',
    G_dB=21.9,
    oip3_dBm=35,
    NF_dB=1.6,
)

minicircuits_zx60_83mp_s = Amplifier(
    # https://www.minicircuits.com/pdfs/ZX60-83LN-S+.pdf
    name='Minicircuits\nZX60-83LN-S+',
    G_dB=20.0,
    oip3_dBm=40.0,
    NF_dB=3.2,
)

minicircuits_pga_103p = Amplifier(
    # https://www.minicircuits.com/pdfs/PGA-103+.pdf
    # estimated @ 3.7 GHz
    name='Minicircuits\nPGA-103+',
    G_dB=7.0,
    oip3_dBm=47.5,
    NF_dB=1.2,
)

analog_hmc994apm5e = Amplifier(
    # https://www.analog.com/media/en/technical-documentation/data-sheets/hmc994apm5e.pdf
    name='Analog\nHMC409LP4',
    G_dB=14,
    oip3_dBm=40.0,
    NF_dB=5,
)

analog_hmc409lp4 = Amplifier(
    # https://www.analog.com/media/en/technical-documentation/data-sheets/hmc409.pdf
    name='Analog\nHMC409LP4',
    G_dB=32,
    oip3_dBm=46.0,
    NF_dB=5.8,
)

microchip_mma053aa = Amplifier(
    # https://ww1.microchip.com/downloads/aemDocuments/documents/RFDS/ProductDocuments/DataSheets/MMA053AA.pdf
    name='Microchip\nMMA053AA',
    G_dB=17.0,
    NF_dB=4.0,
    oip3_dBm=43.0,
)

microchip_maap011327 = Amplifier(
    # this is just the chip - hoping to find an eval board
    # https://cdn.macom.com/datasheets/MAAP-011327.pdf
    name='Microchip\nMAAP011327',
    G_dB=13.0,
    NF_dB=4.0,
    oip3_dBm=45.0,
)

marki_adm_8350psm = Amplifier(
    # https://markimicrowave.com/products/surface-mount/amplifiers/adm-8350psm/datasheet/
    # @ 3.5 GHz
    name='Marki\nADM 8350PSM',
    G_dB=23.0,
    NF_dB=2.0,
    oip3_dBm=38.0,
)

# %% Filters
anatech_ab2619b1221 = Filter(name='Anatech\nAB2619B1221\n2614-2624\nMHz', G_dB=-3.0)

minicircuits_bpf_v1000 = Filter(
    # a potential LO filter (ceramic SMT package)
    name='930-1080\nMHz',
    G_dB=-3.0,
)

minicircuits_vlf_800 = Filter(
    # https://www.minicircuits.com/pdfs/VLF-800+.pdf
    # LO filter that we have on hand for prototyping
    name='0-1080\nMHz',
    G_dB=-1.5,
)

reactel_8C7_3610_x180s11 = Filter(
    # as seen in the SEA project
    name='Reactel\n8C7_3610_x180s11\n3550-3700\nMHz',
    G_dB=-1.0,
)

minicircuits_vbfz_3590 = Filter(
    name='Minicircuits\nVBFZ-3590\n3000-4300\nMHz', G_dB=-2.4
)

# %% Mixers
# For these:
#    G_dB == -(conversion loss in dB)
#    NF_dB == (conversion loss in dB)
marki_t3a_07pa = Mixer(name='Marki\nT3A-07PA', G_dB=-6.5, NF_dB=6.5, iip3_dBm=32.0)

# %% SDRs
ettus_e313 = SDR(name='SDR\nUSRP E313', NF_dB=8.0)

deepwave_air7201b = SDR(name='SDR\nDeepwave Air7201B', NF_dB=18.0)
