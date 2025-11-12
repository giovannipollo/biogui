"""
This module contains the BioGAP interface for sEMG.


Copyright 2023 Mattia Orlandi, Pierangelo Maria Rapa
Modified by Giusy Spacone, 2025

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import struct
from collections import namedtuple

import numpy as np

GAIN = 6


def createCommand():
    """Internal function to create start command."""
    # Byte 0: ADS sampling rate
    # - 6 -> 500sps
    # Byte 1: ADS1298 mode
    # - 0 -> default
    # Byte 2: depends on the number of ADSs
    # Byte 3: chip select (not modifiable)
    command = [6, 0, 2, 4]
    # Byte 4: PGA gain
    # 16 ->  1
    # 32 ->  2
    # 64 ->  4
    #  0 ->  6
    # 80 ->  8
    # 96 -> 12
    gainCmdMap = {
        1: 16,
        2: 32,
        4: 64,
        6: 0,
        8: 80,
        12: 96,
    }
    command.append(gainCmdMap[GAIN])
    # Byte 5: CR (not modifiable)
    command.append(13)
    # Byte 6: LF (not modifiable)
    command.append(10)

    return command


packetSize: int = 234
"""Number of bytes in each package."""

startSeq: list[bytes] = [
    (18).to_bytes(),  # send byte 18 to start
    0.2,  # wait 200 ms
    bytes(createCommand()),
]
"""Sequence of commands to start the board."""

stopSeq: list[bytes] = [(19).to_bytes()]
"""Sequence of commands to stop the board."""

fs: list[float] = [500]
"""Sequence of floats representing the sampling rate of each signal."""

nCh: list[int] = [16]
"""Sequence of integers representing the number of channels of each signal."""


sigInfo: dict = {"biogap": {"fs": 500, "nCh": 16}}
"""Dictionary containing the signals information."""


def decodeFn(data: bytes) -> dict[str, np.ndarray]:
    """
    Function to decode the binary data received from BioGAP into a single sEMG signal.

    Parameters
    ----------
    data : bytes
        A packet of bytes.

    Returns
    -------
    SigsPacket
        Named tuple containing the EMG packet with shape (nSamp, nCh).
    """
    nSamp = 4
    nCh = 16

    # ADC parameters
    vRef = 2.5
    gain = 6.0
    nBit = 24
    # Byte 1: Header; Byte2: Counter;
    dataTmp = bytearray(
        data[2:26]  # ADS A sample1   (8 channels)
        + data[58:82]  # ADS A sample2
        + data[114:138]  # ADS A sample3
        + data[170:194]  # ADS A sample4
        + data[26:50]  # ADSB sample1     (8 channels)
        + data[82:106]  # ADSB sample2
        + data[138:162]  # ADS2 sample3
        + data[194:218]  # ADSB sample4
    )

    # Convert 24-bit to 32-bit integer
    pos = 0
    for _ in range(len(dataTmp) // 3):
        prefix = 255 if dataTmp[pos] > 127 else 0
        dataTmp.insert(pos, prefix)
        pos += 4
    emg = np.asarray(struct.unpack(f">{nSamp *nCh}i", dataTmp), dtype=np.int32)

    # Reshape and convert ADC readings to uV
    emg = emg.reshape(nSamp, nCh)
    emg = emg * (vRef / gain / 2**nBit)  # V
    emg *= 1_000_000  # uV
    emg = emg.astype(np.float32)

    return {"biogap": emg}
