"""
Combined interface for BioGAP Ultra sEMG and microphone streaming.

This interface handles both sEMG and microphone packets from a single data source.
Packets are distinguished by their header byte:
- 0xAA: Microphone packet
- Other: sEMG packet

Each packet type is decoded and returned immediately. The other signal returns
an empty array for that packet. This allows each signal to update at its natural
rate without artificial synchronization.

Copyright 2025 ETH Zurich and University of Bologna

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

import numpy as np

# ============== Microphone Configuration ==============
MIC_SAMPLE_RATE = 16000  # Hz
MIC_SAMPLES_PER_PACKET = 115  # 16-bit samples per BLE packet
MIC_HEADER = 0xAA
MIC_TRAILER = 0x55

# ============== sEMG Configuration ==============
EMG_SAMPLE_RATE = 500  # Hz
EMG_N_CHANNELS = 16
EMG_N_SAMPLES_PER_PACKET = 4
EMG_GAIN = 6


def createCommand():
    """Internal function to create start command for sEMG."""
    command = [6, 0, 2, 4]
    gainCmdMap = {1: 16, 2: 32, 4: 64, 6: 0, 8: 80, 12: 96}
    command.append(gainCmdMap[EMG_GAIN])
    command.append(13)  # CR
    command.append(10)  # LF
    return command


# ============  == Interface Definition ==============
packetSize: int = 234
"""Number of bytes in each package."""

startSeq: list[bytes | float] = [
    (18).to_bytes(),  # Start sEMG streaming
    0.2,  # Wait 200 ms
    bytes(createCommand()),
    0.2,  # Wait 200 ms
    (26).to_bytes(),  # Start microphone streaming
]
"""Sequence of commands to start both sEMG and microphone streaming."""

stopSeq: list[bytes | float] = [
    (19).to_bytes(),  # Stop sEMG streaming
    0.2,  # Wait 200 ms
    (27).to_bytes(),  # Stop microphone streaming
]
"""Sequence of commands to stop both streams."""

fs: list[float] = [EMG_SAMPLE_RATE, MIC_SAMPLE_RATE]
"""Sequence of floats representing the sampling rate of each signal."""

nCh: list[int] = [EMG_N_CHANNELS, 1]
"""Sequence of integers representing the number of channels of each signal."""

sigInfo: dict = {
    "biogap": {"fs": EMG_SAMPLE_RATE, "nCh": EMG_N_CHANNELS},
    "mic": {"fs": MIC_SAMPLE_RATE, "nCh": 1},
}
"""Dictionary containing the signals information."""


def _decode_mic(data: bytes) -> np.ndarray:
    """Decode microphone packet."""
    audio_data = data[2 : 2 + MIC_SAMPLES_PER_PACKET * 2]
    audio = np.array(
        struct.unpack(f"<{MIC_SAMPLES_PER_PACKET}h", audio_data), dtype=np.int16
    )
    audio = audio.reshape(-1, 1)
    audio = audio.astype(np.float32) / 32768.0
    return audio


def _decode_emg(data: bytes) -> np.ndarray:
    """Decode sEMG packet."""
    nSamp = 4
    nCh = 16
    vRef = 2.5
    gain = 6.0
    nBit = 24

    dataTmp = bytearray(
        data[2:26]
        + data[58:82]
        + data[114:138]
        + data[170:194]
        + data[26:50]
        + data[82:106]
        + data[138:162]
        + data[194:218]
    )

    # Convert 24-bit to 32-bit integer
    pos = 0
    for _ in range(len(dataTmp) // 3):
        prefix = 255 if dataTmp[pos] > 127 else 0
        dataTmp.insert(pos, prefix)
        pos += 4
    emg = np.asarray(struct.unpack(f">{nSamp * nCh}i", dataTmp), dtype=np.int32)

    emg = emg.reshape(nSamp, nCh)
    emg = emg * (vRef / gain / 2**nBit)  # V
    emg *= 1_000_000  # uV
    emg = emg.astype(np.float32)
    return emg


def decodeFn(data: bytes) -> dict[str, np.ndarray]:
    """
    Function to decode binary data received from BioGAP.

    Distinguishes between microphone and sEMG packets based on the header byte.
    Returns the decoded signal immediately, with an empty array for the other
    signal type.

    Parameters
    ----------
    data : bytes
        A packet of 234 bytes.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing the decoded signals:
        - For mic packets: {"biogap": empty, "mic": audio_data}
        - For sEMG packets: {"biogap": emg_data, "mic": empty}
    """
    if len(data) != packetSize:
        raise ValueError(f"Invalid packet size: {len(data)}, expected {packetSize}")

    header = data[0]

    if header == MIC_HEADER:
        # This is a microphone packet
        trailer = data[-1]
        if trailer != MIC_TRAILER:
            raise ValueError(
                f"Invalid mic trailer: 0x{trailer:02X}, expected 0x{MIC_TRAILER:02X}"
            )
        audio = _decode_mic(data)
        emg = np.zeros((0, EMG_N_CHANNELS), dtype=np.float32)
        return {"biogap": emg, "mic": audio}
    else:
        # This is an sEMG packet
        emg = _decode_emg(data)
        audio = np.zeros((0, 1), dtype=np.float32)
        return {"biogap": emg, "mic": audio}
