"""
This module contains the BioGAP interface for combined EEG and microphone streaming.

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

# EEG configuration
SAMPLE_RATE_EEG = 500  # Hz
SAMPLES_PER_PACKET_EEG = 4  # samples per packet
EEG_N_CHANNELS = 16

# Microphone configuration
SAMPLE_RATE_MIC = 16000  # Hz
SAMPLE_BIT_WIDTH = 16  # bits
SAMPLES_PER_PACKET_MIC = 64  # 16-bit samples per BLE packet

# Packet format constants
EEG_HEADER = 0x55
EEG_TRAILER = 0xAA
EEG_PACKET_SIZE = 210

MIC_HEADER = 0xAA
MIC_TRAILER = 0x55
MIC_PACKET_SIZE = 131


packetSize = [(EEG_HEADER, EEG_PACKET_SIZE), (MIC_HEADER, MIC_PACKET_SIZE)]
"""List of (header_byte, packet_size) tuples for EEG and MIC packets."""

startSeq: list[bytes] = [
    (18).to_bytes(),  # START_EEG_STREAMING command
    0.2,  # Wait 200 ms
    (26).to_bytes(),  # START_MIC_STREAMING command
]
"""Sequence of commands to start EEG and microphone streaming."""

stopSeq: list[bytes] = [
    (19).to_bytes(),  # STOP_EEG_STREAMING command
    0.2,  # Wait 200 ms
    (27).to_bytes(),  # STOP_MIC_STREAMING command
]
"""Sequence of commands to stop EEG and microphone streaming."""

fs: list[float] = [SAMPLE_RATE_EEG, SAMPLE_RATE_MIC]
"""Sequence of floats representing the sampling rate of each signal."""

nCh: list[int] = [EEG_N_CHANNELS, 1]
"""Sequence of integers representing the number of channels of each signal."""

sigInfo: dict = {
    "eeg": {"fs": SAMPLE_RATE_EEG, "nCh": EEG_N_CHANNELS},
    "mic": {"fs": SAMPLE_RATE_MIC, "nCh": 1}
}
"""Dictionary containing the signals information."""


def _decode_eeg(data: bytes) -> np.ndarray:
    """Decode EEG packet.
    Packet structure (210 bytes total):
    - 1 byte: Header (0x55)
    - 1 byte: Packet counter
    - 4 bytes: Timestamp (microseconds, for cross-packet synchronization)
    - 200 bytes: 4 samples × 50 bytes per sample
      - 24 bytes: ADS1298_A data (8 channels × 3 bytes)
      - 24 bytes: ADS1298_B data (8 channels × 3 bytes)
      - 1 byte: Counter_extra
      - 1 byte: Trigger
    - 3 bytes: Metadata (reserved for future use)
    - 1 byte: Trailer (0xAA)
    """
    nSamp = 4
    nCh = 16
    nChSingleADS = 8
    vRef = 2.4
    gain = 6.0
    nBit = 24

    counter = bytearray(data[1:2])

    # Cast the counter to np.int32
    counter = np.asarray(struct.unpack(">B", counter), dtype=np.int32)
    
    dataADSATmp = bytearray(
        data[6:30] + data[56:80] + data[106:130] + data[156:180]
    )
    dataADSBTmp = bytearray(
       data[30:54] + data[80:104] + data[130:154] + data[180:204]
    )

    pos = 0
    for _ in range(len(dataADSATmp) // 3):
        prefix = 255 if dataADSATmp[pos] > 127 else 0
        dataADSATmp.insert(pos, prefix)
        pos += 4
    emgADSA = np.asarray(struct.unpack(f">{nSamp *nChSingleADS}i", dataADSATmp), dtype=np.int32)
    emgADSA = emgADSA.reshape(nSamp, nChSingleADS)
    pos = 0
    for _ in range(len(dataADSBTmp) // 3):
        prefix = 255 if dataADSBTmp[pos] > 127 else 0
        dataADSBTmp.insert(pos, prefix)
        pos += 4
    emgADSB = np.asarray(struct.unpack(f">{nSamp *nChSingleADS}i", dataADSBTmp), dtype=np.int32)
    emgADSB = emgADSB.reshape(nSamp, nChSingleADS)
    emgAllChannels = np.concatenate((emgADSA, emgADSB), axis=1)  # (nSamp, 16)

    counter = counter.reshape(1, 1)
    emg = emgAllChannels * (vRef / (gain * (2 ** (nBit - 1) - 1)))
    emg *= 10e6  # uV
    emg = emg.astype(np.float32)
    return emg


def _decode_mic(data: bytes) -> np.ndarray:
    """Decode microphone packet.
    Packet structure (131 bytes total):
    - 1 byte header (0xAA)
    - 1 byte counter
    - 64 samples of 16-bit signed audio data (128 bytes)
    - 1 byte trailer (0x55)
    """
    audio_data = data[2:2 + SAMPLES_PER_PACKET_MIC * 2]

    # Unpack 16-bit signed samples (little-endian)
    audio = np.array(
        struct.unpack(f"<{SAMPLES_PER_PACKET_MIC}h", audio_data),
        dtype=np.int16
    )

    # Reshape to (nSamp, nCh) format
    audio = audio.reshape(-1, 1)

    # Convert to float32 normalized to [-1.0, 1.0] range
    audio = audio.astype(np.float32) / 32768.0

    return audio


def decodeFn(data: bytes) -> dict[str, np.ndarray]:
    """
    Function to decode binary data received from BioGAP.

    Distinguishes between microphone and EEG packets based on the header byte
    and packet size. Returns the decoded signal immediately, with an empty array
    for the other signal type.

    Parameters
    ----------
    data : bytes
        A packet of either 131 bytes (MIC) or 210 bytes (EEG).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing the decoded signals:
        - For mic packets: {"eeg": empty, "mic": audio_data}
        - For EEG packets: {"eeg": eeg_data, "mic": empty}
    """
    packet_len = len(data)
    header = data[0]
    if packet_len == MIC_PACKET_SIZE and header == MIC_HEADER:
        # This is a microphone packet
        trailer = data[-1]
        if trailer != MIC_TRAILER:
            raise ValueError(f"Invalid mic trailer: 0x{trailer:02X}, expected 0x{MIC_TRAILER:02X}")
        audio = _decode_mic(data)
        return {"mic": audio, "eeg": None}
    elif packet_len == EEG_PACKET_SIZE and header == EEG_HEADER:
        # This is an EEG packet
        trailer = data[-1]
        if trailer != EEG_TRAILER:
            raise ValueError(f"Invalid EEG trailer: 0x{trailer:02X}, expected 0x{EEG_TRAILER:02X}")
        eeg = _decode_eeg(data)
        return {"eeg": eeg, "mic": None}
    else:
        raise ValueError(f"Invalid packet: size={packet_len}, header=0x{header:02X}")