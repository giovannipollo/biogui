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
MIC_SAMPLES_PER_PACKET = 64  # 16-bit samples per BLE packet
MIC_HEADER = 0xAA
MIC_TRAILER = 0x55
MIC_PACKET_SIZE = 131  # 1 header + 1 counter + 64*2 audio + 1 trailer

# ============== sEMG Configuration ==============
EMG_SAMPLE_RATE = 500  # Hz
EMG_N_CHANNELS = 16
EMG_N_SAMPLES_PER_PACKET = 4
EMG_GAIN = 6
EMG_HEADER = 0x55
EMG_TRAILER = 0xAA
EMG_PACKET_SIZE = 234


def createCommand():
    """Internal function to create start command for sEMG."""
    command = [6, 0, 2, 4]
    gainCmdMap = {1: 16, 2: 32, 4: 64, 6: 0, 8: 80, 12: 96}
    command.append(gainCmdMap[EMG_GAIN])
    command.append(13)  # CR
    command.append(10)  # LF
    return command


# ============  == Interface Definition ==============
def _packetSizeFn(buffer: bytes) -> tuple[int, int] | None:
    """
    Determine packet size and offset from buffer.

    This function scans the buffer to find valid packet boundaries.
    It returns a tuple of (packet_size, offset) where:
    - packet_size: the size of the valid packet found
    - offset: number of garbage bytes to skip before the valid packet

    Parameters
    ----------
    buffer : bytes
        The current data buffer.

    Returns
    -------
    tuple[int, int] or None
        (packet_size, offset) if a valid packet is found, None otherwise.
    """
    if len(buffer) < 2:
        return None

    # Scan for valid packet start
    for i in range(len(buffer)):
        if i + MIC_PACKET_SIZE <= len(buffer):
            # Check for MIC packet: header=0xAA, trailer=0x55
            if buffer[i] == MIC_HEADER and buffer[i + MIC_PACKET_SIZE - 1] == MIC_TRAILER:
                return (MIC_PACKET_SIZE, i)

        if i + EMG_PACKET_SIZE <= len(buffer):
            # Check for EMG packet: header=0x55, trailer=0xAA
            if buffer[i] == EMG_HEADER and buffer[i + EMG_PACKET_SIZE - 1] == EMG_TRAILER:
                return (EMG_PACKET_SIZE, i)

        # If we've checked this position and neither packet type matches,
        # and we don't have enough data for either packet type, stop scanning
        if i + MIC_PACKET_SIZE > len(buffer) and i + EMG_PACKET_SIZE > len(buffer):
            break

    # Not enough data or no valid packet found
    return None


# Export the function as packetSize - the system will detect it's callable
packetSize = _packetSizeFn
"""Callable that determines packet size dynamically based on buffer contents."""

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

    Distinguishes between microphone and sEMG packets based on the header byte
    and packet size. Returns the decoded signal immediately, with an empty array
    for the other signal type.

    Parameters
    ----------
    data : bytes
        A packet of either 131 bytes (MIC) or 234 bytes (EMG).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing the decoded signals:
        - For mic packets: {"biogap": empty, "mic": audio_data}
        - For sEMG packets: {"biogap": emg_data, "mic": empty}
    """
    packet_len = len(data)
    header = data[0]

    if packet_len == MIC_PACKET_SIZE and header == MIC_HEADER:
        # This is a microphone packet
        trailer = data[-1]
        if trailer != MIC_TRAILER:
            raise ValueError(
                f"Invalid mic trailer: 0x{trailer:02X}, expected 0x{MIC_TRAILER:02X}"
            )
        audio = _decode_mic(data)
        emg = np.zeros((0, EMG_N_CHANNELS), dtype=np.float32)
        return {"biogap": emg, "mic": audio}
    elif packet_len == EMG_PACKET_SIZE and header == EMG_HEADER:
        # This is an sEMG packet
        trailer = data[-1]
        if trailer != EMG_TRAILER:
            raise ValueError(
                f"Invalid EMG trailer: 0x{trailer:02X}, expected 0x{EMG_TRAILER:02X}"
            )
        emg = _decode_emg(data)
        audio = np.zeros((0, 1), dtype=np.float32)
        return {"biogap": emg, "mic": audio}
    else:
        raise ValueError(
            f"Invalid packet: size={packet_len}, header=0x{header:02X}"
        )
