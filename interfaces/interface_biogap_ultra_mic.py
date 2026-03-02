"""
This module contains the BioGAP interface for microphone streaming.

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

# Microphone configuration
SAMPLE_RATE = 16000  # Hz
SAMPLE_BIT_WIDTH = 16  # bits
SAMPLES_PER_PACKET = 64  # 16-bit samples per BLE packet

# Packet format constants
MIC_HEADER = 0xAA
MIC_TRAILER = 0x55
MIC_PACKET_SIZE = 131

packetSize: int = [(MIC_HEADER, MIC_PACKET_SIZE)]
"""Number of bytes in each package."""

startSeq: list[bytes] = [
    0.5,
    (26).to_bytes()  # START_MIC_STREAMING command
]
"""Sequence of commands to start microphone streaming."""

stopSeq: list[bytes] = [
    0.5,
    (27).to_bytes()  # STOP_MIC_STREAMING command
]
"""Sequence of commands to stop microphone streaming."""

fs: list[float] = [SAMPLE_RATE]
"""Sequence of floats representing the sampling rate of each signal."""

nCh: list[int] = [1]
"""Sequence of integers representing the number of channels of each signal."""

sigInfo: dict = {"mic": {"fs": SAMPLE_RATE, "nCh": 1}}
"""Dictionary containing the signals information."""


def decodeFn(data: bytes) -> dict[str, np.ndarray]:
    """
    Function to decode the binary data received from BioGAP microphone into audio samples.

    Parameters
    ----------
    data : bytes
        A packet of 131 bytes containing microphone data.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing the audio samples with shape (nSamp, nCh).
        
    Packet format:
    - 1 byte header (0xAA)
    - 1 byte counter
    - 64 samples of 16-bit signed audio data (128 bytes)
    - 1 byte trailer (0x55)
    
    """
    
    header = data[0]
    trailer = data[-1]
    
    if header != MIC_HEADER:
        raise ValueError(f"Invalid header: 0x{header:02X}, expected 0x{MIC_HEADER:02X}")
    
    if trailer != MIC_TRAILER:
        raise ValueError(f"Invalid trailer: 0x{trailer:02X}, expected 0x{MIC_TRAILER:02X}")
    
    if len(data) != MIC_PACKET_SIZE:
        raise ValueError(f"Invalid packet size: {len(data)}, expected {MIC_PACKET_SIZE}")
    
    audio_data = data[2:2 + SAMPLES_PER_PACKET * 2]
    
    # Unpack 16-bit signed samples (little-endian)
    audio = np.array(
        struct.unpack(f"<{SAMPLES_PER_PACKET}h", audio_data),
        dtype=np.int16
    )
    
    # Reshape to (nSamp, nCh) format
    audio = audio.reshape(-1, 1)
    
    # Convert to float32 normalized to [-1.0, 1.0] range
    audio = audio.astype(np.float32) / 32768.0
    
    return {"mic": audio}