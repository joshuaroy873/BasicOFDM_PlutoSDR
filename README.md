# PlutoSDR OFDM System

A basic OFDM communication system for the ADALM Pluto SDR, created with GitHub Copilot assistance.

## Current Status
⚠️ **Reception Quality**: Currently experiencing poor reception with ~1% data match rate due to RF channel conditions and 20dB attenuation.

## What This Project Does
- **OFDM Modulation**: 64-point FFT with QPSK modulation
- **Frame Synchronization**: Preamble-based correlation for frame detection
- **Error Detection**: CRC32 checksums for data validation
- **SDR Interface**: Direct control of ADALM Pluto SDR hardware
- **Hex Data Transmission**: Sends/receives hexadecimal values (00-FF)
- **Loopback Testing**: Automated testing of complete transmission chain

## System Specifications
- **Frequency**: 2.4 GHz
- **Bandwidth**: 10 MHz
- **Sample Rate**: 10 MSPS
- **Modulation**: QPSK
- **Frame Structure**: 7-byte headers with sequence numbers and CRC32

## Usage
```bash
# Run loopback test
python examples/ofdm_loopback_test.py
```

*Note: This project was developed with GitHub Copilot assistance for educational and experimental purposes.*