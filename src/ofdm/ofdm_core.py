"""
OFDM Core Implementation

OFDM modulation/demodulation system with frame synchronization and error detection.

Features:
- BPSK, QPSK, and 16-QAM modulation
- Frame synchronization with correlation-based detection  
- CRC32 error detection and validation
- Signal processing for SDR applications
"""

import numpy as np
import scipy.signal as signal
from typing import List, Tuple, Optional, Dict
import logging
import zlib  # For CRC32 calculation
import struct  # For binary data packing

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorDetector:
    """
    Error detection utilities for OFDM transmission
    
    Provides CRC calculation, frame validation, and comprehensive error reporting
    """
    
    @staticmethod
    def calculate_crc32(data: List[str]) -> int:
        """Calculate CRC32 for hex data list"""
        # Convert hex strings to bytes
        byte_data = bytes([int(hex_val, 16) for hex_val in data])
        return zlib.crc32(byte_data) & 0xffffffff
    
    @staticmethod
    def create_frame_header(sequence_num: int, data_length: int, crc32: int) -> List[str]:
        """
        Create frame header with sequence number, length, and CRC
        
        Args:
            sequence_num: Frame sequence number (0-255)
            data_length: Number of data bytes (0-65535)
            crc32: CRC32 checksum
            
        Returns:
            Header as list of hex strings [seq, len_hi, len_lo, crc3, crc2, crc1, crc0]
        """
        header = []
        header.append(f"{sequence_num:02X}")  # Sequence number (1 byte)
        header.append(f"{(data_length >> 8) & 0xFF:02X}")  # Length high byte
        header.append(f"{data_length & 0xFF:02X}")  # Length low byte
        header.append(f"{(crc32 >> 24) & 0xFF:02X}")  # CRC32 byte 3
        header.append(f"{(crc32 >> 16) & 0xFF:02X}")  # CRC32 byte 2
        header.append(f"{(crc32 >> 8) & 0xFF:02X}")   # CRC32 byte 1
        header.append(f"{crc32 & 0xFF:02X}")          # CRC32 byte 0
        return header
    
    @staticmethod
    def parse_frame_header(header: List[str]) -> Dict:
        """
        Parse frame header to extract metadata
        
        Args:
            header: List of 7 hex strings
            
        Returns:
            Dictionary with sequence_num, data_length, crc32
        """
        if len(header) < 7:
            return {"valid": False, "error": "Header too short"}
        
        try:
            # Header format: [seq_hi, seq_lo, length, crc3, crc2, crc1, crc0]
            sequence_num = (int(header[0], 16) << 8) | int(header[1], 16)  # 2 bytes
            data_length = int(header[2], 16)  # 1 byte
            crc32 = (int(header[3], 16) << 24) | (int(header[4], 16) << 16) | \
                   (int(header[5], 16) << 8) | int(header[6], 16)  # 4 bytes
            
            return {
                "valid": True,
                "sequence_num": sequence_num,
                "data_length": data_length,
                "crc32": crc32
            }
        except ValueError:
            return {"valid": False, "error": "Invalid hex values in header"}
    
    @staticmethod
    def validate_frame(received_data: List[str], expected_crc32: int = None) -> Dict:
        """
        Validate received frame data with comprehensive error checking
        
        Args:
            received_data: List of hex strings from demodulation
            expected_crc32: Expected CRC32 checksum (optional)
            
        Returns:
            Dictionary with validation results and error analysis
        """
        if len(received_data) < 7:
            return {
                "valid": False,
                "error": "FRAME_TOO_SHORT",
                "error_type": "FRAME_TOO_SHORT",
                "details": f"Frame has {len(received_data)} bytes, minimum 7 required"
            }
        
        # Try to parse header
        header_info = ErrorDetector.parse_frame_header(received_data[:7])
        
        if not header_info["valid"]:
            # Header parsing failed, try fallback analysis
            return ErrorDetector._fallback_frame_analysis(received_data, expected_crc32)
        
        sequence_num = header_info["sequence_num"]
        expected_length = header_info["data_length"]
        header_crc32 = header_info["crc32"]
        
        # Check if frame length makes sense
        actual_payload_length = len(received_data) - 7  # Subtract header
        
        if expected_length != actual_payload_length:
            return {
                "valid": False,
                "error": f"LENGTH_MISMATCH: expected {expected_length}, got {actual_payload_length}",
                "error_type": "LENGTH_MISMATCH",
                "sequence_num": sequence_num,
                "expected_length": expected_length,
                "actual_length": actual_payload_length,
                "header_crc32": f"0x{header_crc32:08X}",
                "payload": received_data[7:]
            }
        
        # Extract payload and calculate CRC32
        payload = received_data[7:]
        if not payload:
            return {
                "valid": False,
                "error": "NO_PAYLOAD",
                "error_type": "NO_PAYLOAD",
                "sequence_num": sequence_num
            }
        
        calculated_crc32 = ErrorDetector.calculate_crc32(payload)
        
        # Check CRC32 match
        if calculated_crc32 != header_crc32:
            return {
                "valid": False,
                "error": f"CRC_MISMATCH: calculated 0x{calculated_crc32:08X}, header 0x{header_crc32:08X}",
                "error_type": "CRC_MISMATCH",
                "sequence_num": sequence_num,
                "calculated_crc32": f"0x{calculated_crc32:08X}",
                "header_crc32": f"0x{header_crc32:08X}",
                "payload": payload
            }
        
        # If expected CRC32 provided, check against it too
        if expected_crc32 is not None and expected_crc32 != calculated_crc32:
            return {
                "valid": False,
                "error": f"EXPECTED_CRC_MISMATCH: calculated 0x{calculated_crc32:08X}, expected 0x{expected_crc32:08X}",
                "error_type": "EXPECTED_CRC_MISMATCH",  
                "sequence_num": sequence_num,
                "calculated_crc32": f"0x{calculated_crc32:08X}",
                "expected_crc32": f"0x{expected_crc32:08X}",
                "payload": payload
            }
        
        # Frame is valid
        return {
            "valid": True,
            "sequence_num": sequence_num,
            "payload": payload,
            "crc32": f"0x{calculated_crc32:08X}"
        }
    
    @staticmethod
    def _fallback_frame_analysis(received_data: List[str], expected_crc32: int = None) -> Dict:
        """
        Fallback analysis when header parsing fails due to corruption
        
        Args:
            received_data: Corrupted received data
            expected_crc32: Expected CRC32 if known
            
        Returns:
            Analysis results with corruption assessment
        """
        total_length = len(received_data)
        
        # Try different possible payload lengths and check CRC32
        best_match = None
        best_score = 0
        
        # Try lengths around expected values (256 + 7 = 263)
        for payload_start in range(min(10, total_length)):
            for payload_length in [256, 128, 64, 32, 16]:  # Common lengths
                if payload_start + payload_length <= total_length:
                    payload = received_data[payload_start:payload_start + payload_length]
                    calculated_crc32 = ErrorDetector.calculate_crc32(payload)
                    
                    # Score based on how reasonable the payload looks
                    score = ErrorDetector._score_payload_validity(payload, expected_crc32, calculated_crc32)
                    
                    if score > best_score:
                        best_score = score
                        best_match = {
                            "payload_start": payload_start,
                            "payload_length": payload_length,
                            "payload": payload,
                            "calculated_crc32": calculated_crc32,
                            "score": score
                        }
        
        corruption_level = ErrorDetector._assess_corruption_level(received_data, expected_crc32)
        
        return {
            "valid": False,
            "error": "HEADER_CORRUPTED",
            "error_type": "HEADER_CORRUPTED", 
            "total_length": total_length,
            "corruption_level": corruption_level,
            "best_payload_guess": best_match,
            "raw_data_sample": received_data[:20]  # First 20 bytes for analysis
        }
    
    @staticmethod
    def _score_payload_validity(payload: List[str], expected_crc32: int, calculated_crc32: int) -> float:
        """Score how likely a payload segment is valid"""
        score = 0.0
        
        # Length score (prefer common lengths)
        length = len(payload)
        if length == 256:
            score += 10.0  # Perfect match for 00-FF
        elif length in [128, 64, 32]:
            score += 5.0
        elif length > 0:
            score += 1.0
            
        # CRC32 match score
        if expected_crc32 is not None and calculated_crc32 == expected_crc32:
            score += 50.0  # Very high score for CRC match
        
        # Hex value distribution score (check if it looks like structured data)
        if length > 10:
            unique_values = len(set(payload))
            if unique_values > length * 0.5:  # Good diversity
                score += 2.0
            if unique_values == length:  # All unique (like 00-FF)
                score += 5.0
                
        return score
    
    @staticmethod  
    def _assess_corruption_level(data: List[str], expected_crc32: int = None) -> str:
        """Assess the level of data corruption"""
        if len(data) == 0:
            return "COMPLETE"
            
        # Check for patterns that suggest partial recovery
        unique_count = len(set(data))
        total_count = len(data)
        
        if unique_count < total_count * 0.1:
            return "SEVERE"
        elif unique_count < total_count * 0.3:
            return "HIGH"  
        elif unique_count < total_count * 0.6:
            return "MODERATE"
        else:
            return "LOW"

class OFDMModulator:
    """
    OFDM Modulator with error correction capabilities
    """
    
    def __init__(self, n_fft=64, n_cp=16, n_pilot=8, modulation='QPSK'):
        """
        Initialize OFDM modulator
        
        Args:
            n_fft: FFT size (64)
            n_cp: Cyclic prefix length (16)
            n_pilot: Number of pilot subcarriers (8) 
            modulation: Modulation scheme ('BPSK', 'QPSK', '16QAM')
        """
        self.n_fft = n_fft
        self.n_cp = n_cp
        self.n_pilot = n_pilot
        self.modulation = modulation
        self.sample_rate = 10e6  # 10 MSPS
        
        # Setup constellation
        self._setup_constellation()
        
        # Calculate subcarrier allocation
        self.n_data = n_fft - n_pilot - 1  # -1 for DC
        
        # Pilot positions (evenly spaced)
        self.pilot_indices = np.linspace(1, n_fft-1, n_pilot, dtype=int)
        
        # Data subcarrier indices (excluding pilots and DC)
        all_indices = np.arange(1, n_fft)
        self.data_indices = np.setdiff1d(all_indices, self.pilot_indices)
        self.n_data = len(self.data_indices)
        
        # Setup pilot symbols (BPSK for robust channel estimation)
        self.pilot_symbols = np.ones(len(self.pilot_indices), dtype=complex)
        
        # Calculate symbol duration
        self.symbol_duration = (n_fft + n_cp) / self.sample_rate
        
        # Generate preamble for frame synchronization
        self.preamble = self._generate_preamble()
        
        logging.info(f"OFDM Modulator initialized:")
        logging.info(f"  FFT size: {n_fft}")
        logging.info(f"  Data subcarriers: {self.n_data}")
        logging.info(f"  Pilot subcarriers: {n_pilot}")
        logging.info(f"  Modulation: {modulation}")
        logging.info(f"  Symbol duration: {(n_fft + n_cp) / 10e6 * 1e6:.2f} μs")
    
    def _setup_subcarriers(self):
        """Setup subcarrier allocation (data, pilots, nulls)"""
        # Subcarrier indices (avoid DC at index 0)
        all_indices = np.arange(1, self.n_fft//2)  # Only positive frequencies
        
        # Pilot subcarrier positions (evenly spaced)
        pilot_spacing = len(all_indices) // self.n_pilots
        self.pilot_indices = all_indices[::pilot_spacing][:self.n_pilots]
        
        # Data subcarrier positions (remaining subcarriers)
        self.data_indices = np.setdiff1d(all_indices, self.pilot_indices)
        
        # Mirror for negative frequencies
        self.pilot_indices = np.concatenate([self.pilot_indices, 
                                           self.n_fft - self.pilot_indices])
        self.data_indices = np.concatenate([self.data_indices,
                                          self.n_fft - self.data_indices])
    
    def _setup_constellation(self):
        """Setup modulation constellation"""
        if self.modulation == 'BPSK':
            self.constellation = np.array([1+0j, -1+0j])
            self.bits_per_symbol = 1
        elif self.modulation == 'QPSK':
            self.constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
            self.bits_per_symbol = 2
        elif self.modulation == '16QAM':
            # 16-QAM constellation
            points = [-3, -1, 1, 3]
            constellation = []
            for i in points:
                for q in points:
                    constellation.append(i + 1j*q)
            self.constellation = np.array(constellation) / np.sqrt(10)
            self.bits_per_symbol = 4
        else:
            raise ValueError(f"Unsupported modulation: {self.modulation}")
    
    def _generate_pilots(self) -> np.ndarray:
        """Generate pilot symbols for channel estimation"""
        # Use BPSK pilots for simplicity
        pilot_bits = np.random.randint(0, 2, len(self.pilot_indices))
        return 2 * pilot_bits - 1  # Convert to BPSK symbols
    
    def _generate_preamble(self) -> np.ndarray:
        """
        Generate frame synchronization preamble
        
        Returns:
            Time-domain preamble sequence for frame detection
        """
        # Create a known sequence in frequency domain for good correlation properties
        # Use Zadoff-Chu sequence or similar for good autocorrelation
        preamble_freq = np.zeros(self.n_fft, dtype=complex)
        
        # Fill with alternating pattern for good correlation
        for i in range(1, self.n_fft//2):
            if i % 2 == 0:
                preamble_freq[i] = 1 + 0j
                preamble_freq[self.n_fft - i] = 1 + 0j
            else:
                preamble_freq[i] = -1 + 0j
                preamble_freq[self.n_fft - i] = -1 + 0j
        
        # IFFT to time domain
        preamble_time = np.fft.ifft(preamble_freq)
        
        # Add cyclic prefix
        cp = preamble_time[-self.n_cp:]
        preamble_with_cp = np.concatenate([cp, preamble_time])
        
        # Repeat preamble twice for better detection
        full_preamble = np.concatenate([preamble_with_cp, preamble_with_cp])
        
        return full_preamble
    
    def hex_to_bits(self, hex_values: List[str]) -> np.ndarray:
        """
        Convert list of hex strings to bit array
        
        Args:
            hex_values: List of hex strings (e.g., ['FF', 'A0', '33'])
            
        Returns:
            Bit array
        """
        bits = []
        for hex_val in hex_values:
            # Convert hex to integer, then to 8-bit binary
            int_val = int(hex_val, 16)
            bit_string = format(int_val, '08b')
            bits.extend([int(b) for b in bit_string])
        
        return np.array(bits)
    
    def bits_to_symbols(self, bits: np.ndarray) -> np.ndarray:
        """
        Map bits to constellation symbols
        
        Args:
            bits: Input bit array
            
        Returns:
            Complex symbols
        """
        # Pad bits if necessary
        n_pad = (-len(bits)) % self.bits_per_symbol
        if n_pad > 0:
            bits = np.concatenate([bits, np.zeros(n_pad, dtype=int)])
        
        # Group bits and map to symbols
        bit_groups = bits.reshape(-1, self.bits_per_symbol)
        symbols = []
        
        for group in bit_groups:
            # Convert bit group to integer index
            index = 0
            for i, bit in enumerate(group):
                index += bit * (2 ** (self.bits_per_symbol - 1 - i))
            symbols.append(self.constellation[index])
        
        return np.array(symbols)
    
    def modulate_ofdm_symbol(self, data_symbols: np.ndarray) -> np.ndarray:
        """
        Create one OFDM symbol
        
        Args:
            data_symbols: Data symbols for this OFDM symbol
            
        Returns:
            Time-domain OFDM symbol with cyclic prefix
        """
        # Initialize frequency domain symbol
        freq_symbol = np.zeros(self.n_fft, dtype=complex)
        
        # Place data symbols
        n_data_this_symbol = min(len(data_symbols), len(self.data_indices))
        freq_symbol[self.data_indices[:n_data_this_symbol]] = data_symbols[:n_data_this_symbol]
        
        # Place pilot symbols
        freq_symbol[self.pilot_indices] = self.pilot_symbols
        
        # IFFT to time domain
        time_symbol = np.fft.ifft(freq_symbol)
        
        # Add cyclic prefix
        cp = time_symbol[-self.n_cp:]
        ofdm_symbol = np.concatenate([cp, time_symbol])
        
        return ofdm_symbol
    
    def modulate(self, hex_data: List[str], sequence_num: int = 0) -> np.ndarray:
        """
        Complete OFDM modulation chain with error detection
        
        Args:
            hex_data: List of hex strings to transmit
            sequence_num: Frame sequence number (0-255)
            
        Returns:
            Complex baseband OFDM signal with preamble and error checking
        """
        logger.info(f"Modulating {len(hex_data)} hex values: {hex_data}")
        
        # Calculate CRC32 for the data
        crc32 = ErrorDetector.calculate_crc32(hex_data)
        logger.info(f"Calculated CRC32: 0x{crc32:08X}")
        
        # Create frame header (sequence, length, CRC)
        header = ErrorDetector.create_frame_header(sequence_num, len(hex_data), crc32)
        logger.info(f"Frame header: {header}")
        
        # Combine header and data
        frame_data = header + hex_data
        logger.info(f"Frame with header: {len(frame_data)} total hex values")
        
        # Convert hex to bits
        bits = self.hex_to_bits(frame_data)
        logger.info(f"Total bits: {len(bits)}")
        
        # Map bits to symbols
        data_symbols = self.bits_to_symbols(bits)
        logger.info(f"Data symbols: {len(data_symbols)}")
        
        # Calculate number of OFDM symbols needed
        symbols_per_ofdm = len(self.data_indices)
        n_ofdm_symbols = int(np.ceil(len(data_symbols) / symbols_per_ofdm))
        logger.info(f"OFDM symbols needed: {n_ofdm_symbols}")
        
        # Pad data symbols if necessary
        total_symbols_needed = n_ofdm_symbols * symbols_per_ofdm
        if len(data_symbols) < total_symbols_needed:
            padding = np.zeros(total_symbols_needed - len(data_symbols), dtype=complex)
            data_symbols = np.concatenate([data_symbols, padding])
        
        # Start with preamble
        ofdm_signal = list(self.preamble)
        
        # Generate OFDM symbols
        for i in range(n_ofdm_symbols):
            start_idx = i * symbols_per_ofdm
            end_idx = start_idx + symbols_per_ofdm
            symbol_data = data_symbols[start_idx:end_idx]
            
            ofdm_symbol = self.modulate_ofdm_symbol(symbol_data)
            ofdm_signal.extend(ofdm_symbol)
        
        ofdm_signal = np.array(ofdm_signal)
        logger.info(f"Generated OFDM signal length: {len(ofdm_signal)} samples")
        logger.info(f"Signal duration: {len(ofdm_signal)/self.sample_rate*1000:.2f} ms")
        
        return ofdm_signal


class OFDMDemodulator:
    """
    OFDM Demodulator for Pluto SDR
    
    Implements OFDM demodulation with channel estimation and equalization.
    """
    
    def __init__(self, modulator: OFDMModulator):
        """
        Initialize demodulator with same parameters as modulator
        
        Args:
            modulator: Corresponding OFDMModulator instance
        """
        self.mod = modulator  # Reference to modulator for parameters
        
        # Copy key parameters
        self.n_fft = modulator.n_fft
        self.n_cp = modulator.n_cp
        self.n_data = modulator.n_data
        self.constellation = modulator.constellation
        self.bits_per_symbol = modulator.bits_per_symbol
        self.data_indices = modulator.data_indices
        self.pilot_indices = modulator.pilot_indices
        self.pilot_symbols = modulator.pilot_symbols
        self.preamble = modulator.preamble
        self.sample_rate = modulator.sample_rate
        
        # Frame detection parameters
        self.correlation_threshold = 0.15  # Lower threshold for 20dB attenuated signals
        self.min_frame_gap = 5000  # Minimum samples between frames (5ms @ 10MSPS)
        self.max_frames_expected = 1  # Maximum number of frames to detect (expect only 1)
        
        logger.info("OFDM Demodulator initialized")
    
    def detect_frames(self, received_signal: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect OFDM frames in received signal using preamble correlation
        
        Args:
            received_signal: Received complex baseband signal
            
        Returns:
            List of tuples (start_index, end_index) for detected frames
        """
        logger.info(f"Detecting frames in signal of length {len(received_signal)}")
        
        # Cross-correlate with preamble
        correlation = np.correlate(received_signal, self.preamble, mode='full')
        correlation_mag = np.abs(correlation)
        
        # Normalize correlation
        preamble_energy = np.sum(np.abs(self.preamble)**2)
        signal_energy = np.convolve(np.abs(received_signal)**2, 
                                   np.ones(len(self.preamble)), mode='full')
        
        # Avoid division by zero
        signal_energy = np.maximum(signal_energy, 1e-10)
        normalized_correlation = correlation_mag / np.sqrt(preamble_energy * signal_energy)
        
        # Debug: Log correlation statistics
        max_corr = np.max(normalized_correlation)
        mean_corr = np.mean(normalized_correlation)
        logger.info(f"Frame detection: max_corr={max_corr:.3f}, mean_corr={mean_corr:.3f}, threshold={self.correlation_threshold}")
        
        # Find peaks above threshold, but only keep the strongest ones
        peaks = []
        threshold_exceeded = normalized_correlation > self.correlation_threshold
        peak_indices = np.where(threshold_exceeded)[0]
        
        if len(peak_indices) > 0:
            logger.info(f"Found {len(peak_indices)} correlation peaks above threshold")
            
            # Get correlation values for peaks and sort by strength
            peak_values = normalized_correlation[peak_indices]
            sorted_indices = np.argsort(peak_values)[::-1]  # Sort descending
            
            # Only keep the strongest peaks that are also local maxima
            for idx in sorted_indices[:self.max_frames_expected]:  # Limit to expected frames
                i = peak_indices[idx]
                # Check if this is a local maximum
                local_max = True
                for j in range(max(0, i-20), min(len(normalized_correlation), i+21)):  # Wider window
                    if j != i and normalized_correlation[j] > normalized_correlation[i]:
                        local_max = False
                        break
                if local_max:
                    peaks.append(i)
        else:
            logger.info("No correlation peaks found above threshold")
        
        # Convert correlation indices to signal indices
        # correlation is 'full' mode, so offset by len(preamble)-1
        frame_starts = []
        for peak in peaks:
            signal_start = peak - len(self.preamble) + 1
            if 0 <= signal_start < len(received_signal):
                frame_starts.append(signal_start)
        
        # Remove duplicate detections (too close together) and limit number of frames
        filtered_starts = []
        for start in sorted(frame_starts):
            if len(filtered_starts) >= self.max_frames_expected:
                logger.info(f"Limiting frame detection to {self.max_frames_expected} frames")
                break
            if not filtered_starts or start - filtered_starts[-1] > self.min_frame_gap:
                filtered_starts.append(start)
        
        # Estimate frame lengths and create frame boundaries
        frames = []
        preamble_len = len(self.preamble)
        
        for i, start in enumerate(filtered_starts):
            # Frame starts after preamble
            data_start = start + preamble_len
            
            # More realistic frame length estimation
            # For 256 hex values (2048 bits), we need ~37 OFDM symbols with QPSK
            estimated_ofdm_symbols = 30  # Conservative estimate for typical hex data
            estimated_data_length = estimated_ofdm_symbols * (self.n_fft + self.n_cp)
            data_end = min(data_start + estimated_data_length, len(received_signal))
            
            if data_start < len(received_signal):
                frames.append((data_start, data_end))
        
        logger.info(f"Detected {len(frames)} frames with correlation threshold {self.correlation_threshold}")
        
        return frames
    
    def remove_cyclic_prefix(self, ofdm_symbols: np.ndarray) -> np.ndarray:
        """
        Remove cyclic prefix from OFDM symbols
        
        Args:
            ofdm_symbols: Time domain OFDM symbols with CP
            
        Returns:
            OFDM symbols without CP
        """
        symbol_length = self.n_fft + self.n_cp
        n_symbols = len(ofdm_symbols) // symbol_length
        
        symbols_no_cp = []
        for i in range(n_symbols):
            start = i * symbol_length + self.n_cp  # Skip CP
            end = start + self.n_fft
            symbols_no_cp.extend(ofdm_symbols[start:end])
        
        return np.array(symbols_no_cp)
    
    def estimate_channel(self, freq_symbol: np.ndarray) -> np.ndarray:
        """
        Estimate channel response using pilot subcarriers
        
        Args:
            freq_symbol: Frequency domain OFDM symbol
            
        Returns:
            Channel estimate for all subcarriers
        """
        # Extract received pilots
        received_pilots = freq_symbol[self.pilot_indices]
        
        # Calculate channel response at pilot locations
        pilot_channel = received_pilots / self.pilot_symbols
        
        # Interpolate channel response for all subcarriers
        # Simple linear interpolation
        channel_est = np.ones(self.n_fft, dtype=complex)
        
        # For simplicity, use average channel response
        # In practice, you'd use proper interpolation
        avg_channel = np.mean(pilot_channel)
        channel_est[self.data_indices] = avg_channel
        
        return channel_est
    
    def equalize(self, freq_symbol: np.ndarray, channel_est: np.ndarray) -> np.ndarray:
        """
        Equalize received symbol using channel estimate
        
        Args:
            freq_symbol: Received frequency domain symbol
            channel_est: Channel estimate
            
        Returns:
            Equalized symbol
        """
        # Zero-forcing equalization
        equalized = freq_symbol.copy()
        
        # Only equalize data subcarriers
        for idx in self.data_indices:
            if np.abs(channel_est[idx]) > 0.01:  # Avoid division by zero
                equalized[idx] = freq_symbol[idx] / channel_est[idx]
        
        return equalized
    
    def symbols_to_bits(self, symbols: np.ndarray) -> np.ndarray:
        """
        Demap symbols to bits using maximum likelihood detection
        
        Args:
            symbols: Received symbols
            
        Returns:
            Detected bits
        """
        bits = []
        
        for symbol in symbols:
            # Find closest constellation point
            distances = np.abs(self.constellation - symbol)
            closest_idx = np.argmin(distances)
            
            # Convert index to bits
            bit_string = format(closest_idx, f'0{self.bits_per_symbol}b')
            bits.extend([int(b) for b in bit_string])
        
        return np.array(bits)
    
    def bits_to_hex(self, bits: np.ndarray) -> List[str]:
        """
        Convert bit array to hex strings
        
        Args:
            bits: Input bit array
            
        Returns:
            List of hex strings
        """
        # Ensure bits length is multiple of 8
        n_pad = (-len(bits)) % 8
        if n_pad > 0:
            bits = np.concatenate([bits, np.zeros(n_pad, dtype=int)])
        
        hex_values = []
        for i in range(0, len(bits), 8):
            byte_bits = bits[i:i+8]
            # Convert bits to integer
            byte_val = 0
            for j, bit in enumerate(byte_bits):
                byte_val += bit * (2 ** (7 - j))
            hex_values.append(f"{byte_val:02X}")
        
        return hex_values
    
    def demodulate(self, received_signal: np.ndarray) -> List[str]:
        """
        Complete OFDM demodulation chain with frame detection
        
        Args:
            received_signal: Received complex baseband signal
            
        Returns:
            List of detected hex strings
        """
        logger.info(f"Demodulating signal of length {len(received_signal)}")
        
        # Detect frames in the signal
        frames = self.detect_frames(received_signal)
        
        if not frames:
            logger.warning("No frames detected in received signal")
            return []
        
        all_hex_data = []
        
        # Process each detected frame
        for frame_idx, (start_idx, end_idx) in enumerate(frames):
            logger.info(f"Processing frame {frame_idx+1}/{len(frames)}: samples {start_idx}-{end_idx}")
            
            frame_signal = received_signal[start_idx:end_idx]
            
            # Remove cyclic prefix
            symbols_no_cp = self.remove_cyclic_prefix(frame_signal)
            
            # Calculate number of OFDM symbols
            n_ofdm_symbols = len(symbols_no_cp) // self.n_fft
            logger.info(f"Processing {n_ofdm_symbols} OFDM symbols in frame {frame_idx+1}")
            
            frame_data_symbols = []
            
            # Process each OFDM symbol
            for i in range(n_ofdm_symbols):
                start = i * self.n_fft
                end = start + self.n_fft
                
                if end > len(symbols_no_cp):
                    break
                    
                time_symbol = symbols_no_cp[start:end]
                
                # FFT to frequency domain
                freq_symbol = np.fft.fft(time_symbol)
                
                # Channel estimation
                channel_est = self.estimate_channel(freq_symbol)
                
                # Equalization
                equalized = self.equalize(freq_symbol, channel_est)
                
                # Extract data symbols
                data_symbols = equalized[self.data_indices]
                frame_data_symbols.extend(data_symbols)
            
            if frame_data_symbols:
                frame_data_symbols = np.array(frame_data_symbols)
                logger.info(f"Extracted {len(frame_data_symbols)} data symbols from frame {frame_idx+1}")
                
                # Symbol to bits
                bits = self.symbols_to_bits(frame_data_symbols)
                logger.info(f"Detected {len(bits)} bits from frame {frame_idx+1}")
                
                # Bits to hex
                hex_data = self.bits_to_hex(bits)
                logger.info(f"Recovered {len(hex_data)} hex values from frame {frame_idx+1}")
                
                # Validate frame with error detection
                validation_result = ErrorDetector.validate_frame(hex_data)
                
                if validation_result["valid"]:
                    logger.info(f"✅ Frame {frame_idx+1} validation PASSED")
                    logger.info(f"   Sequence: {validation_result['sequence_num']}")
                    logger.info(f"   Data length: {validation_result['data_length']}")
                    logger.info(f"   CRC32: 0x{validation_result['crc32']:08X}")
                    
                    # Add only the data payload (excluding header)
                    all_hex_data.extend(validation_result["data_payload"])
                else:
                    logger.error(f"❌ Frame {frame_idx+1} validation FAILED")
                    logger.error(f"   Error type: {validation_result.get('error_type', 'UNKNOWN')}")
                    logger.error(f"   Error: {validation_result['error']}")
                    
                    # Still add raw data for analysis, but mark as invalid
                    logger.warning(f"   Adding raw data for analysis: {hex_data[:10]}...")
                    all_hex_data.extend(hex_data)
        
        logger.info(f"Total recovered hex data: {all_hex_data}")
        return all_hex_data
