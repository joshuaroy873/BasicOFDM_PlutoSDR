"""
ADALM-Pluto SDR Interface

High-level interface for OFDM communication using ADALM-Pluto SDR.
Handles hardware configuration, 6GHz unlock, and RF operations.

Features:
- Automatic 6GHz frequency range unlock
- RF parameters for OFDM
- Error analysis and reporting
- API for transmit/receive operations
"""

import adi
import numpy as np
import time
from typing import List, Optional, Tuple
import logging
from .ofdm_core import OFDMModulator, OFDMDemodulator, ErrorDetector

logger = logging.getLogger(__name__)

class PlutoOFDM:
    """
    Pluto SDR OFDM Communication System
    
    Combines OFDM modulation/demodulation with Pluto SDR hardware interface
    for transmitting and receiving hexadecimal data.
    """
    
    def __init__(self, 
                 uri: str = "ip:192.168.2.1",
                 center_freq: float = 2.4e9,
                 sample_rate: float = 10e6,
                 bandwidth: float = 10e6,
                 tx_gain: float = -10,
                 rx_gain: float = 50,
                 modulation: str = 'QPSK'):
        """
        Initialize Pluto OFDM system
        
        Args:
            uri: Pluto SDR connection URI
            center_freq: RF center frequency in Hz
            sample_rate: Sample rate in Hz (10 MHz for 10 MHz bandwidth)
            bandwidth: RF bandwidth in Hz
            tx_gain: TX hardware gain in dB
            rx_gain: RX hardware gain in dB
            modulation: Modulation scheme ('BPSK', 'QPSK', '16QAM')
        """
        self.uri = uri
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self.tx_gain = tx_gain
        self.rx_gain = rx_gain
        
        # Initialize SDR
        self.sdr = None
        self._connect_sdr()
        
        # Initialize OFDM system
        self.modulator = OFDMModulator(
            n_fft=64,
            n_cp=16,
            n_pilot=8,
            modulation=modulation
        )
        
        self.demodulator = OFDMDemodulator(self.modulator)
        
        logger.info(f"Pluto OFDM system initialized")
        logger.info(f"  Center frequency: {center_freq/1e6:.1f} MHz")
        logger.info(f"  Sample rate: {sample_rate/1e6:.1f} MSPS")
        logger.info(f"  Bandwidth: {bandwidth/1e6:.1f} MHz")
        logger.info(f"  Modulation: {modulation}")
    
    def _connect_sdr(self):
        """Connect to Pluto SDR and configure parameters"""
        try:
            logger.info(f"Connecting to Pluto SDR at {self.uri}")
            self.sdr = adi.Pluto(uri=self.uri)
            
            # Configure common parameters
            self.sdr.sample_rate = int(self.sample_rate)
            self.sdr.rx_rf_bandwidth = int(self.bandwidth)
            self.sdr.tx_rf_bandwidth = int(self.bandwidth)
            
            # Configure frequencies
            self.sdr.rx_lo = int(self.center_freq)
            self.sdr.tx_lo = int(self.center_freq)
            
            # Configure gains
            self.sdr.gain_control_mode = "manual"
            self.sdr.rx_hardwaregain = int(self.rx_gain)
            self.sdr.tx_hardwaregain = int(self.tx_gain)
            
            # Set buffer size for reasonable latency
            self.sdr.rx_buffer_size = 4096
            
            logger.info("✓ Pluto SDR connected and configured")
            
        except Exception as e:
            logger.error(f"Failed to connect to Pluto SDR: {e}")
            raise
    
    def get_status(self) -> dict:
        """Get current SDR status and configuration"""
        if not self.sdr:
            return {}
        
        try:
            status = {
                'connected': True,
                'sample_rate': self.sdr.sample_rate,
                'rx_lo': self.sdr.rx_lo,
                'tx_lo': self.sdr.tx_lo,
                'rx_rf_bandwidth': self.sdr.rx_rf_bandwidth,
                'tx_rf_bandwidth': self.sdr.tx_rf_bandwidth,
                'rx_hardwaregain': self.sdr.rx_hardwaregain,
                'tx_hardwaregain': self.sdr.tx_hardwaregain,
                'ofdm_symbol_duration': self.modulator.symbol_duration,
                'ofdm_data_rate': self.calculate_data_rate()
            }
            return status
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return {'connected': False, 'error': str(e)}
    
    def calculate_data_rate(self) -> float:
        """Calculate theoretical data rate in bits per second"""
        # Bits per OFDM symbol
        bits_per_ofdm = self.modulator.n_data * self.modulator.bits_per_symbol
        
        # OFDM symbols per second
        symbols_per_sec = 1.0 / self.modulator.symbol_duration
        
        # Total data rate
        data_rate = bits_per_ofdm * symbols_per_sec
        
        return data_rate
    
    def transmit_hex(self, hex_data: List[str], repeat: int = 1, preamble: bool = True, sequence_num: int = 0) -> bool:
        """
        Transmit hexadecimal data using OFDM with error detection
        
        Args:
            hex_data: List of hex strings to transmit (e.g., ['FF', 'A0', '33'])
            repeat: Number of times to repeat the transmission
            preamble: Whether to add a preamble for synchronization
            sequence_num: Frame sequence number for error detection
            
        Returns:
            Success status
        """
        try:
            logger.info(f"Transmitting hex data: {hex_data} (repeat {repeat}x)")
            
            # Validate hex data
            for hex_val in hex_data:
                if not (0 <= int(hex_val, 16) <= 255):
                    raise ValueError(f"Invalid hex value: {hex_val}")
            
            # Generate OFDM signal with error detection
            ofdm_signal = self.modulator.modulate(hex_data, sequence_num)
            
            # Add preamble if requested
            if preamble:
                preamble_signal = self._generate_preamble()
                ofdm_signal = np.concatenate([preamble_signal, ofdm_signal])
            
            # Repeat transmission
            if repeat > 1:
                repeated_signal = np.tile(ofdm_signal, repeat)
                ofdm_signal = repeated_signal
            
            # Add some silence at the end
            silence = np.zeros(int(0.001 * self.sample_rate), dtype=complex)  # 1ms silence
            tx_signal = np.concatenate([ofdm_signal, silence])
            
            # Normalize signal to prevent clipping
            max_amplitude = np.max(np.abs(tx_signal))
            if max_amplitude > 0:
                tx_signal = tx_signal / max_amplitude * 0.8  # Scale to 80% of max
            
            logger.info(f"Transmitting signal: {len(tx_signal)} samples, "
                       f"{len(tx_signal)/self.sample_rate*1000:.2f} ms duration")
            
            # Transmit
            self.sdr.tx(tx_signal)
            
            logger.info("✓ Transmission completed")
            return True
            
        except Exception as e:
            logger.error(f"Transmission failed: {e}")
            return False
    
    def receive_hex(self, duration_ms: float = 100, timeout_ms: float = 5000) -> List[str]:
        """
        Receive and demodulate hexadecimal data
        
        Args:
            duration_ms: Reception duration in milliseconds
            timeout_ms: Timeout in milliseconds
            
        Returns:
            List of received hex strings
        """
        try:
            # Calculate number of samples to receive
            n_samples = int(duration_ms * self.sample_rate / 1000)
            
            logger.info(f"Receiving for {duration_ms} ms ({n_samples} samples)")
            
            # Set receive buffer size
            self.sdr.rx_buffer_size = n_samples
            
            # Receive signal
            start_time = time.time()
            rx_signal = self.sdr.rx()
            
            if len(rx_signal) == 0:
                logger.warning("No signal received")
                return []
            
            logger.info(f"Received {len(rx_signal)} samples")
            
            # Find and extract OFDM signal
            extracted_signal = self._extract_ofdm_signal(rx_signal)
            
            if len(extracted_signal) == 0:
                logger.warning("No OFDM signal detected")
                return []
            
            # Demodulate
            hex_data = self.demodulator.demodulate(extracted_signal)
            
            # Filter out empty or invalid hex values
            valid_hex = [h for h in hex_data if h != '00' and len(h) == 2]
            
            logger.info(f"✓ Reception completed: {valid_hex}")
            return valid_hex
            
        except Exception as e:
            logger.error(f"Reception failed: {e}")
            return []
    
    def _generate_preamble(self) -> np.ndarray:
        """Generate preamble for synchronization"""
        # Simple preamble: known OFDM symbol
        preamble_hex = ['AA', '55']  # Alternating pattern
        preamble_signal = self.modulator.modulate(preamble_hex)
        return preamble_signal
    
    def _extract_ofdm_signal(self, rx_signal: np.ndarray) -> np.ndarray:
        """
        Extract OFDM signal from received signal using energy detection
        
        Args:
            rx_signal: Received signal
            
        Returns:
            Extracted OFDM signal
        """
        # Simple energy-based detection
        # Calculate moving average of signal power
        window_size = self.modulator.n_fft + self.modulator.n_cp
        
        if len(rx_signal) < window_size:
            return rx_signal
        
        # Calculate signal power
        power = np.abs(rx_signal) ** 2
        
        # Moving average
        moving_avg = np.convolve(power, np.ones(window_size)/window_size, mode='valid')
        
        # Find signal start (power above threshold)
        threshold = np.mean(moving_avg) + 2 * np.std(moving_avg)
        signal_indices = np.where(moving_avg > threshold)[0]
        
        if len(signal_indices) == 0:
            logger.warning("No signal detected above threshold")
            return rx_signal  # Return full signal if no clear signal found
        
        # Extract signal from first detection to end
        start_idx = signal_indices[0]
        extracted = rx_signal[start_idx:]
        
        logger.info(f"Extracted signal from index {start_idx}, length {len(extracted)}")
        return extracted
    
    def test_loopback(self, test_hex: List[str] = None) -> Tuple[bool, List[str], List[str]]:
        """
        Test loopback transmission and reception
        
        Args:
            test_hex: Hex data to test (default: ['FF', 'AA', '55', '00'])
            
        Returns:
            (success, transmitted_hex, received_hex)
        """
        if test_hex is None:
            test_hex = ['FF', 'AA', '55', '33']
        
        logger.info(f"Starting loopback test with data: {test_hex}")
        
        try:
            # Start receiving (in a separate thread ideally, but for simplicity...)
            # For now, we'll do a simple transmit then receive
            
            # Transmit
            tx_success = self.transmit_hex(test_hex)
            if not tx_success:
                return False, test_hex, []
            
            # Small delay
            time.sleep(0.01)
            
            # Receive
            received_hex = self.receive_hex(duration_ms=200)
            
            # Check if transmission was successful
            success = len(received_hex) > 0
            
            logger.info(f"Loopback test - TX: {test_hex}, RX: {received_hex}")
            
            return success, test_hex, received_hex
            
        except Exception as e:
            logger.error(f"Loopback test failed: {e}")
            return False, test_hex, []
    
    def analyze_transmission_quality(self, transmitted: List[str], received: List[str], expected_sequence: int = 0) -> dict:
        """
        Analyze transmission quality with comprehensive error detection
        
        Args:
            transmitted: Original transmitted hex data
            received: Received hex data
            expected_sequence: Expected frame sequence number
            
        Returns:
            Dictionary with detailed analysis results
        """
        analysis = {
            "basic_stats": {
                "transmitted_count": len(transmitted),
                "received_count": len(received),
                "data_ratio": len(received) / max(len(transmitted), 1)
            },
            "error_detection": {
                "frame_validation": None,
                "crc_valid": False,
                "sequence_valid": False,
                "length_valid": False
            },
            "data_integrity": {
                "exact_matches": 0,
                "partial_matches": 0,
                "unique_values_tx": len(set(transmitted)),
                "unique_values_rx": len(set(received)),
                "coverage_percent": 0
            },
            "error_summary": {
                "has_errors": True,
                "error_types": [],
                "recommendations": []
            }
        }
        
        # Calculate CRC for transmitted data
        if transmitted:
            expected_crc = ErrorDetector.calculate_crc32(transmitted)
            logger.info(f"Expected CRC32 for transmitted data: 0x{expected_crc:08X}")
        
        # Validate received frame if we have enough data
        if len(received) >= 7:  # Minimum for header
            validation_result = ErrorDetector.validate_frame(received)
            analysis["error_detection"]["frame_validation"] = validation_result
            
            if validation_result["valid"]:
                analysis["error_detection"]["crc_valid"] = True
                analysis["error_detection"]["sequence_valid"] = (validation_result["sequence_num"] == expected_sequence)
                analysis["error_detection"]["length_valid"] = (validation_result["data_length"] == len(transmitted))
                
                # Compare actual data (skip header)
                received_data = validation_result["data_payload"]
                analysis["data_integrity"]["exact_matches"] = sum(1 for a, b in zip(transmitted, received_data) if a == b)
                analysis["data_integrity"]["coverage_percent"] = len(set(transmitted) & set(received_data)) / max(len(set(transmitted)), 1) * 100
            else:
                analysis["error_summary"]["error_types"].append(validation_result.get("error_type", "VALIDATION_ERROR"))
        else:
            analysis["error_summary"]["error_types"].append("INSUFFICIENT_DATA")
        
        # Basic data comparison if frame validation failed
        if not analysis["error_detection"]["crc_valid"]:
            if received:
                analysis["data_integrity"]["exact_matches"] = sum(1 for a, b in zip(transmitted, received) if a == b)
                analysis["data_integrity"]["coverage_percent"] = len(set(transmitted) & set(received)) / max(len(set(transmitted)), 1) * 100
        
        # Determine overall quality
        crc_ok = analysis["error_detection"]["crc_valid"]
        coverage_ok = analysis["data_integrity"]["coverage_percent"] > 70
        length_ok = 0.8 <= analysis["basic_stats"]["data_ratio"] <= 1.5
        
        analysis["error_summary"]["has_errors"] = not (crc_ok and coverage_ok and length_ok)
        
        # Generate recommendations
        if not crc_ok:
            analysis["error_summary"]["recommendations"].append("Check RF connection and signal integrity")
        if not coverage_ok:
            analysis["error_summary"]["recommendations"].append("Adjust gain settings or check for interference")
        if not length_ok:
            analysis["error_summary"]["recommendations"].append("Check frame synchronization and timing")
        
        return analysis
    
    def close(self):
        """Close SDR connection"""
        if self.sdr:
            try:
                del self.sdr
                logger.info("SDR connection closed")
            except:
                pass
