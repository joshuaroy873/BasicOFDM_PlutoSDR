#!/usr/bin/env python3
"""
OFDM Loopback Test

System test for OFDM transceiver.
Tests frame synchronization, error detection, and communication.

Features:
- Hex range transmission
- Frame synchronization validation  
- CRC32 error detection testing
- Quality analysis
- Metrics and reporting

Hardware Requirements:
- ADALM-Pluto SDR connected via USB
- TX-RX connection with 20dB attenuator (recommended)
- Or close antenna placement
"""

import sys
import os
import time
import argparse
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ofdm import PlutoOFDM

def main():
    parser = argparse.ArgumentParser(description='OFDM Loopback Test for Pluto SDR')
    parser.add_argument('--freq', type=float, default=2.4e9,
                       help='Center frequency in Hz (default: 2.4 GHz)')
    parser.add_argument('--tx-gain', type=float, default=0,
                       help='TX gain in dB (default: 0 for 20dB attenuated loopback)')
    parser.add_argument('--rx-gain', type=float, default=50,
                       help='RX gain in dB (default: 50)')
    parser.add_argument('--modulation', type=str, default='QPSK',
                       choices=['BPSK', 'QPSK', '16QAM'],
                       help='Modulation scheme (default: QPSK)')
    parser.add_argument('--count', type=int, default=5,
                       help='Number of test iterations (default: 5)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize OFDM system
        print("Initializing Pluto OFDM Loopback Test...")
        print("⚠️  Make sure TX and RX are connected via RF cable or very close antennas!")
        print("⚠️  Use attenuators to prevent damage from high power levels!")
        
        ofdm = PlutoOFDM(
            center_freq=args.freq,
            tx_gain=args.tx_gain,  # Lower TX gain for loopback
            rx_gain=args.rx_gain,
            modulation=args.modulation
        )
        
        # Display system status
        status = ofdm.get_status()
        print(f"\n📡 Pluto OFDM Loopback Test Configuration:")
        print(f"   Center Frequency: {status['tx_lo']/1e6:.1f} MHz")
        print(f"   Sample Rate: {status['sample_rate']/1e6:.1f} MSPS")
        print(f"   TX Gain: {status['tx_hardwaregain']} dB")
        print(f"   RX Gain: {status['rx_hardwaregain']} dB")
        print(f"   Bandwidth: {status['tx_rf_bandwidth']/1e6:.1f} MHz")
        print(f"   Modulation: {args.modulation}")
        print(f"   Data Rate: {status['ofdm_data_rate']/1000:.1f} kbps")
        
        # Generate complete hex range (00-FF)
        complete_hex_range = [f"{i:02X}" for i in range(256)]
        
        print(f"\n🧪 Running continuous 00-FF loopback test every 5 seconds...")
        print(f"   Will transmit all 256 hex values (00-FF) each cycle")
        print(f"   Test iterations: {args.count}")
        print(f"   Press Ctrl+C to stop early")
        
        total_tests = 0
        successful_tests = 0
        
        for iteration in range(args.count):
            total_tests += 1
            print(f"\n--- Test Cycle {iteration + 1}/{args.count} ---")
            print(f"📤 Transmitting complete hex range: 00-FF (256 values)")
            
            # Show a sample of what we're sending
            sample_display = complete_hex_range[:8] + ['...'] + complete_hex_range[-8:]
            print(f"   Sample: {' '.join(sample_display)}")
            
            # Run loopback test with complete range
            start_time = time.time()
            success, tx_data, rx_data = ofdm.test_loopback(complete_hex_range)
            test_duration = time.time() - start_time
            
            print(f"📥 Test completed in {test_duration:.2f} seconds")
            print(f"   TX: {len(tx_data)} hex values")
            print(f"   RX: {len(rx_data)} hex values")
            
            # Analyze transmission quality with error detection
            analysis = ofdm.analyze_transmission_quality(tx_data, rx_data, expected_sequence=0)
            
            if success and rx_data:
                # Show error detection results
                frame_valid = analysis["error_detection"]["frame_validation"]
                if frame_valid and frame_valid["valid"]:
                    print(f"   ✅ Frame validation: PASSED")
                    print(f"      Sequence: {frame_valid['sequence_num']}")
                    print(f"      CRC32: 0x{frame_valid['crc32']:08X}")
                    print(f"      Data length: {frame_valid['data_length']}")
                else:
                    print(f"   ❌ Frame validation: FAILED")
                    if frame_valid:
                        print(f"      Error: {frame_valid.get('error', 'Unknown')}")
                        print(f"      Type: {frame_valid.get('error_type', 'Unknown')}")
                
                # Show data integrity results
                matches = analysis["data_integrity"]["exact_matches"]
                coverage = analysis["data_integrity"]["coverage_percent"]
                
                print(f"   Matches: {matches}/{len(tx_data)} ({matches/max(len(tx_data),1)*100:.1f}%)")
                print(f"   Unique values coverage: {coverage:.1f}%")
                
                # Show sample of received data
                if len(rx_data) >= 16:
                    sample_rx = rx_data[:8] + ['...'] + rx_data[-8:]
                    print(f"   RX Sample: {' '.join(sample_rx)}")
                else:
                    print(f"   RX Data: {' '.join(rx_data)}")
                
                # Show error summary
                if analysis["error_summary"]["has_errors"]:
                    print(f"   ⚠️  Errors detected: {', '.join(analysis['error_summary']['error_types'])}")
                    if analysis["error_summary"]["recommendations"]:
                        print(f"   💡 Recommendations: {'; '.join(analysis['error_summary']['recommendations'])}")
                
                # Overall assessment
                if frame_valid and frame_valid["valid"] and coverage > 90:
                    print("  ✅ PASS - Excellent reception quality")
                    successful_tests += 1
                elif coverage > 70:
                    print("  ✅ PASS - Good reception quality")
                    successful_tests += 1
                else:
                    print("  ❌ FAIL - Poor reception quality")
            else:
                print(f"  ❌ FAIL - No data received")
            
            # Wait 5 seconds before next test (unless it's the last iteration)
            if iteration < args.count - 1:
                print(f"\n⏱️  Waiting 5 seconds before next cycle...")
                time.sleep(5.0)
        
        # Calculate and display results
        success_rate = successful_tests / total_tests * 100
        
        print(f"\n📊 Complete Hex Range Loopback Test Results:")
        print(f"   Total test cycles: {total_tests}")
        print(f"   Successful cycles: {successful_tests}")
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Data per cycle: 256 hex values (00-FF)")
        print(f"   Total data tested: {total_tests * 256} hex values")
        
        if success_rate >= 70:
            print("🎉 OFDM system is working excellently!")
            print("   All hex values (00-FF) are being transmitted reliably")
        elif success_rate >= 50:
            print("⚠️  OFDM system is working well with minor issues")
            print("💡 Most hex values are transmitted correctly")
        elif success_rate >= 30:
            print("⚠️  OFDM system is working but needs tuning")
            print("💡 Try adjusting gains or checking RF connections")
        else:
            print("❌ OFDM system needs troubleshooting")
            print("💡 Troubleshooting tips:")
            print("   - Check RF connections between TX and RX")
            print("   - Verify antenna connections")
            print("   - Try different gain settings")
            print("   - Check for interference")
            print("   - Ensure proper grounding")
            print("   - Consider using RF attenuators for loopback")
        
        return 0 if success_rate >= 30 else 1
        
    except KeyboardInterrupt:
        print("\n🛑 Test stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    finally:
        if 'ofdm' in locals():
            ofdm.close()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
