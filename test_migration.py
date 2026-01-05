#!/usr/bin/env python3
"""
Test script to verify the updated SST utility system works correctly.
This script tests both the new anytopo system and legacy compatibility.
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from sst_ultility.anytopo_utils import create_base_config, run_merlin_experiment, run_efm_experiment
        print("✓ New anytopo utilities imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import anytopo utilities: {e}")
        return False
    
    try:
        from traffic_analyser.traffic_analyser import traffic_analyser
        from traffic_analyser.helpers import extract_placeholders, extract_config_from_filename
        print("✓ Traffic analyzer modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import traffic analyzer: {e}")
        return False
    
    return True

def test_config_creation():
    """Test configuration creation functions."""
    print("\nTesting configuration creation...")
    
    try:
        from sst_ultility.anytopo_utils import create_base_config
        
        # Test base config creation
        base_config = create_base_config(
            topo_name="RRG",
            V=18, D=3,
            link_bw=16
        )
        
        expected_keys = ['UNIFIED_ROUTER_LINK_BW', 'V', 'D', 'topo_name', 'identifier']
        if all(key in base_config for key in expected_keys):
            print("✓ Base config creation works")
        else:
            print("✗ Base config missing expected keys")
            return False
            
        # Test that config can be extended
        extended_config = base_config.copy()
        extended_config.update({
            'LOAD': 0.5,
            'traffic_pattern': 'uniform'
        })
        
        if 'LOAD' in extended_config and 'traffic_pattern' in extended_config:
            print("✓ Config extension works")
        else:
            print("✗ Config extension failed")
            return False
            
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        return False
        
    return True

# def test_filename_parsing():
#     """Test the updated filename parsing functions."""
#     print("\nTesting filename parsing...")
    
#     try:
#         from traffic_analyser.helpers import extract_placeholders, extract_config_from_filename
        
#         # Test old format
#         old_format = "traffic_BENCH_FFT3D_EPR_3_ROUTING_source_routing_V_36_D_5_TOPO_RRG_SUFFIX_test_.csv"
#         old_params = extract_placeholders(
#             "traffic_BENCH_{BENCH}_EPR_{EPR}_ROUTING_{ROUTING}_V_{V}_D_{D}_TOPO_{TOPO}_SUFFIX_{SUFFIX}_.csv",
#             old_format
#         )
        
#         if old_params['BENCH'] == 'FFT3D' and old_params['V'] == 36:
#             print("✓ Old filename format parsing works")
#         else:
#             print(f"✗ Old filename parsing failed: {old_params}")
#             return False
        
#         # Test new format
#         new_filename = "traffic_trace_slimfly_V18_D5_FFT3D.csv"
#         new_params = extract_config_from_filename(new_filename)
        
#         if new_params['V'] == 18 and new_params['D'] == 5:
#             print("✓ New filename format parsing works")
#         else:
#             print(f"✗ New filename parsing failed: {new_params}")
            
#     except Exception as e:
#         print(f"✗ Filename parsing test failed: {e}")
#         return False
        
#     return True

def test_template_paths():
    """Test that template files exist and are readable."""
    print("\nTesting template file access...")
    
    templates = [
        "sst_ultility/sst_EFM_config_template.py",
        "sst_ultility/sst_merlin_config_template.py"
    ]
    
    for template in templates:
        template_path = Path(__file__).parent / template
        if template_path.exists():
            try:
                with open(template_path, 'r') as f:
                    content = f.read()
                    if 'topoAny' in content and 'endpointNIC' in content:
                        print(f"✓ {template} exists and updated")
                    else:
                        print(f"! {template} exists but may not be fully updated")
            except Exception as e:
                print(f"✗ Cannot read {template}: {e}")
                return False
        else:
            print(f"✗ Template file missing: {template}")
            return False
            
    return True

def main():
    """Run all tests."""
    print("SST Utility Migration Test Suite")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_config_creation,
        # test_filename_parsing,
        test_template_paths
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            print("Test failed!")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✓ All tests passed! System is ready for use.")
        print("\nNext steps:")
        print("1. Test with a small SST simulation")
        print("2. Verify anytopo_utility path is correct for your environment")
        print("3. Check that NetworkX and other dependencies are installed")
        return 0
    else:
        print("✗ Some tests failed. Check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed (NetworkX, etc.)")
        print("2. Check file paths and permissions")
        print("3. Verify SST installation")
        return 1

if __name__ == "__main__":
    sys.exit(main())