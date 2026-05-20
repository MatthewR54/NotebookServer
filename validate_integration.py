#!/usr/bin/env python3
"""
Validation script for deepfake detector preprocessing integration
Tests import and basic functionality
"""

import sys
from pathlib import Path

# Add workspace to path
sys.path.insert(0, str(Path.cwd()))

print("=" * 70)
print("VALIDATION: Deepfake Detector Preprocessing Integration")
print("=" * 70)

# Test 1: Import preprocessing_adapters
print("\n[1/5] Testing preprocessing_adapters import...")
try:
    from preprocessing_adapters import (
        UFDPreprocessor, D3Preprocessor, DistilDIREPreprocessor,
        ModelLoaders, InferenceHelper
    )
    print("  ✓ Successfully imported all preprocessing adapters")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    sys.exit(1)

# Test 2: Check model paths exist
print("\n[2/5] Checking model paths...")
models_dir = Path("./models")
required_models = {
    "mirage_model_4_universalfakedetect": "UFD",
    "mirage_model_5_distildire": "DistilDIRE",
    "mirage_model_6_d3": "D3"
}

all_found = True
for model_dir, model_name in required_models.items():
    model_path = models_dir / model_dir
    if model_path.exists():
        print(f"  ✓ {model_name} model directory found: {model_path}")
    else:
        print(f"  ✗ {model_name} model directory NOT found: {model_path}")
        all_found = False

if not all_found:
    print("  ⚠ Some model directories missing (expected if first time)")

# Test 3: Check preprocessing utils
print("\n[3/5] Testing preprocessing utilities...")
try:
    import torch
    from PIL import Image
    import numpy as np
    
    # Create dummy image
    dummy_img = Image.new('RGB', (256, 256), color='red')
    
    # Test UFD preprocessor initialization
    device = "cpu"
    print("  - Testing UFD preprocessor...")
    try:
        ufd_proc = UFDPreprocessor(device=device)
        print("    ✓ UFD preprocessor initialized")
    except Exception as e:
        print(f"    ⚠ UFD preprocessor (may need transformers): {str(e)[:50]}...")
    
    # Test D3 preprocessor
    print("  - Testing D3 preprocessor...")
    try:
        d3_proc = D3Preprocessor(device=device)
        print("    ✓ D3 preprocessor initialized")
    except Exception as e:
        print(f"    ✗ D3 preprocessor failed: {e}")
    
    # Test DistilDIRE preprocessor
    print("  - Testing DistilDIRE preprocessor...")
    try:
        distildire_proc = DistilDIREPreprocessor(device=device)
        print("    ✓ DistilDIRE preprocessor initialized")
    except Exception as e:
        print(f"    ✗ DistilDIRE preprocessor failed: {e}")
    
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check streamlit_app imports
print("\n[4/5] Testing streamlit_app compatibility...")
try:
    # Just check if the main imports work
    import streamlit as st
    print("  ✓ Streamlit available")
except ImportError:
    print("  ⚠ Streamlit not installed (expected if not running via streamlit)")

# Test 5: Summary
print("\n[5/5] Checking file structure...")
expected_files = [
    "streamlit_app.py",
    "preprocessing_adapters.py",
    "precompute_dire_offline.py",
]

all_exist = True
for file in expected_files:
    if Path(file).exists():
        print(f"  ✓ {file}")
    else:
        print(f"  ✗ {file} NOT FOUND")
        all_exist = False

# Final summary
print("\n" + "=" * 70)
if all_exist:
    print("✓ VALIDATION COMPLETE - All files present")
    print("\nNEXT STEPS:")
    print("1. Run Streamlit: streamlit run streamlit_app.py")
    print("2. For DistilDIRE pre-computation (optional):")
    print("   python precompute_dire_offline.py --data_root <path> --save_root <path>")
    print("3. Test each model with sample images")
else:
    print("✗ VALIDATION INCOMPLETE - Some files missing")

print("=" * 70 + "\n")
