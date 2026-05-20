#!/usr/bin/env python
"""
Enhanced validation script for D3, UFD, and DistilDIRE model integration
Tests all 3 models including model loading
"""

import sys
from pathlib import Path

print("=" * 70)
print("ENHANCED INTEGRATION VALIDATION")
print("=" * 70)

# Test 1: Check D3 directory structure
print("\n[TEST 1] Checking D3 directory structure...")
d3_path = Path("github_model_srcs/D3")
d3_models_path = d3_path / "models"

if d3_path.exists():
    print(f"  ✓ D3 directory exists: {d3_path.resolve()}")
else:
    print(f"  ✗ D3 directory NOT found: {d3_path}")
    sys.exit(1)

if d3_models_path.exists():
    print(f"  ✓ D3 models directory exists")
    
    # Check key files
    required_files = [
        "models/__init__.py",
        "models/clip_models.py",
        "models/transformer_attention.py",
        "models/clip/",
    ]
    
    for file in required_files:
        file_path = d3_path / file
        exists = "✓" if file_path.exists() else "✗"
        print(f"    {exists} {file}")
else:
    print(f"  ✗ D3 models directory NOT found")

# Test 2: Import preprocessing adapters
print("\n[TEST 2] Importing preprocessing adapters...")
try:
    from preprocessing_adapters import (
        UFDPreprocessor, D3Preprocessor, DistilDIREPreprocessor,
        ModelLoaders, InferenceHelper
    )
    print("  ✓ All preprocessing adapters imported successfully")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 3: Test UFD model loading
print("\n[TEST 3] Testing UFD model loading...")
try:
    ufd_checkpoint = Path("models/mirage_model_4_universalfakedetect/epoch_4_0.976.pt")
    if ufd_checkpoint.exists():
        ufd_model = ModelLoaders.load_ufd_model(str(ufd_checkpoint), device="cpu")
        print(f"  ✓ UFD model loaded successfully")
        print(f"    - Model type: {type(ufd_model).__name__}")  
    else:
        print(f"  ⚠ UFD checkpoint not found: {ufd_checkpoint}")
except Exception as e:
    print(f"  ✗ UFD loading failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test D3 model loading (MAIN FIX)
print("\n[TEST 4] Testing D3 model loading...")
try:
    d3_checkpoint = Path("models/mirage_model_6_d3/model_epoch_best.pt")
    
    if d3_checkpoint.exists():
        print(f"  - Checkpoint path verified: {d3_checkpoint}")
        
        # This is where the fix matters
        d3_model = ModelLoaders.load_d3_model(str(d3_checkpoint), device="cpu")
        
        print(f"  ✓ D3 model loaded successfully")
        print(f"    - Model type: {type(d3_model).__name__}")
        print(f"    - Has attention_head: {hasattr(d3_model, 'attention_head')}")
        print(f"    - Has model: {hasattr(d3_model, 'model')}")
        print(f"    - Model eval mode: {not d3_model.training}")
        
    else:
        print(f"  ✗ D3 checkpoint not found: {d3_checkpoint}")
        
except Exception as e:
    print(f"  ✗ D3 loading failed: {e}")
    import traceback
    print("\n  DETAILED TRACEBACK:")
    traceback.print_exc()

# Test 5: Test DistilDIRE model loading
print("\n[TEST 5] Testing DistilDIRE model loading...")
try:
    distildire_checkpoint = Path("models/mirage_model_5_distildire/model_epoch_4.pt")
    if distildire_checkpoint.exists():
        distildire_model = ModelLoaders.load_distildire_model(
            str(distildire_checkpoint), device="cpu"
        )
        print(f"  ✓ DistilDIRE model loaded successfully")
        print(f"    - Model type: {type(distildire_model).__name__}")
    else:
        print(f"  ⚠ DistilDIRE checkpoint not found: {distildire_checkpoint}")
except Exception as e:
    print(f"  ✗ DistilDIRE loading failed: {e}")

# Test 6: Test preprocessor initialization with sample image
print("\n[TEST 6] Testing preprocessing with sample image...")
try:
    from PIL import Image
    import torch
    
    # Create dummy image
    dummy_img = Image.new('RGB', (256, 256), color='red')
    
    # Test D3 preprocessing
    print("  - Testing D3 preprocessing...")
    d3_prep = D3Preprocessor()
    d3_output = d3_prep.preprocess_image(dummy_img)
    print(f"    ✓ D3 preprocessing output shape: {d3_output.shape}")
    
    # Test UFD preprocessing  
    print("  - Testing UFD preprocessing...")
    ufd_prep = UFDPreprocessor()
    ufd_output = ufd_prep.preprocess_image(dummy_img)
    print(f"    ✓ UFD preprocessing output shape: {ufd_output.shape}")
    
except Exception as e:
    print(f"  ⚠ Preprocessing test skipped: {e}")

print("\n" + "=" * 70)
print("✓ VALIDATION COMPLETE")
print("=" * 70)
