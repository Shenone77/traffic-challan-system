"""
diagnose_model.py
-----------------
Run this ONCE from your project root to find out exactly what's
inside best_model.pth and which loading strategy works.

Usage:
    python diagnose_model.py
"""

import os, sys
import numpy as np

PTH = os.path.join('models', 'best_model.pth')

if not os.path.exists(PTH):
    print(f"❌ File not found: {PTH}")
    sys.exit(1)

file_size = os.path.getsize(PTH)
print(f"📁 File: {PTH}  ({file_size / 1024:.1f} KB)\n")

# ── 1. Peek at raw bytes (magic numbers) ─────────────────────────────────────
with open(PTH, 'rb') as f:
    header = f.read(16)

print(f"🔍 First 16 bytes (hex): {header.hex()}")
if header[:2] == b'\x89H' or header[1:4] == b'HDF':
    print("   → Looks like an HDF5 file (.h5) — Keras can load this directly")
elif header[:2] == b'PK':
    print("   → Looks like a ZIP/PyTorch file — torch.load() needed")
elif header[:6] == b'\x80\x02ctorch':
    print("   → Looks like a pickled PyTorch object")
else:
    print("   → Unknown format")

print()

# ── 2. Try torch.load ────────────────────────────────────────────────────────
print("=" * 50)
print("Strategy 1 & 2 & 3 — torch.load()")
print("=" * 50)
try:
    import torch
    obj = torch.load(PTH, map_location='cpu')
    print(f"✅ torch.load() succeeded")
    print(f"   Type: {type(obj)}")

    if isinstance(obj, dict):
        print(f"   Dict keys ({len(obj)}): {list(obj.keys())[:10]}")
        sample_vals = list(obj.values())[:3]
        for i, v in enumerate(sample_vals):
            print(f"   val[{i}]: type={type(v)}, ", end='')
            if hasattr(v, 'shape'):
                print(f"shape={v.shape}")
            else:
                print(f"value={str(v)[:60]}")

        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            print("   → Strategy 2 (state_dict of tensors) is likely correct ✅")

    elif isinstance(obj, (list, tuple)):
        print(f"   List/tuple of {len(obj)} items")
        for i, v in enumerate(obj[:3]):
            if hasattr(v, 'shape'):
                print(f"   [{i}] shape={getattr(v,'shape',None)} dtype={getattr(v,'dtype',None)}")
        print("   → Strategy 3 (weight list) is likely correct ✅")

    elif hasattr(obj, 'predict'):
        print("   → Strategy 1 (full model object) — can call .predict() directly ✅")

    else:
        print(f"   → Unknown object, repr: {repr(obj)[:200]}")

except Exception as e:
    print(f"❌ torch.load() failed: {e}")

print()

# ── 3. Try keras load_model ──────────────────────────────────────────────────
print("=" * 50)
print("Strategy 4 — keras load_model()")
print("=" * 50)
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    model = load_model(PTH, compile=False)
    print(f"✅ keras load_model() succeeded")
    print(f"   Input shape:  {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    print(f"   Layers: {len(model.layers)}")
    # quick test
    dummy = np.zeros((1, 28, 28, 3), dtype=np.float32)
    out   = model.predict(dummy, verbose=0)
    print(f"   Dummy predict output shape: {out.shape}  sum={out.sum():.4f}")
    print("   → Strategy 4 works ✅  Use this in anpr_model_v2.py")
except Exception as e:
    print(f"❌ keras load_model() failed: {e}")

print()

# ── 4. Try keras load_weights on rebuilt arch ─────────────────────────────────
print("=" * 50)
print("Strategy 5 — keras load_weights() on rebuilt architecture")
print("=" * 50)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

    model = Sequential([
        Conv2D(16, (22,22), input_shape=(28,28,3), activation='relu', padding='same'),
        Conv2D(32, (16,16), activation='relu', padding='same'),
        Conv2D(64, (8,8),   activation='relu', padding='same'),
        Conv2D(64, (4,4),   activation='relu', padding='same'),
        MaxPooling2D(pool_size=(4,4)),
        Dropout(0.4),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(36, activation='softmax'),
    ])
    model.load_weights(PTH)
    dummy = np.zeros((1,28,28,3), dtype=np.float32)
    out   = model.predict(dummy, verbose=0)
    print(f"✅ keras load_weights() succeeded")
    print(f"   Dummy predict output shape: {out.shape}  sum={out.sum():.4f}")
    print("   → Strategy 5 works ✅")
except Exception as e:
    print(f"❌ keras load_weights() failed: {e}")

print()
print("=" * 50)
print("DONE. Copy the ✅ strategy number into anpr_model_v2.py")
print("The _load_cnn() function will try all strategies automatically,")
print("but this tells you which one to expect.")
print("=" * 50)
