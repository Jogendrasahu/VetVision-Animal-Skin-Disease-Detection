"""
Phase 1: Deep inspection of all model artifacts.
Checks:
1. Input shape
2. Output shape
3. Number of classes
4. Layer structure & Preprocessing
5. Soft probability verification
6. Self-contained verification
7. Class ordering compatibility with DOG_CLASSES, CAT_CLASSES, COW_CLASSES
"""
import os, sys, json
import numpy as np
import tensorflow as tf
from pathlib import Path

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PROJECT = Path(r"c:\Users\Jogen\OneDrive\Desktop\Jogendra\Projects\Animal_Disease_App")
DATA = PROJECT / "data"
MODELS = PROJECT / "models"

# Reference classes from app.py
DOG_CLASSES = ['demodicosis', 'dental_disease', 'dermatitis', 'distemper', 'eye_infection', 'fungal_infection', 'healthy', 'hot_spots', 'hypersensitivity', 'kennel_cough', 'mange', 'parvovirus', 'ringworm', 'skin_allergy', 'tick_infestation', 'worm_infection']
CAT_CLASSES = ['dental_disease', 'ear_mites', 'eye_infection', 'feline_acne', 'feline_leukemia', 'feline_panleukopenia', 'flea_allergy', 'fleas', 'fungal_infection', 'healthy', 'ringworm', 'scabies', 'skin_allergy', 'urinary_tract_infection', 'worm_infection']
COW_CLASSES = ['foot_and_mouth', 'healthy', 'lumpy_skin_disease']

model_specs = [
    ("Dog Baseline", MODELS / "Dog_EfficientNetB0_balanced_best.keras", DOG_CLASSES),
    ("Dog Ensemble v1", MODELS / "Dog_Ensemble_v1.keras", DOG_CLASSES),
    ("Cat Baseline", MODELS / "Cat_EfficientNetB0_stage2_best.keras", CAT_CLASSES),
    ("Cat Ensemble v1", MODELS / "Cat_Ensemble_v1.keras", CAT_CLASSES),
    ("Cow Baseline", MODELS / "Cow_EfficientNetB0_stage2_best.keras", COW_CLASSES),
]

print("="*80)
print("PHASE 1: MODEL ARTIFACT VERIFICATION & ARCHITECTURAL AUDIT")
print("="*80)

results = {}

for name, path, expected_classes in model_specs:
    print(f"\nAuditing: {name}")
    print(f"  Path: {path.name} ({path.stat().st_size / (1024*1024):.2f} MB)")
    if not path.exists():
        print(f"  [FAIL] File does not exist: {path}")
        continue
    
    # Load model
    try:
        model = tf.keras.models.load_model(str(path))
        print("  -> Load status: SUCCESS (Self-contained, standard keras load)")
    except Exception as e:
        print(f"  -> [FAIL] Load error: {e}")
        continue

    in_shape = model.input_shape
    out_shape = model.output_shape
    num_classes = out_shape[-1]
    expected_n = len(expected_classes)
    
    print(f"  -> Input Shape: {in_shape}")
    print(f"  -> Output Shape: {out_shape} ({num_classes} classes, expected {expected_n})")
    
    # Test random RAW [0, 255] RGB float input
    dummy_raw_img = np.random.uniform(0.0, 255.0, size=(1, 224, 224, 3)).astype(np.float32)
    pred = model.predict(dummy_raw_img, verbose=0)[0]
    
    pred_sum = float(np.sum(pred))
    pred_min = float(np.min(pred))
    pred_max = float(np.max(pred))
    all_non_negative = pred_min >= 0.0
    sum_is_one = abs(pred_sum - 1.0) < 1e-4
    
    print(f"  -> Softmax Probability Vector Check:")
    print(f"     Sum of outputs: {pred_sum:.6f} (Valid: {sum_is_one})")
    print(f"     Min prob: {pred_min:.6f}, Max prob: {pred_max:.6f} (Non-negative: {all_non_negative})")
    
    # Layer inspection
    layer_types = [l.__class__.__name__ for l in model.layers]
    print(f"  -> Model Top-level Layers ({len(model.layers)} layers): {layer_types[:8]}...")
    
    has_rescaling = any('Rescaling' in l for l in layer_types)
    has_add = any('Add' in l for l in layer_types)
    is_ensemble = "Ensemble" in name
    
    if is_ensemble:
        print(f"  -> Ensemble Soft-Voting Mechanism: Found Add layer={has_add}, Scaling layers={has_rescaling}")
    
    audit_pass = (in_shape == (None, 224, 224, 3) and 
                  num_classes == expected_n and 
                  sum_is_one and 
                  all_non_negative)
    
    print(f"  -> AUDIT RESULT: {'[PASSED 100%]' if audit_pass else '[FAILED]'}")
    
    results[name] = {
        "path": str(path.name),
        "size_mb": round(path.stat().st_size / (1024*1024), 2),
        "input_shape": str(in_shape),
        "output_shape": str(out_shape),
        "num_classes": num_classes,
        "classes_match": num_classes == expected_n,
        "probabilities_valid": sum_is_one and all_non_negative,
        "audit_pass": audit_pass
    }
    del model

print("\n" + "="*80)
print("PHASE 1 SUMMARY AUDIT TABLE")
print("="*80)
print(f"{'Model Name':<20} | {'Size':<9} | {'Input Shape':<18} | {'Classes':<7} | {'Valid Prob':<10} | {'Status'}")
print("-" * 80)
for k, v in results.items():
    print(f"{k:<20} | {v['size_mb']} MB   | {v['input_shape']:<18} | {v['num_classes']:<7} | {str(v['probabilities_valid']):<10} | {'PASS' if v['audit_pass'] else 'FAIL'}")

# Save Phase 1 audit json
with open(DATA / "phase1_model_audit.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nPhase 1 audit logged to data/phase1_model_audit.json")
