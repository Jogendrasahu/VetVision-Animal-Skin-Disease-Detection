"""
Phase 7: Test-Time Augmentation (TTA) Evaluation
Evaluates:
1. Baseline vs Ensemble vs Ensemble + TTA on Validation Set
2. Tests 4 medically safe transforms:
   - Original
   - Horizontal Flip
   - Slight Center Crop (95% crop resized to 224)
   - Mild Brightness (+5%)
3. Analyzes inference latency vs accuracy gain
"""
import os, sys, json, time
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PROJECT = Path(r"c:\Users\Jogen\OneDrive\Desktop\Jogendra\Projects\Animal_Disease_App")
DATA = PROJECT / "data"
MODELS = PROJECT / "models"

DOG_CLASSES = ['demodicosis', 'dental_disease', 'dermatitis', 'distemper', 'eye_infection', 'fungal_infection', 'healthy', 'hot_spots', 'hypersensitivity', 'kennel_cough', 'mange', 'parvovirus', 'ringworm', 'skin_allergy', 'tick_infestation', 'worm_infection']
CAT_CLASSES = ['dental_disease', 'ear_mites', 'eye_infection', 'feline_acne', 'feline_leukemia', 'feline_panleukopenia', 'flea_allergy', 'fleas', 'fungal_infection', 'healthy', 'ringworm', 'scabies', 'skin_allergy', 'urinary_tract_infection', 'worm_infection']

def load_images_and_labels(manifest_path, classes):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        items = json.load(f)
    cls2idx = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for x in items:
        if x['label'] in cls2idx and Path(x['path']).exists():
            paths.append(x['path'])
            labels.append(cls2idx[x['label']])
    return paths, np.array(labels, dtype=np.int32)

def evaluate_tta_for_species(species, classes, model_filename, manifest_filename):
    print(f"\n--- Evaluating TTA on {species.upper()} ({manifest_filename}) ---")
    paths, y_true = load_images_and_labels(DATA / manifest_filename, classes)
    
    # Load model
    model = tf.keras.models.load_model(MODELS / model_filename)
    
    # Preprocessing pipelines for TTA
    def parse_orig(p):
        raw = tf.io.read_file(p)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [224, 224])
        return tf.cast(img, tf.float32)
        
    def parse_hflip(p):
        raw = tf.io.read_file(p)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [224, 224])
        img = tf.image.flip_left_right(img)
        return tf.cast(img, tf.float32)
        
    def parse_crop(p):
        raw = tf.io.read_file(p)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [236, 236])
        img = tf.image.central_crop(img, 224 / 236)
        img = tf.image.resize(img, [224, 224])
        return tf.cast(img, tf.float32)
        
    def parse_bright(p):
        raw = tf.io.read_file(p)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [224, 224])
        img = tf.image.adjust_brightness(img, 0.05 * 255.0)
        img = tf.clip_by_value(img, 0.0, 255.0)
        return tf.cast(img, tf.float32)
        
    # Evaluate 1-view (Standard)
    ds_orig = tf.data.Dataset.from_tensor_slices(paths).map(parse_orig, num_parallel_calls=tf.data.AUTOTUNE).batch(32)
    t0 = time.time()
    p_orig = model.predict(ds_orig, verbose=0)
    t_std = time.time() - t0
    
    acc_std = accuracy_score(y_true, np.argmax(p_orig, axis=1))
    f1_std = f1_score(y_true, np.argmax(p_orig, axis=1), average='macro')
    
    # Evaluate TTA (4 views)
    ds_hflip = tf.data.Dataset.from_tensor_slices(paths).map(parse_hflip, num_parallel_calls=tf.data.AUTOTUNE).batch(32)
    ds_crop = tf.data.Dataset.from_tensor_slices(paths).map(parse_crop, num_parallel_calls=tf.data.AUTOTUNE).batch(32)
    ds_bright = tf.data.Dataset.from_tensor_slices(paths).map(parse_bright, num_parallel_calls=tf.data.AUTOTUNE).batch(32)
    
    t0 = time.time()
    p_hflip = model.predict(ds_hflip, verbose=0)
    p_crop = model.predict(ds_crop, verbose=0)
    p_bright = model.predict(ds_bright, verbose=0)
    t_tta = time.time() - t0 + t_std
    
    p_tta = (p_orig * 0.40) + (p_hflip * 0.25) + (p_crop * 0.20) + (p_bright * 0.15)
    acc_tta = accuracy_score(y_true, np.argmax(p_tta, axis=1))
    f1_tta = f1_score(y_true, np.argmax(p_tta, axis=1), average='macro')
    bal_std = balanced_accuracy_score(y_true, np.argmax(p_orig, axis=1))
    bal_tta = balanced_accuracy_score(y_true, np.argmax(p_tta, axis=1))
    
    print(f"  Standard 1-View : Acc={acc_std*100:.2f}%, BalAcc={bal_std*100:.2f}%, MacroF1={f1_std:.4f} (Latency: {t_std:.2f}s)")
    print(f"  4-View TTA      : Acc={acc_tta*100:.2f}%, BalAcc={bal_tta*100:.2f}%, MacroF1={f1_tta:.4f} (Latency: {t_tta:.2f}s, {t_tta/t_std:.1f}x slower)")
    
    return {
        "species": species,
        "n_samples": len(y_true),
        "standard": {
            "accuracy": float(acc_std),
            "balanced_accuracy": float(bal_std),
            "macro_f1": float(f1_std),
            "time_seconds": float(t_std)
        },
        "tta": {
            "accuracy": float(acc_tta),
            "balanced_accuracy": float(bal_tta),
            "macro_f1": float(f1_tta),
            "time_seconds": float(t_tta)
        },
        "delta_accuracy": float(acc_tta - acc_std),
        "delta_macro_f1": float(f1_tta - f1_std)
    }

def main():
    print("="*75)
    print("PHASE 7: RIGOROUS TEST-TIME AUGMENTATION (TTA) BENCHMARK")
    print("="*75)
    
    # Evaluate TTA on Validation Set first
    dog_val_tta = evaluate_tta_for_species("dog", DOG_CLASSES, "Dog_Ensemble_v1.keras", "manifest_dog_val.json")
    cat_val_tta = evaluate_tta_for_species("cat", CAT_CLASSES, "Cat_Ensemble_v1.keras", "manifest_cat_val.json")
    
    results = {
        "dog_validation": dog_val_tta,
        "cat_validation": cat_val_tta
    }
    
    with open(DATA / "tta_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[DONE] Saved TTA results to: data/tta_benchmark_results.json")

if __name__ == "__main__":
    main()

