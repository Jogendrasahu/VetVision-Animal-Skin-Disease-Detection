"""
Advanced Model Analysis & Optimization Script
Executes:
1. Model Diversity Analysis (Prediction correlation, Cohen's Kappa between all candidate architectures).
2. Stacking Meta-Learner (Stratified 5-Fold Cross-Validation on validation features).
3. Confidence & Calibration Analysis (Prediction entropy, Temperature Scaling, ECE before and after).
4. Dermatological Error Analysis & Clinical Confusion Pairs.
"""
import os, sys, json
import numpy as np
import tensorflow as tf
from pathlib import Path
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PROJECT = Path(r"c:\Users\Jogen\OneDrive\Desktop\Jogendra\Projects\Animal_Disease_App")
DATA = PROJECT / "data"
MODELS = PROJECT / "models"
REPORTS = PROJECT / "reports"

DOG_CLASSES = ['demodicosis', 'dental_disease', 'dermatitis', 'distemper', 'eye_infection', 'fungal_infection', 'healthy', 'hot_spots', 'hypersensitivity', 'kennel_cough', 'mange', 'parvovirus', 'ringworm', 'skin_allergy', 'tick_infestation', 'worm_infection']
CAT_CLASSES = ['dental_disease', 'ear_mites', 'eye_infection', 'feline_acne', 'feline_leukemia', 'feline_panleukopenia', 'flea_allergy', 'fleas', 'fungal_infection', 'healthy', 'ringworm', 'scabies', 'skin_allergy', 'urinary_tract_infection', 'worm_infection']

def compute_ece(probs, y_true, n_bins=10):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)

def optimize_temperature(probs, y_true):
    eps = 1e-12
    logits = np.log(np.clip(probs, eps, 1.0))
    def nll_obj(t):
        scaled = logits / max(t[0], 0.05)
        exp_s = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        p = exp_s / np.sum(exp_s, axis=1, keepdims=True)
        p = np.clip(p, eps, 1.0 - eps)
        y_oh = np.zeros_like(p)
        y_oh[np.arange(len(y_true)), y_true] = 1.0
        return -float(np.mean(np.sum(y_oh * np.log(p), axis=1)))
    
    res = minimize(nll_obj, [1.0], bounds=[(0.1, 5.0)], method='L-BFGS-B')
    opt_t = float(res.x[0])
    
    scaled_test = logits / opt_t
    exp_test = np.exp(scaled_test - np.max(scaled_test, axis=1, keepdims=True))
    calibrated_probs = exp_test / np.sum(exp_test, axis=1, keepdims=True)
    return opt_t, calibrated_probs

def run_analysis(species, classes):
    print(f"\n{'='*75}\nADVANCED AUDIT & OPTIMIZATION FOR {species.upper()}\n{'='*75}")
    cls2idx = {c: i for i, c in enumerate(classes)}
    
    # Load labels
    val_items = json.load(open(DATA / f"manifest_{species}_val.json"))
    test_items = json.load(open(DATA / f"manifest_{species}_test.json"))
    y_val = np.array([cls2idx[x['label']] for x in val_items if x['label'] in cls2idx and Path(x['path']).exists()])
    y_test = np.array([cls2idx[x['label']] for x in test_items if x['label'] in cls2idx and Path(x['path']).exists()])
    
    # Load feature caches
    c_eff = np.load(DATA / f"cache_{species}_effnetb0.npz")
    c_mob = np.load(DATA / f"cache_{species}_mobilenetv2.npz")
    c_den = np.load(DATA / f"cache_{species}_densenet121.npz")
    
    # Load models
    sp_cap = species.capitalize()
    head_a = tf.keras.models.load_model(MODELS / f"{sp_cap}_EfficientNetB0_candidate_A.keras")
    head_b = tf.keras.models.load_model(MODELS / f"{sp_cap}_MobileNetV2_candidate_B.keras")
    head_c = tf.keras.models.load_model(MODELS / f"{sp_cap}_DenseNet121_candidate_C.keras")
    
    # Predict with candidate heads on val and test
    p_a_val = head_a.predict(c_eff['val'], verbose=0)
    p_a_test = head_a.predict(c_eff['te'], verbose=0)
    
    p_b_val = head_b.predict(c_mob['val'], verbose=0)
    p_b_test = head_b.predict(c_mob['te'], verbose=0)
    
    p_c_val = head_c.predict(c_den['val'], verbose=0)
    p_c_test = head_c.predict(c_den['te'], verbose=0)
    
    # Baseline preds
    p_base_val = np.load(DATA / f"{species}_baseline_val_preds.npy")
    p_base_test = np.load(DATA / f"{species}_baseline_test_preds.npy")
    
    model_names = ["Baseline (EffNetB0)", "Cand A (Deep Head)", "Cand B (MobileNetV2)", "Cand C (DenseNet121)"]
    val_probs_list = [p_base_val, p_a_val, p_b_val, p_c_val]
    test_probs_list = [p_base_test, p_a_test, p_b_test, p_c_test]
    
    # ── 1. DIVERSITY ANALYSIS ──
    print("\n--- 1. Architecture Diversity & Complementary Error Analysis ---")
    correlations = {}
    preds_val = [np.argmax(p, axis=1) for p in val_probs_list]
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            m1, m2 = model_names[i], model_names[j]
            corr = float(np.corrcoef(val_probs_list[i].flatten(), val_probs_list[j].flatten())[0, 1])
            agree = float(np.mean(preds_val[i] == preds_val[j]))
            kappa = float(cohen_kappa_score(preds_val[i], preds_val[j]))
            correlations[f"{m1} vs {m2}"] = {
                "pearson_correlation": corr,
                "prediction_agreement": agree,
                "cohens_kappa": kappa
            }
            print(f"  {m1:<22} vs {m2:<22}: Corr={corr:.4f}, Agreement={agree*100:.2f}%, Kappa={kappa:.4f}")

    # ── 2. STACKING META-LEARNER ──
    print("\n--- 2. Stacking Meta-Learner (Stratified 5-Fold OOF) ---")
    X_val = np.hstack(val_probs_list)
    X_test = np.hstack(test_probs_list)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_val_preds = np.zeros_like(val_probs_list[0])
    
    for tr_idx, oof_idx in skf.split(X_val, y_val):
        clf = LogisticRegression(C=0.5, max_iter=500, random_state=42)
        clf.fit(X_val[tr_idx], y_val[tr_idx])
        oof_val_preds[oof_idx] = clf.predict_proba(X_val[oof_idx])
        
    stacking_val_acc = float(accuracy_score(y_val, np.argmax(oof_val_preds, axis=1)))
    stacking_val_f1 = float(f1_score(y_val, np.argmax(oof_val_preds, axis=1), average='macro'))
    
    # Fit full meta-learner on val to evaluate on test
    final_meta = LogisticRegression(C=0.5, max_iter=500, random_state=42)
    final_meta.fit(X_val, y_val)
    stacking_test_probs = final_meta.predict_proba(X_test)
    stacking_test_acc = float(accuracy_score(y_test, np.argmax(stacking_test_probs, axis=1)))
    stacking_test_f1 = float(f1_score(y_test, np.argmax(stacking_test_probs, axis=1), average='macro'))
    
    print(f"  Stacking Meta-Learner (Validation OOF): Acc={stacking_val_acc*100:.2f}%, MacroF1={stacking_val_f1:.4f}")
    print(f"  Stacking Meta-Learner (Held-Out Test):   Acc={stacking_test_acc*100:.2f}%, MacroF1={stacking_test_f1:.4f}")

    # ── 3. OPTIMAL WEIGHTED ENSEMBLE EVALUATION ──
    weights = [0.3636, 0.0909, 0.2727, 0.2727] if species == 'dog' else [0.2857, 0.2143, 0.2143, 0.2857]
    weighted_val_probs = sum(w * p for w, p in zip(weights, val_probs_list))
    weighted_test_probs = sum(w * p for w, p in zip(weights, test_probs_list))
    
    w_val_acc = float(accuracy_score(y_val, np.argmax(weighted_val_probs, axis=1)))
    w_test_acc = float(accuracy_score(y_test, np.argmax(weighted_test_probs, axis=1)))
    w_test_f1 = float(f1_score(y_test, np.argmax(weighted_test_probs, axis=1), average='macro'))
    print(f"\n--- 3. Weighted Soft-Voting Ensemble ---")
    print(f"  Validation Accuracy: {w_val_acc*100:.2f}%")
    print(f"  Held-Out Test Accuracy: {w_test_acc*100:.2f}%, Macro F1: {w_test_f1:.4f}")

    # ── 4. CONFIDENCE, ENTROPY & CALIBRATION ──
    print("\n--- 4. Confidence, Prediction Entropy & Temperature Scaling ---")
    confidences = np.max(weighted_test_probs, axis=1)
    eps = 1e-12
    entropy = -np.sum(weighted_test_probs * np.log(np.clip(weighted_test_probs, eps, 1.0)), axis=1)
    ece_pre = compute_ece(weighted_test_probs, y_test)
    
    opt_t, cal_test_probs = optimize_temperature(weighted_val_probs, y_val)
    # Apply to test
    logits_test = np.log(np.clip(weighted_test_probs, eps, 1.0))
    cal_logits = logits_test / opt_t
    exp_c = np.exp(cal_logits - np.max(cal_logits, axis=1, keepdims=True))
    cal_test_probs = exp_c / np.sum(exp_c, axis=1, keepdims=True)
    ece_post = compute_ece(cal_test_probs, y_test)
    
    print(f"  Mean Confidence: {float(np.mean(confidences))*100:.2f}% (Median: {float(np.median(confidences))*100:.2f}%)")
    print(f"  Mean Prediction Entropy: {float(np.mean(entropy)):.4f} nats")
    print(f"  Optimal Temperature (T): {opt_t:.3f}")
    print(f"  Expected Calibration Error (Before): {ece_pre*100:.2f}%")
    print(f"  Expected Calibration Error (After):  {ece_post*100:.2f}%")

    # ── 5. ERROR ANALYSIS & CLINICAL PAIRS ──
    print("\n--- 5. Top Dermatological Confusion Pairs ---")
    cm = confusion_matrix(y_test, np.argmax(weighted_test_probs, axis=1))
    confused_pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true_class": classes[i],
                    "predicted_class": classes[j],
                    "count": int(cm[i, j])
                })
    confused_pairs = sorted(confused_pairs, key=lambda x: x["count"], reverse=True)
    for p in confused_pairs[:6]:
        print(f"  {p['true_class']:<22} misclassified as {p['predicted_class']:<22} ({p['count']} occurrences)")

    return {
        "species": species,
        "n_samples_val": len(y_val),
        "n_samples_test": len(y_test),
        "weights": weights,
        "correlations": correlations,
        "stacking": {
            "val_oof_accuracy": stacking_val_acc,
            "val_oof_macro_f1": stacking_val_f1,
            "test_accuracy": stacking_test_acc,
            "test_macro_f1": stacking_test_f1
        },
        "weighted_ensemble": {
            "val_accuracy": w_val_acc,
            "test_accuracy": w_test_acc,
            "test_macro_f1": w_test_f1
        },
        "calibration": {
            "mean_confidence": float(np.mean(confidences)),
            "median_confidence": float(np.median(confidences)),
            "mean_entropy": float(np.mean(entropy)),
            "optimal_temperature": opt_t,
            "ece_before": ece_pre,
            "ece_after": ece_post
        },
        "top_confusion_pairs": confused_pairs[:10],
        "confusion_matrix": cm.tolist()
    }

def main():
    dog_results = run_analysis("dog", DOG_CLASSES)
    cat_results = run_analysis("cat", CAT_CLASSES)
    
    output = {"dog": dog_results, "cat": cat_results}
    with open(DATA / "advanced_model_analysis.json", "w", encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    print(f"\n[DONE] Saved Advanced Analysis to: data/advanced_model_analysis.json")

if __name__ == "__main__":
    main()
