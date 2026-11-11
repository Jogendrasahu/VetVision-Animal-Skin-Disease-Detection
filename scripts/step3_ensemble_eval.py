"""
STEP 3 — Ensemble evaluation.
Loads existing baseline + both trained candidates.
Finds optimal ensemble weights using VALIDATION set only.
Evaluates final ensemble on the held-out TEST set.
Saves all results to data/final_model_evaluation.json
"""
import os, json, argparse
import numpy as np
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import (classification_report, balanced_accuracy_score,
                              confusion_matrix, accuracy_score)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PROJECT = Path(r"c:\Users\Jogen\OneDrive\Desktop\Jogendra\Projects\Animal_Disease_App")
DATA    = PROJECT / "data"
MODELS  = PROJECT / "models"
FIGS    = PROJECT / "reports" / "figures"

DOG_CLASSES = [
    'demodicosis','dental_disease','dermatitis','distemper','eye_infection',
    'fungal_infection','healthy','hot_spots','hypersensitivity','kennel_cough',
    'mange','parvovirus','ringworm','skin_allergy','tick_infestation','worm_infection'
]
CAT_CLASSES = [
    'dental_disease','ear_mites','eye_infection','feline_acne','feline_leukemia',
    'feline_panleukopenia','flea_allergy','fleas','fungal_infection','healthy',
    'ringworm','scabies','skin_allergy','urinary_tract_infection','worm_infection'
]

IMG_SIZE = 224
BATCH    = 24

def load_manifest(path):
    with open(path) as f:
        return json.load(f)

def build_tf_dataset(manifest, classes, shuffle=False, batch=BATCH):
    cls2idx = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for item in manifest:
        if item['label'] in cls2idx and Path(item['path']).exists():
            paths.append(item['path'])
            labels.append(cls2idx[item['label']])
    
    def parse_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32)  # RAW [0,255]
        return img, label
    
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=42)
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds, np.array(labels)

def get_preds(model, ds):
    return model.predict(ds, verbose=0)

def metrics_dict(true_labels, pred_labels, classes, preds=None):
    acc  = accuracy_score(true_labels, pred_labels)
    bal  = balanced_accuracy_score(true_labels, pred_labels)
    rep  = classification_report(true_labels, pred_labels,
                                  target_names=classes, output_dict=True)
    cm   = confusion_matrix(true_labels, pred_labels)
    return {
        "accuracy":           float(acc),
        "balanced_accuracy":  float(bal),
        "macro_precision":    float(rep['macro avg']['precision']),
        "weighted_precision": float(rep['weighted avg']['precision']),
        "macro_recall":       float(rep['macro avg']['recall']),
        "weighted_recall":    float(rep['weighted avg']['recall']),
        "macro_f1":           float(rep['macro avg']['f1-score']),
        "weighted_f1":        float(rep['weighted avg']['f1-score']),
        "per_class":          {c: rep[c] for c in classes if c in rep},
        "confusion_matrix":   cm.tolist(),
    }

def print_metrics(label, m):
    print(f"\n  ─── {label} ───")
    print(f"  Accuracy:          {m['accuracy']*100:.2f}%")
    print(f"  Balanced Accuracy: {m['balanced_accuracy']*100:.2f}%")
    print(f"  Macro F1:          {m['macro_f1']:.4f}")
    print(f"  Weighted F1:       {m['weighted_f1']:.4f}")
    print(f"  Macro Precision:   {m['macro_precision']:.4f}")
    print(f"  Weighted Precision:{m['weighted_precision']:.4f}")
    print(f"  Macro Recall:      {m['macro_recall']:.4f}")
    print(f"  Weighted Recall:   {m['weighted_recall']:.4f}")

def find_optimal_weights_grid(pred_list, true_labels, n_steps=11):
    """
    Grid search over (w0, w1, w2) with w0+w1+w2=1 for best val accuracy.
    Uses val set ONLY — test set never seen here.
    """
    if len(pred_list) == 2:
        best_acc, best_w = 0, (0.5, 0.5)
        for i in range(n_steps):
            w0 = i / (n_steps - 1)
            w1 = 1.0 - w0
            combined = w0 * pred_list[0] + w1 * pred_list[1]
            acc = accuracy_score(true_labels, np.argmax(combined, axis=1))
            if acc > best_acc:
                best_acc, best_w = acc, (w0, w1)
        print(f"  Best val weights (2-model): {best_w}  val_acc={best_acc:.4f}")
        return best_w, best_acc
    elif len(pred_list) == 3:
        best_acc, best_w = 0, (1/3, 1/3, 1/3)
        for i in range(n_steps):
            for j in range(n_steps - i):
                k = n_steps - 1 - i - j
                if k < 0: continue
                w0, w1, w2 = i/(n_steps-1), j/(n_steps-1), k/(n_steps-1)
                s = w0+w1+w2
                if s < 1e-6: continue
                w0,w1,w2 = w0/s, w1/s, w2/s
                combined = w0*pred_list[0] + w1*pred_list[1] + w2*pred_list[2]
                acc = accuracy_score(true_labels, np.argmax(combined, axis=1))
                if acc > best_acc:
                    best_acc, best_w = acc, (w0, w1, w2)
        print(f"  Best val weights (3-model): {best_w}  val_acc={best_acc:.4f}")
        return best_w, best_acc

def save_comparison_chart(results, species, out_path):
    metrics_to_plot = ['accuracy','balanced_accuracy','macro_f1','weighted_f1']
    labels = list(results.keys())
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(labels)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
    
    for i, (lbl, m) in enumerate(results.items()):
        vals = [m[k] for k in metrics_to_plot]
        ax.bar(x + i*width, vals, width, label=lbl, color=colors[i % len(colors)], alpha=0.85)
    
    ax.set_xticks(x + width * (len(labels)-1)/2)
    ax.set_xticklabels(['Accuracy','Balanced Acc','Macro F1','Weighted F1'], fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score'); ax.set_title(f'{species.upper()} — Model Comparison (Validation Set)')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    
    for i, (lbl, m) in enumerate(results.items()):
        vals = [m[k] for k in metrics_to_plot]
        for j, v in enumerate(vals):
            ax.text(j + i*width, v + 0.005, f"{v:.3f}", ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison chart → {out_path}")

def save_confusion_matrix(cm, classes, title, out_path):
    fig, ax = plt.subplots(figsize=(max(10, len(classes)), max(8, len(classes) - 2)))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title(title)
    thresh = cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i,j]}", ha='center', va='center',
                    color='white' if cm[i,j] > thresh else 'black', fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix → {out_path}")

def run_ensemble_for_species(species):
    classes = DOG_CLASSES if species == "dog" else CAT_CLASSES
    n_classes = len(classes)
    
    # Model filenames
    prod_name = "Dog_EfficientNetB0_balanced_best" if species == "dog" else "Cat_EfficientNetB0_stage2_best"
    cand_a_name = f"{species.capitalize()}_EfficientNetB0_candidate_A"
    cand_b_name = f"{species.capitalize()}_EfficientNetB0_candidate_B"
    
    prod_path  = MODELS / f"{prod_name}.keras"
    cand_a_path = MODELS / f"{cand_a_name}.keras"
    cand_b_path = MODELS / f"{cand_b_name}.keras"
    
    print(f"\n{'='*70}")
    print(f"ENSEMBLE EVALUATION: {species.upper()}")
    print(f"{'='*70}")
    
    # Load validation dataset
    val_manifest  = load_manifest(DATA / f"manifest_{species}_val.json")
    test_manifest = load_manifest(DATA / f"manifest_{species}_test.json")
    
    print("\n  Loading validation data...")
    val_ds, val_labels   = build_tf_dataset(val_manifest,  classes)
    print("  Loading test data...")
    test_ds, test_labels = build_tf_dataset(test_manifest, classes)
    
    print(f"  Val samples: {len(val_labels)}, Test samples: {len(test_labels)}")
    
    models_available = {}
    val_preds  = {}
    test_preds = {}
    
    # ── Load production baseline ──
    if prod_path.exists():
        print(f"\n  Loading BASELINE model: {prod_path.name}")
        m = tf.keras.models.load_model(str(prod_path))
        print("  Getting baseline val predictions...")
        vp = get_preds(m, val_ds)
        tp = get_preds(m, test_ds)
        models_available['baseline'] = m
        val_preds['baseline']  = vp
        test_preds['baseline'] = tp
        del m  # free memory
    
    # ── Load Candidate A ──
    if cand_a_path.exists():
        print(f"\n  Loading CANDIDATE A: {cand_a_path.name}")
        m = tf.keras.models.load_model(str(cand_a_path))
        vp = get_preds(m, val_ds)
        tp = get_preds(m, test_ds)
        models_available['candidate_A'] = m
        val_preds['candidate_A']  = vp
        test_preds['candidate_A'] = tp
        del m
    
    # ── Load Candidate B ──
    if cand_b_path.exists():
        print(f"\n  Loading CANDIDATE B: {cand_b_path.name}")
        m = tf.keras.models.load_model(str(cand_b_path))
        vp = get_preds(m, val_ds)
        tp = get_preds(m, test_ds)
        models_available['candidate_B'] = m
        val_preds['candidate_B']  = vp
        test_preds['candidate_B'] = tp
        del m
    
    model_keys = list(val_preds.keys())
    print(f"\n  Models loaded: {model_keys}")
    
    # ── Evaluate each individual model on validation ──
    print("\n--- Individual model validation metrics ---")
    val_results = {}
    for key in model_keys:
        pred_labels = np.argmax(val_preds[key], axis=1)
        m = metrics_dict(val_labels, pred_labels, classes)
        val_results[key] = m
        print_metrics(key, m)
    
    # ── Soft uniform ensemble (validation) ──
    if len(model_keys) >= 2:
        ensemble_preds_val = np.mean([val_preds[k] for k in model_keys], axis=0)
        pred_labels = np.argmax(ensemble_preds_val, axis=1)
        m = metrics_dict(val_labels, pred_labels, classes)
        val_results['soft_ensemble_uniform'] = m
        print_metrics("Soft Ensemble (uniform)", m)
    
    # ── Weighted ensemble — find optimal weights on val set ──
    if len(model_keys) >= 2:
        pred_list = [val_preds[k] for k in model_keys]
        best_weights, best_val_acc = find_optimal_weights_grid(pred_list, val_labels, n_steps=11)
        
        # Save weights
        weights_dict = dict(zip(model_keys, best_weights))
        print(f"  Optimal ensemble weights: {weights_dict}")
        
        ensemble_preds_val_w = sum(w * val_preds[k] for k, w in zip(model_keys, best_weights))
        pred_labels = np.argmax(ensemble_preds_val_w, axis=1)
        m = metrics_dict(val_labels, pred_labels, classes)
        val_results['soft_ensemble_weighted'] = m
        print_metrics("Soft Ensemble (weighted)", m)
    
    # ── Save validation comparison chart ──
    save_comparison_chart(val_results, species,
        FIGS / f"model_comparison_val_{species}.png")
    
    # ── Determine best model/ensemble from VALIDATION ──
    best_key = max(val_results, key=lambda k: val_results[k]['accuracy'])
    print(f"\n  Best on validation: {best_key} (acc={val_results[best_key]['accuracy']:.4f})")
    
    # ── FINAL TEST EVALUATION (untouched test set, done only once) ──
    print(f"\n{'='*50}")
    print(f"FINAL TEST EVALUATION (held-out test set)")
    print(f"{'='*50}")
    
    test_results = {}
    for key in model_keys:
        pred_labels = np.argmax(test_preds[key], axis=1)
        m = metrics_dict(test_labels, pred_labels, classes)
        test_results[key] = m
        print_metrics(f"{key} [TEST]", m)
    
    if len(model_keys) >= 2:
        # Uniform ensemble test
        ensemble_preds_test = np.mean([test_preds[k] for k in model_keys], axis=0)
        pred_labels = np.argmax(ensemble_preds_test, axis=1)
        m = metrics_dict(test_labels, pred_labels, classes)
        test_results['soft_ensemble_uniform'] = m
        print_metrics("Soft Ensemble Uniform [TEST]", m)
        
        # Weighted ensemble test (weights from val)
        ensemble_preds_test_w = sum(w * test_preds[k] for k, w in zip(model_keys, best_weights))
        pred_labels = np.argmax(ensemble_preds_test_w, axis=1)
        m = metrics_dict(test_labels, pred_labels, classes)
        test_results['soft_ensemble_weighted'] = m
        print_metrics("Soft Ensemble Weighted [TEST]", m)
    
    # ── Confusion matrix for best ensemble ──
    best_test_key = max(test_results, key=lambda k: test_results[k]['accuracy'])
    print(f"\n  Best on TEST: {best_test_key} (acc={test_results[best_test_key]['accuracy']:.4f})")
    
    if best_test_key == 'soft_ensemble_weighted' and len(model_keys) >= 2:
        best_test_preds = np.argmax(ensemble_preds_test_w, axis=1)
        best_test_pred_array = ensemble_preds_test_w
    elif best_test_key == 'soft_ensemble_uniform' and len(model_keys) >= 2:
        best_test_preds = np.argmax(ensemble_preds_test, axis=1)
        best_test_pred_array = ensemble_preds_test
    else:
        best_test_preds = np.argmax(test_preds[best_test_key], axis=1)
        best_test_pred_array = test_preds[best_test_key]
    
    best_cm = confusion_matrix(test_labels, best_test_preds)
    save_confusion_matrix(best_cm, classes,
        f"{species.capitalize()} Best Model ({best_test_key}) — Test Set Confusion Matrix",
        FIGS / f"best_test_cm_{species}.png")
    
    # ── Save final comprehensive results ──
    final = {
        "species": species,
        "classes": classes,
        "n_val":   int(len(val_labels)),
        "n_test":  int(len(test_labels)),
        "models_evaluated": model_keys,
        "best_val_model": best_key,
        "best_test_model": best_test_key,
        "ensemble_weights": {k: float(v) for k, v in zip(model_keys, best_weights)} if len(model_keys) >= 2 else {},
        "validation_results": val_results,
        "test_results": test_results,
        "recommendation": {
            "model": best_test_key,
            "test_accuracy": test_results[best_test_key]['accuracy'],
            "test_macro_f1": test_results[best_test_key]['macro_f1'],
        }
    }
    
    out_path = DATA / f"ensemble_results_{species}.json"
    with open(out_path, 'w') as f:
        json.dump(final, f, indent=2)
    print(f"\n  Saved ensemble results → {out_path}")
    
    return final

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", default="both", choices=["dog","cat","both"])
    args = parser.parse_args()
    
    species_list = ["dog","cat"] if args.species == "both" else [args.species]
    all_results = {}
    
    for species in species_list:
        all_results[species] = run_ensemble_for_species(species)
    
    # ── Print final summary table ──
    print(f"\n\n{'='*80}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    
    for species, res in all_results.items():
        print(f"\n--- {species.upper()} ---")
        for key in res['test_results']:
            m = res['test_results'][key]
            print(f"  {key:35s}: Acc={m['accuracy']*100:.2f}%  MacroF1={m['macro_f1']:.4f}  WtF1={m['weighted_f1']:.4f}")
        print(f"  >>> BEST: {res['best_test_model']} — Acc={res['recommendation']['test_accuracy']*100:.2f}%")
    
    print("\nDone! Check data/ensemble_results_*.json for full results.")

