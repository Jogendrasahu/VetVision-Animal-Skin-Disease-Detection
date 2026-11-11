"""
Comprehensive, Leakage-Free, Multi-Architecture Training & Ensemble Pipeline
for VetVision AI (Dog and Cat Disease Classifiers).

Species:
  - Dog: 16 classes
  - Cat: 15 classes

Architectures:
  1. Baseline: Existing production model
  2. Candidate A: EfficientNetB0 (Deep Swish Head + Cosine Decay + Label Smoothing)
  3. Candidate B: MobileNetV2 (Inverted Residuals + Dropout Head + AdamW)
  4. Candidate C: DenseNet121 (Dense Connectivity + Heavy Regularization)

Ensemble:
  - Soft Uniform Average
  - Soft Weighted Average (Weights tuned strictly on VALIDATION set)
  - End-to-End Unified Keras Ensemble Model Export (RAW [0, 255] input compatible)

Outputs:
  - Saved models in models/
  - Metrics in data/
  - Authentic figures in reports/figures/
  - Experiment log in reports/model_improvement_experiments.csv
"""
import os, sys, json, time, csv, shutil
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import (
    classification_report, accuracy_score, balanced_accuracy_score, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Environmental settings for reproducibility and stability
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.random.set_seed(42)
np.random.seed(42)

PROJECT = Path(r"c:\Users\Jogen\OneDrive\Desktop\Jogendra\Projects\Animal_Disease_App")
DATA = PROJECT / "data"
MODELS = PROJECT / "models"
REPORTS = PROJECT / "reports"
FIGS = REPORTS / "figures"

FIGS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

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
BATCH_SIZE = 32

def parse_image_raw(path):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return tf.cast(img, tf.float32)

def extract_features_for_manifest(manifest_path, classes, backbone_fn, preprocess_layer=None):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    cls2idx = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for item in manifest:
        if item['label'] in cls2idx and Path(item['path']).exists():
            paths.append(item['path'])
            labels.append(cls2idx[item['label']])

    def _mapper(p):
        img = parse_image_raw(p)
        if preprocess_layer is not None:
            img = preprocess_layer(img)
        return img

    ds = tf.data.Dataset.from_tensor_slices(paths).map(_mapper, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE)
    features = backbone_fn.predict(ds, verbose=0)
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)

def compute_weights_dict(labels, n_classes):
    cw = compute_class_weight('balanced', classes=np.arange(n_classes), y=labels)
    return dict(enumerate(cw))

def calc_all_metrics(y_true, y_pred_probs, classes):
    y_pred = np.argmax(y_pred_probs, axis=1)
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    rep = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_precision": float(rep['macro avg']['precision']),
        "weighted_precision": float(rep['weighted avg']['precision']),
        "macro_recall": float(rep['macro avg']['recall']),
        "weighted_recall": float(rep['weighted avg']['recall']),
        "macro_f1": float(rep['macro avg']['f1-score']),
        "weighted_f1": float(rep['weighted avg']['f1-score']),
        "per_class": {c: rep[c] for c in classes if c in rep},
        "cm": cm.tolist()
    }

def plot_and_save_cm(cm, classes, title, filepath):
    fig, ax = plt.subplots(figsize=(max(9, len(classes)*0.6), max(7, len(classes)*0.55)))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=10)
    ax.set_ylabel('True Label', fontweight='bold', fontsize=10)
    ax.set_title(title, fontweight='bold', fontsize=11, pad=12)
    thresh = cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=7,
                    color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.savefig(filepath, dpi=160, bbox_inches='tight')
    plt.close()

def plot_training_curves(history_dict, title, filepath):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    
    epochs = range(1, len(history_dict['accuracy']) + 1)
    ax1.plot(epochs, history_dict['accuracy'], 'b-', lw=2, label='Train Accuracy')
    ax1.plot(epochs, history_dict['val_accuracy'], 'r-', lw=2, label='Val Accuracy')
    ax1.set_title('Accuracy vs Epoch', fontsize=11)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')
    
    ax2.plot(epochs, history_dict['loss'], 'b-', lw=2, label='Train Loss')
    ax2.plot(epochs, history_dict['val_loss'], 'r-', lw=2, label='Val Loss')
    ax2.set_title('Loss vs Epoch', fontsize=11)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=160, bbox_inches='tight')
    plt.close()

def plot_per_class_f1(per_class_metrics, classes, title, filepath):
    fig, ax = plt.subplots(figsize=(10, max(6, len(classes)*0.35)))
    f1_scores = [per_class_metrics[c]['f1-score'] for c in classes]
    y_pos = np.arange(len(classes))
    colors = ['#2ca02c' if f >= 0.85 else '#1f77b4' if f >= 0.70 else '#d62728' for f in f1_scores]
    bars = ax.barh(y_pos, f1_scores, color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('F1-Score', fontweight='bold', fontsize=10)
    ax.set_title(title, fontweight='bold', fontsize=11, pad=10)
    ax.set_xlim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='x')
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height()/2, f"{w:.3f}", va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(filepath, dpi=160, bbox_inches='tight')
    plt.close()

def run_pipeline_for_species(species):
    classes = DOG_CLASSES if species == "dog" else CAT_CLASSES
    n_classes = len(classes)
    sp_cap = species.capitalize()
    
    print("\n" + "="*80)
    print(f"STARTING COMPREHENSIVE EXPERIMENT PIPELINE FOR: {species.upper()} ({n_classes} Classes)")
    print("="*80)
    
    train_manifest = DATA / f"manifest_{species}_train.json"
    val_manifest = DATA / f"manifest_{species}_val.json"
    test_manifest = DATA / f"manifest_{species}_test.json"
    
    # ── 1. Load Baseline Predictions ──
    print("\n[Step 1] Loading verified baseline predictions...")
    with open(DATA / f"baseline_{species}_eval.json", 'r', encoding='utf-8') as f:
        base_eval = json.load(f)
    base_val_preds = np.load(str(DATA / f"{species}_baseline_val_preds.npy"))
    base_test_preds = np.load(str(DATA / f"{species}_baseline_test_preds.npy"))
    
    # Get ground truth labels
    with open(val_manifest) as f: val_m = json.load(f)
    cls2idx = {c: i for i, c in enumerate(classes)}
    y_val = np.array([cls2idx[x['label']] for x in val_m if x['label'] in cls2idx and Path(x['path']).exists()])
    
    with open(test_manifest) as f: test_m = json.load(f)
    y_test = np.array([cls2idx[x['label']] for x in test_m if x['label'] in cls2idx and Path(x['path']).exists()])
    
    print(f"  Validation samples: {len(y_val)} | Test samples: {len(y_test)}")
    baseline_val_m = calc_all_metrics(y_val, base_val_preds, classes)
    baseline_test_m = calc_all_metrics(y_test, base_test_preds, classes)
    print(f"  Baseline Val Accuracy : {baseline_val_m['accuracy']*100:.2f}% (Macro F1: {baseline_val_m['macro_f1']:.4f})")
    print(f"  Baseline Test Accuracy: {baseline_test_m['accuracy']*100:.2f}% (Macro F1: {baseline_test_m['macro_f1']:.4f})")
    
    # Storage for candidates
    candidates = {
        "Baseline": {
            "val_preds": base_val_preds,
            "test_preds": base_test_preds,
            "val_metrics": baseline_val_m,
            "test_metrics": baseline_test_m,
            "arch": "EfficientNetB0 (Balanced Best)",
            "model_path": str(MODELS / f"{'Dog_EfficientNetB0_balanced_best' if species=='dog' else 'Cat_EfficientNetB0_stage2_best'}.keras")
        }
    }
    
    class_weights = compute_weights_dict(
        np.array([cls2idx[x['label']] for x in json.load(open(train_manifest)) if x['label'] in cls2idx and Path(x['path']).exists()]),
        n_classes
    )
    
    # ── 2. Candidate A: EfficientNetB0 (Deep Swish Head + Cosine Decay + Label Smoothing) ──
    print("\n[Step 2] Training Candidate A: EfficientNetB0 Deep Swish Head...")
    eff_cache = DATA / f"cache_{species}_effnetb0.npz"
    if eff_cache.exists():
        print(f"  Loading cached EfficientNetB0 features from {eff_cache.name}...")
        cdata = np.load(str(eff_cache))
        X_tr_eff, X_val_eff, X_te_eff = cdata['tr'], cdata['val'], cdata['te']
    else:
        print("  Extracting EfficientNetB0 pooled features...")
        eff_bb = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg')
        X_tr_eff, _ = extract_features_for_manifest(train_manifest, classes, eff_bb)
        X_val_eff, _ = extract_features_for_manifest(val_manifest, classes, eff_bb)
        X_te_eff, _ = extract_features_for_manifest(test_manifest, classes, eff_bb)
        np.savez_compressed(str(eff_cache), tr=X_tr_eff, val=X_val_eff, te=X_te_eff)
        del eff_bb
    
    # Train Candidate A Head
    tf.keras.utils.set_random_seed(42)
    head_a = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1280,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(512, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ], name="Head_Candidate_A")
    
    head_a.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.06),
        metrics=['accuracy']
    )
    
    y_tr_1hot = tf.keras.utils.to_categorical(np.array([cls2idx[x['label']] for x in json.load(open(train_manifest)) if x['label'] in cls2idx and Path(x['path']).exists()]), n_classes)
    y_val_1hot = tf.keras.utils.to_categorical(y_val, n_classes)
    
    ha_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
    ]
    h_a = head_a.fit(X_tr_eff, y_tr_1hot, validation_data=(X_val_eff, y_val_1hot),
                     epochs=40, batch_size=32, class_weight=class_weights,
                     callbacks=ha_callbacks, verbose=0)
    
    val_preds_a = head_a.predict(X_val_eff, verbose=0)
    test_preds_a = head_a.predict(X_te_eff, verbose=0)
    cand_a_val_m = calc_all_metrics(y_val, val_preds_a, classes)
    cand_a_test_m = calc_all_metrics(y_test, test_preds_a, classes)
    print(f"  Candidate A Val Accuracy : {cand_a_val_m['accuracy']*100:.2f}% (Macro F1: {cand_a_val_m['macro_f1']:.4f})")
    print(f"  Candidate A Test Accuracy: {cand_a_test_m['accuracy']*100:.2f}% (Macro F1: {cand_a_test_m['macro_f1']:.4f})")
    
    # Assemble full end-to-end Keras model for Candidate A (RAW [0, 255] input)
    inputs_a = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="raw_image_input")
    bb_a = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg')
    bb_a.trainable = False
    feats_a = bb_a(inputs_a)
    out_a = head_a(feats_a)
    full_model_a = tf.keras.Model(inputs=inputs_a, outputs=out_a, name=f"{sp_cap}_Candidate_A_EffNetB0")
    cand_a_path = MODELS / f"{sp_cap}_EfficientNetB0_candidate_A.keras"
    full_model_a.save(str(cand_a_path))
    
    plot_training_curves(h_a.history, f"{sp_cap} Candidate A (EfficientNetB0) Training Curves", FIGS / f"curves_{species}_candidate_A.png")
    
    candidates["Candidate A (EffNetB0 Deep Head)"] = {
        "val_preds": val_preds_a, "test_preds": test_preds_a,
        "val_metrics": cand_a_val_m, "test_metrics": cand_a_test_m,
        "arch": "EfficientNetB0 (Deep Swish Head + AdamW)",
        "model_path": str(cand_a_path), "history": h_a.history
    }
    
    # ── 3. Candidate B: MobileNetV2 (Inverted Residuals + Dropout Head) ──
    print("\n[Step 3] Training Candidate B: MobileNetV2 Inverted Residual Backbone...")
    mob_cache = DATA / f"cache_{species}_mobilenetv2.npz"
    mob_prep = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1.0)
    if mob_cache.exists():
        print(f"  Loading cached MobileNetV2 features from {mob_cache.name}...")
        cdata = np.load(str(mob_cache))
        X_tr_mob, X_val_mob, X_te_mob = cdata['tr'], cdata['val'], cdata['te']
    else:
        print("  Extracting MobileNetV2 pooled features...")
        mob_bb = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg')
        X_tr_mob, _ = extract_features_for_manifest(train_manifest, classes, mob_bb, preprocess_layer=mob_prep)
        X_val_mob, _ = extract_features_for_manifest(val_manifest, classes, mob_bb, preprocess_layer=mob_prep)
        X_te_mob, _ = extract_features_for_manifest(test_manifest, classes, mob_bb, preprocess_layer=mob_prep)
        np.savez_compressed(str(mob_cache), tr=X_tr_mob, val=X_val_mob, te=X_te_mob)
        del mob_bb

    tf.keras.utils.set_random_seed(7)
    head_b = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1280,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(384, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ], name="Head_Candidate_B")

    head_b.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-3, weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )

    hb_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
    ]
    h_b = head_b.fit(X_tr_mob, y_tr_1hot, validation_data=(X_val_mob, y_val_1hot),
                     epochs=40, batch_size=32, class_weight=class_weights,
                     callbacks=hb_callbacks, verbose=0)

    val_preds_b = head_b.predict(X_val_mob, verbose=0)
    test_preds_b = head_b.predict(X_te_mob, verbose=0)
    cand_b_val_m = calc_all_metrics(y_val, val_preds_b, classes)
    cand_b_test_m = calc_all_metrics(y_test, test_preds_b, classes)
    print(f"  Candidate B Val Accuracy : {cand_b_val_m['accuracy']*100:.2f}% (Macro F1: {cand_b_val_m['macro_f1']:.4f})")
    print(f"  Candidate B Test Accuracy: {cand_b_test_m['accuracy']*100:.2f}% (Macro F1: {cand_b_test_m['macro_f1']:.4f})")

    # Assemble full end-to-end Keras model for Candidate B (RAW [0, 255] input)
    inputs_b = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="raw_image_input")
    prep_b = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1.0)(inputs_b)
    bb_b = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg')
    bb_b.trainable = False
    feats_b = bb_b(prep_b)
    out_b = head_b(feats_b)
    full_model_b = tf.keras.Model(inputs=inputs_b, outputs=out_b, name=f"{sp_cap}_Candidate_B_MobileNetV2")
    cand_b_path = MODELS / f"{sp_cap}_MobileNetV2_candidate_B.keras"
    full_model_b.save(str(cand_b_path))

    plot_training_curves(h_b.history, f"{sp_cap} Candidate B (MobileNetV2) Training Curves", FIGS / f"curves_{species}_candidate_B.png")

    candidates["Candidate B (MobileNetV2)"] = {
        "val_preds": val_preds_b, "test_preds": test_preds_b,
        "val_metrics": cand_b_val_m, "test_metrics": cand_b_test_m,
        "arch": "MobileNetV2 (Inverted Residuals + AdamW)",
        "model_path": str(cand_b_path), "history": h_b.history
    }

    # ── 4. Candidate C: DenseNet121 (Dense Connectivity + Heavy Regularization) ──
    print("\n[Step 4] Training Candidate C: DenseNet121 Dense Connectivity Backbone...")
    dense_cache = DATA / f"cache_{species}_densenet121.npz"
    dense_prep = tf.keras.layers.Rescaling(scale=1./255.0)
    if dense_cache.exists():
        print(f"  Loading cached DenseNet121 features from {dense_cache.name}...")
        cdata = np.load(str(dense_cache))
        X_tr_dense, X_val_dense, X_te_dense = cdata['tr'], cdata['val'], cdata['te']
    else:
        print("  Extracting DenseNet121 pooled features...")
        dense_bb = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg')
        X_tr_dense, _ = extract_features_for_manifest(train_manifest, classes, dense_bb, preprocess_layer=dense_prep)
        X_val_dense, _ = extract_features_for_manifest(val_manifest, classes, dense_bb, preprocess_layer=dense_prep)
        X_te_dense, _ = extract_features_for_manifest(test_manifest, classes, dense_bb, preprocess_layer=dense_prep)
        np.savez_compressed(str(dense_cache), tr=X_tr_dense, val=X_val_dense, te=X_te_dense)
        del dense_bb

    tf.keras.utils.set_random_seed(99)
    head_c = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ], name="Head_Candidate_C")

    head_c.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=8e-4, weight_decay=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )

    hc_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
    ]
    h_c = head_c.fit(X_tr_dense, y_tr_1hot, validation_data=(X_val_dense, y_val_1hot),
                     epochs=40, batch_size=32, class_weight=class_weights,
                     callbacks=hc_callbacks, verbose=0)

    val_preds_c = head_c.predict(X_val_dense, verbose=0)
    test_preds_c = head_c.predict(X_te_dense, verbose=0)
    cand_c_val_m = calc_all_metrics(y_val, val_preds_c, classes)
    cand_c_test_m = calc_all_metrics(y_test, test_preds_c, classes)
    print(f"  Candidate C Val Accuracy : {cand_c_val_m['accuracy']*100:.2f}% (Macro F1: {cand_c_val_m['macro_f1']:.4f})")
    print(f"  Candidate C Test Accuracy: {cand_c_test_m['accuracy']*100:.2f}% (Macro F1: {cand_c_test_m['macro_f1']:.4f})")

    # Assemble full end-to-end Keras model for Candidate C (RAW [0, 255] input)
    inputs_c = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="raw_image_input")
    prep_c = tf.keras.layers.Rescaling(scale=1./255.0)(inputs_c)
    bb_c = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling='avg')
    bb_c.trainable = False
    feats_c = bb_c(prep_c)
    out_c = head_c(feats_c)
    full_model_c = tf.keras.Model(inputs=inputs_c, outputs=out_c, name=f"{sp_cap}_Candidate_C_DenseNet121")
    cand_c_path = MODELS / f"{sp_cap}_DenseNet121_candidate_C.keras"
    full_model_c.save(str(cand_c_path))

    plot_training_curves(h_c.history, f"{sp_cap} Candidate C (DenseNet121) Training Curves", FIGS / f"curves_{species}_candidate_C.png")

    candidates["Candidate C (DenseNet121)"] = {
        "val_preds": val_preds_c, "test_preds": test_preds_c,
        "val_metrics": cand_c_val_m, "test_metrics": cand_c_test_m,
        "arch": "DenseNet121 (Dense Connectivity + AdamW)",
        "model_path": str(cand_c_path), "history": h_c.history
    }

    # ── 5. Ensemble Experiments ──
    print("\n" + "-"*60)
    print(f"[Step 5] Building Diverse Ensembles for {species.upper()}...")
    print("-"*60)
    
    # Names of models in ensemble
    ens_models = ["Baseline", "Candidate A (EffNetB0 Deep Head)", "Candidate B (MobileNetV2)", "Candidate C (DenseNet121)"]
    val_pred_list = [candidates[m]["val_preds"] for m in ens_models]
    test_pred_list = [candidates[m]["test_preds"] for m in ens_models]
    
    # 5.1 Soft Uniform Ensemble (Equal 25% weight)
    val_soft_uniform = np.mean(val_pred_list, axis=0)
    test_soft_uniform = np.mean(test_pred_list, axis=0)
    ens_uniform_val_m = calc_all_metrics(y_val, val_soft_uniform, classes)
    ens_uniform_test_m = calc_all_metrics(y_test, test_soft_uniform, classes)
    print(f"  Uniform Ensemble Val Accuracy : {ens_uniform_val_m['accuracy']*100:.2f}% (Macro F1: {ens_uniform_val_m['macro_f1']:.4f})")
    print(f"  Uniform Ensemble Test Accuracy: {ens_uniform_test_m['accuracy']*100:.2f}% (Macro F1: {ens_uniform_test_m['macro_f1']:.4f})")
    
    candidates["Soft Ensemble (Uniform)"] = {
        "val_preds": val_soft_uniform, "test_preds": test_soft_uniform,
        "val_metrics": ens_uniform_val_m, "test_metrics": ens_uniform_test_m,
        "arch": "Ensemble (Equal Weight: Baseline + EffNet + MobileNet + DenseNet)",
        "weights": [0.25, 0.25, 0.25, 0.25]
    }
    
    # 5.2 Weighted Soft Ensemble (Grid search on VALIDATION SET ONLY)
    print("\n  Optimizing ensemble weights using VALIDATION SET ONLY (grid search)...")
    best_w = None
    best_val_score = -1
    
    # Search over diverse weight combinations (steps of 0.1)
    for w0 in np.linspace(0.1, 0.5, 5):
        for w1 in np.linspace(0.1, 0.5, 5):
            for w2 in np.linspace(0.1, 0.5, 5):
                for w3 in np.linspace(0.1, 0.5, 5):
                    weights = np.array([w0, w1, w2, w3])
                    weights = weights / np.sum(weights)
                    weighted_val = sum(w * p for w, p in zip(weights, val_pred_list))
                    val_acc = accuracy_score(y_val, np.argmax(weighted_val, axis=1))
                    if val_acc > best_val_score:
                        best_val_score = val_acc
                        best_w = weights
    
    print(f"  Optimal Validation Weights found:")
    for name, w in zip(ens_models, best_w):
        print(f"    - {name:35s}: {w*100:.1f}%")
    print(f"  Best Validation Accuracy with Weighted Ensemble: {best_val_score*100:.2f}%")
    
    val_soft_weighted = sum(w * p for w, p in zip(best_w, val_pred_list))
    test_soft_weighted = sum(w * p for w, p in zip(best_w, test_pred_list))
    ens_weighted_val_m = calc_all_metrics(y_val, val_soft_weighted, classes)
    ens_weighted_test_m = calc_all_metrics(y_test, test_soft_weighted, classes)
    
    print(f"\n  >>> Weighted Ensemble Test Accuracy: {ens_weighted_test_m['accuracy']*100:.2f}% (Macro F1: {ens_weighted_test_m['macro_f1']:.4f})")
    
    candidates["Soft Ensemble (Weighted)"] = {
        "val_preds": val_soft_weighted, "test_preds": test_soft_weighted,
        "val_metrics": ens_weighted_val_m, "test_metrics": ens_weighted_test_m,
        "arch": f"Weighted Ensemble ({', '.join([f'{m.split()[0]}:{w:.2f}' for m, w in zip(ens_models, best_w)])})",
        "weights": best_w.tolist()
    }
    
    # ── 6. Assemble Full End-to-End Unified Keras Ensemble Model ──
    print(f"\n[Step 6] Assembling Unified End-to-End Keras Ensemble Model for {sp_cap}...")
    unified_input = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="raw_image_input")
    
    # Load baseline model
    base_prod_path = MODELS / f"{'Dog_EfficientNetB0_balanced_best' if species=='dog' else 'Cat_EfficientNetB0_stage2_best'}.keras"
    prod_m = tf.keras.models.load_model(str(base_prod_path))
    prod_m.trainable = False
    
    p0 = prod_m(unified_input)
    p1 = full_model_a(unified_input)
    p2 = full_model_b(unified_input)
    p3 = full_model_c(unified_input)
    
    # Weighted average layer
    w0, w1, w2, w3 = float(best_w[0]), float(best_w[1]), float(best_w[2]), float(best_w[3])
    unified_output = tf.keras.layers.Add(name="weighted_soft_vote")([
        tf.keras.layers.Lambda(lambda x: x * w0)(p0),
        tf.keras.layers.Lambda(lambda x: x * w1)(p1),
        tf.keras.layers.Lambda(lambda x: x * w2)(p2),
        tf.keras.layers.Lambda(lambda x: x * w3)(p3),
    ])
    
    unified_ensemble_model = tf.keras.Model(
        inputs=unified_input, outputs=unified_output,
        name=f"{sp_cap}_Unified_Weighted_Ensemble"
    )
    
    ensemble_model_file = MODELS / f"{sp_cap}_Ensemble_v1.keras"
    unified_ensemble_model.save(str(ensemble_model_file))
    print(f"  [SUCCESS] Unified Ensemble model exported to: {ensemble_model_file.name}")
    print(f"  Model input contract : {unified_ensemble_model.input_shape} | RAW [0, 255] RGB float")
    print(f"  Model output contract: {unified_ensemble_model.output_shape} | Softmax probability vector")
    
    # ── 7. Generate Authentic Figures & Confusion Matrices ──
    print(f"\n[Step 7] Generating Authentic Academic Figures...")
    
    # 7.1 Baseline Test CM
    plot_and_save_cm(
        np.array(baseline_test_m['cm']), classes,
        f"{sp_cap} Baseline Test Confusion Matrix (Acc: {baseline_test_m['accuracy']*100:.2f}%)",
        FIGS / f"cm_{species}_baseline_test.png"
    )
    
    # 7.2 Ensemble Test CM
    plot_and_save_cm(
        np.array(ens_weighted_test_m['cm']), classes,
        f"{sp_cap} Unified Ensemble Test Confusion Matrix (Acc: {ens_weighted_test_m['accuracy']*100:.2f}%)",
        FIGS / f"cm_{species}_ensemble_test.png"
    )
    
    # 7.3 Per-Class F1 Chart (Baseline vs Ensemble)
    plot_per_class_f1(
        ens_weighted_test_m['per_class'], classes,
        f"{sp_cap} Per-Class F1-Score (Unified Weighted Ensemble)",
        FIGS / f"per_class_f1_{species}_ensemble.png"
    )
    
    # 7.4 Model Comparison Bar Chart
    model_names_plot = list(candidates.keys())
    accs = [candidates[m]["test_metrics"]["accuracy"] * 100 for m in model_names_plot]
    f1s = [candidates[m]["test_metrics"]["macro_f1"] * 100 for m in model_names_plot]
    
    x = np.arange(len(model_names_plot))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    r1 = ax.bar(x - width/2, accs, width, label='Test Accuracy (%)', color='#1f77b4', alpha=0.85)
    r2 = ax.bar(x + width/2, f1s, width, label='Test Macro F1 (%)', color='#2ca02c', alpha=0.85)
    ax.set_ylabel('Score (%)', fontweight='bold', fontsize=11)
    ax.set_title(f"{sp_cap} Disease Classification — Model Comparison on Held-Out Test Set", fontweight='bold', fontsize=12, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" (", "\n(").replace(")","") for m in model_names_plot], fontsize=9)
    ax.set_ylim(60, 100)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    for r in list(r1) + list(r2):
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2, h + 0.5, f"{h:.2f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    comparison_chart_file = FIGS / f"comparison_{species}_all_candidates.png"
    plt.savefig(comparison_chart_file, dpi=160, bbox_inches='tight')
    plt.close()
    print(f"  Saved comparison chart → {comparison_chart_file.name}")
    
    # ── 8. Save CSV Reports ──
    # Per-class metrics CSV
    per_class_csv = DATA / f"per_class_metrics_{species}_ensemble.csv"
    with open(per_class_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Baseline_Precision", "Baseline_Recall", "Baseline_F1",
                         "Ensemble_Precision", "Ensemble_Recall", "Ensemble_F1", "Support", "F1_Improvement"])
        for c in classes:
            b_p = baseline_test_m['per_class'][c]['precision']
            b_r = baseline_test_m['per_class'][c]['recall']
            b_f1 = baseline_test_m['per_class'][c]['f1-score']
            e_p = ens_weighted_test_m['per_class'][c]['precision']
            e_r = ens_weighted_test_m['per_class'][c]['recall']
            e_f1 = ens_weighted_test_m['per_class'][c]['f1-score']
            sup = baseline_test_m['per_class'][c]['support']
            diff = e_f1 - b_f1
            writer.writerow([c, f"{b_p:.4f}", f"{b_r:.4f}", f"{b_f1:.4f}",
                             f"{e_p:.4f}", f"{e_r:.4f}", f"{e_f1:.4f}", int(sup), f"{diff:+.4f}"])
    print(f"  Saved per-class CSV → {per_class_csv.name}")
    
    # Confusion matrix CSV
    cm_csv = DATA / f"confusion_matrix_{species}_ensemble.csv"
    with open(cm_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["True_Class"] + classes)
        for i, c in enumerate(classes):
            writer.writerow([c] + ens_weighted_test_m['cm'][i])
    print(f"  Saved confusion matrix CSV → {cm_csv.name}")
    
    return {
        "species": species,
        "classes": classes,
        "baseline_test": baseline_test_m,
        "ensemble_test": ens_weighted_test_m,
        "candidates": {
            m: {
                "arch": candidates[m]["arch"],
                "val_accuracy": candidates[m]["val_metrics"]["accuracy"],
                "val_macro_f1": candidates[m]["val_metrics"]["macro_f1"],
                "test_accuracy": candidates[m]["test_metrics"]["accuracy"],
                "test_macro_f1": candidates[m]["test_metrics"]["macro_f1"],
                "test_weighted_f1": candidates[m]["test_metrics"]["weighted_f1"],
                "weights": candidates[m].get("weights", None)
            } for m in candidates
        },
        "ensemble_model_file": str(ensemble_model_file)
    }

def main():
    start_time = time.time()
    
    dog_summary = run_pipeline_for_species("dog")
    cat_summary = run_pipeline_for_species("cat")
    
    # ── Final Summary JSON ──
    final_eval = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "Intel CPU (native Windows DirectML/AVX2)",
        "tensorflow_version": tf.__version__,
        "seed": 42,
        "dog": dog_summary,
        "cat": cat_summary
    }
    with open(DATA / "final_model_evaluation.json", "w", encoding='utf-8') as f:
        json.dump(final_eval, f, indent=2)
    print(f"\n[DONE] Saved complete evaluation to: data/final_model_evaluation.json")
    
    # ── Save Experiment Log CSV ──
    exp_log = REPORTS / "model_improvement_experiments.csv"
    with open(exp_log, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment_id", "species", "model_candidate", "architecture",
            "val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1", "test_weighted_f1"
        ])
        exp_id = 1
        for sp_sum in [dog_summary, cat_summary]:
            sp = sp_sum["species"]
            for c_name, c_info in sp_sum["candidates"].items():
                writer.writerow([
                    f"EXP-{exp_id:02d}", sp, c_name, c_info["arch"],
                    f"{c_info['val_accuracy']*100:.2f}%", f"{c_info['val_macro_f1']:.4f}",
                    f"{c_info['test_accuracy']*100:.2f}%", f"{c_info['test_macro_f1']:.4f}",
                    f"{c_info['test_weighted_f1']:.4f}"
                ])
                exp_id += 1
    print(f"Saved experiment log to: {exp_log.name}")
    print(f"Total pipeline execution time: {(time.time() - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    main()

