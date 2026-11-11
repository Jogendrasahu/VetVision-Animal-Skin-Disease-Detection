"""
STEP 2 — Train two EfficientNetB0 candidate models for Dog and Cat.
Architecture matches the production model exactly:
  - RAW [0,255] float input
  - Keras RandomFlip/Rotation/Zoom/Contrast augmentation built into model
  - EfficientNetB0 backbone (which internally does its own rescaling)
  - GlobalAveragePooling2D → BatchNorm → Dropout → Dense(n_classes, softmax)

Candidate A (seed=42): augmentation policy matching baseline (flip, rotation±0.08, zoom 0.12, contrast 0.12)
Candidate B (seed=7):  stronger augmentation (flip, rotation±0.12, zoom 0.15, contrast 0.15, brightness)

Usage:
  python scripts/step2_train_candidates.py --species dog --candidate A
  python scripts/step2_train_candidates.py --species dog --candidate B
  python scripts/step2_train_candidates.py --species cat --candidate A
  python scripts/step2_train_candidates.py --species cat --candidate B
"""
import os, json, argparse, shutil
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, balanced_accuracy_score, accuracy_score, confusion_matrix
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PROJECT = Path(r"c:\Users\Jogen\OneDrive\Desktop\Jogendra\Projects\Animal_Disease_App")
DATA    = PROJECT / "data"
MODELS  = PROJECT / "models"
FIGS    = PROJECT / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
(PROJECT / "scripts").mkdir(exist_ok=True)

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
BATCH    = 24   # CPU-friendly

# ─────────────────────────── Data loader ──────────────────────────────────────
def load_manifest(path):
    with open(path) as f: return json.load(f)

def build_tf_dataset(manifest, classes, shuffle=True, batch=BATCH):
    """Builds a tf.data pipeline. Returns (dataset, label_array)."""
    cls2idx = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for item in manifest:
        p = Path(item['path'])
        if item['label'] in cls2idx and p.exists():
            paths.append(str(p))
            labels.append(cls2idx[item['label']])
    print(f"    Loaded {len(paths)} valid images ({len(manifest) - len(paths)} skipped)")

    def parse_img(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        img = tf.cast(img, tf.float32)  # RAW [0,255] — model handles rescaling internally
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=42, reshuffle_each_iteration=True)
    ds = (ds
          .map(parse_img, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(batch)
          .prefetch(tf.data.AUTOTUNE))
    return ds, np.array(labels, dtype=np.int32)

# ─────────────────────────── Model builder ────────────────────────────────────
def build_model(num_classes, seed, rotation_factor, zoom_factor, contrast_factor, dropout=0.35):
    """
    Builds model matching production architecture:
      Input → Augmentation(Sequential) → EfficientNetB0 → GAP → BN → Dropout → Dense
    """
    tf.keras.utils.set_random_seed(seed)

    # ── Augmentation sub-model (same as production) ──
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(rotation_factor, fill_mode="reflect"),
        tf.keras.layers.RandomZoom(zoom_factor,          fill_mode="reflect"),
        tf.keras.layers.RandomContrast(contrast_factor,  value_range=[0, 255]),
    ], name="augmentation")

    # ── EfficientNetB0 backbone (imagenet weights, frozen initially) ──
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    backbone.trainable = False

    # ── Build full model ──
    inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
    x = aug(inputs, training=True)        # aug only during training
    x = backbone(x, training=False)       # frozen backbone
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_head")(x)
    x = tf.keras.layers.Dropout(dropout, seed=seed, name="dropout_head")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name=f"EfficientNetB0_candidate")
    return model, backbone

def compute_weights(labels, n_classes):
    cw = compute_class_weight('balanced', classes=np.arange(n_classes), y=labels)
    return dict(enumerate(cw))

def evaluate(model, ds, true_labels, classes, label=""):
    preds = model.predict(ds, verbose=0)
    pred_labels = np.argmax(preds, axis=1)
    acc = accuracy_score(true_labels, pred_labels)
    bal = balanced_accuracy_score(true_labels, pred_labels)
    rep = classification_report(true_labels, pred_labels, target_names=classes, output_dict=True)
    cm  = confusion_matrix(true_labels, pred_labels)
    print(f"  [{label}] Acc={acc*100:.2f}%  BalAcc={bal*100:.2f}%  MacroF1={rep['macro avg']['f1-score']:.4f}  WtF1={rep['weighted avg']['f1-score']:.4f}")
    return acc, bal, rep, cm, preds

def save_curves(history, species, cand_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{species.upper()} — {cand_name} Training Curves", fontsize=14, fontweight='bold')
    
    epochs_p1 = len(history.get('phase1_accuracy', []))
    all_acc     = history.get('phase1_accuracy',     []) + history.get('phase2_accuracy',     [])
    all_val_acc = history.get('phase1_val_accuracy', []) + history.get('phase2_val_accuracy', [])
    all_loss    = history.get('phase1_loss',         []) + history.get('phase2_loss',         [])
    all_val_loss= history.get('phase1_val_loss',     []) + history.get('phase2_val_loss',     [])
    ep = range(1, len(all_acc)+1)
    
    ax1.plot(ep, all_acc,     'b-', lw=2, label='Train Accuracy')
    ax1.plot(ep, all_val_acc, 'r-', lw=2, label='Val Accuracy')
    if epochs_p1 > 0:
        ax1.axvline(x=epochs_p1, color='gray', ls='--', alpha=0.7, label='Phase 2 start')
    ax1.set(title='Accuracy vs Epoch', xlabel='Epoch', ylabel='Accuracy', ylim=(0,1))
    ax1.legend(); ax1.grid(True, alpha=0.3)
    
    ax2.plot(ep, all_loss,     'b-', lw=2, label='Train Loss')
    ax2.plot(ep, all_val_loss, 'r-', lw=2, label='Val Loss')
    if epochs_p1 > 0:
        ax2.axvline(x=epochs_p1, color='gray', ls='--', alpha=0.7, label='Phase 2 start')
    ax2.set(title='Loss vs Epoch', xlabel='Epoch', ylabel='Loss')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out = FIGS / f"training_curves_{species}_{cand_name}.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved curves → {out.name}")

def save_cm(cm, classes, title, out_path):
    fig, ax = plt.subplots(figsize=(max(10, len(classes)), max(8, len(classes))))
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel('Predicted Label'); ax.set_ylabel('True Label'); ax.set_title(title, fontsize=11)
    thresh = cm.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=7,
                    color='white' if cm[i,j] > thresh else 'black')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

# ─────────────────────────── Training ─────────────────────────────────────────
def train_candidate(species, seed, aug_params, aug_name, classes, cand_letter):
    n_classes  = len(classes)
    cand_name  = f"candidate_{cand_letter}"
    model_path = MODELS / f"{species.capitalize()}_EfficientNetB0_{cand_name}.keras"
    history_path = DATA / f"history_{species}_{cand_name}.json"

    print(f"\n{'='*70}")
    print(f"TRAINING: {species.upper()} — {cand_name}  seed={seed}  aug={aug_name}")
    print(f"{'='*70}")

    # Load data
    print("  Building train dataset...")
    train_ds, train_labels = build_tf_dataset(
        load_manifest(DATA / f"manifest_{species}_train.json"), classes, shuffle=True)
    print("  Building val dataset...")
    val_ds, val_labels = build_tf_dataset(
        load_manifest(DATA / f"manifest_{species}_val.json"), classes, shuffle=False)

    # Class weights
    cw = compute_weights(train_labels, n_classes)
    ratio = max(cw.values()) / min(cw.values())
    print(f"  Class weight max/min ratio: {ratio:.1f}")

    # Build model
    model, backbone = build_model(
        n_classes, seed,
        rotation_factor=aug_params['rotation'],
        zoom_factor=aug_params['zoom'],
        contrast_factor=aug_params['contrast'],
        dropout=aug_params.get('dropout', 0.35)
    )

    all_history = {}

    # ── Phase 1: frozen backbone ──
    print(f"\n  Phase 1 — head only (backbone frozen, epochs≤20)...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    cb1 = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6,
                                          restore_best_weights=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                              patience=3, min_lr=1e-6, verbose=0),
    ]
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=20,
                   class_weight=cw, callbacks=cb1, verbose=1)
    for k in ['accuracy','val_accuracy','loss','val_loss']:
        all_history[f'phase1_{k}'] = [float(v) for v in h1.history.get(k, [])]

    # ── Phase 2: unfreeze top layers ──
    print(f"\n  Phase 2 — fine-tuning top 40 backbone layers (epochs≤30)...")
    backbone.trainable = True
    for layer in backbone.layers[:-40]:
        layer.trainable = False
    frozen_count = sum(1 for l in backbone.layers if not l.trainable)
    total_count  = len(backbone.layers)
    print(f"    Backbone: {total_count - frozen_count}/{total_count} layers trainable")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    cb2 = [
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8,
                                          restore_best_weights=True, verbose=0),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                              patience=4, min_lr=1e-7, verbose=0),
        tf.keras.callbacks.ModelCheckpoint(str(model_path), monitor='val_accuracy',
                                            save_best_only=True, verbose=0),
    ]
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=30,
                   class_weight=cw, callbacks=cb2, verbose=1)
    for k in ['accuracy','val_accuracy','loss','val_loss']:
        all_history[f'phase2_{k}'] = [float(v) for v in h2.history.get(k, [])]

    # Save history JSON
    with open(history_path, 'w') as f:
        json.dump(all_history, f, indent=2)
    print(f"  History saved → {history_path.name}")

    # Training curves
    save_curves(all_history, species, cand_name)

    # Evaluate on val
    print("\n  Evaluating on VALIDATION set...")
    val_acc, val_bal, val_rep, val_cm, val_preds = evaluate(
        model, val_ds, val_labels, classes, label="VAL")

    # Save val preds array for ensemble
    np.save(str(DATA / f"val_preds_{species}_{cand_name}.npy"), val_preds)

    # Save candidate result JSON
    result = {
        "species": species, "candidate": cand_name, "seed": seed,
        "augmentation": aug_name,
        "val_accuracy": float(val_acc),
        "val_balanced_accuracy": float(val_bal),
        "val_macro_f1": float(val_rep['macro avg']['f1-score']),
        "val_weighted_f1": float(val_rep['weighted avg']['f1-score']),
        "val_macro_precision": float(val_rep['macro avg']['precision']),
        "val_macro_recall": float(val_rep['macro avg']['recall']),
        "model_path": str(model_path),
        "phase1_epochs": len(all_history.get('phase1_accuracy', [])),
        "phase2_epochs": len(all_history.get('phase2_accuracy', [])),
    }
    with open(DATA / f"candidate_metrics_{species}_{cand_name}.json", 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Candidate metrics saved.")

    save_cm(val_cm, classes,
            f"{species.capitalize()} {cand_name} — Validation Confusion Matrix",
            FIGS / f"val_cm_{species}_{cand_name}.png")

    return model, val_preds, val_labels, val_acc


# ─────────────────────────── Main ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--species",   required=True, choices=["dog","cat"])
    parser.add_argument("--candidate", default="both", choices=["A","B","both"])
    args = parser.parse_args()

    classes = DOG_CLASSES if args.species == "dog" else CAT_CLASSES

    # Backup existing production model (once)
    prod_name = "Dog_EfficientNetB0_balanced_best" if args.species == "dog" \
                else "Cat_EfficientNetB0_stage2_best"
    prod_path   = MODELS / f"{prod_name}.keras"
    backup_path = MODELS / f"{prod_name}_BASELINE_BACKUP.keras"
    if prod_path.exists() and not backup_path.exists():
        shutil.copy(str(prod_path), str(backup_path))
        print(f"Backed up baseline model → {backup_path.name}")

    # Augmentation configs
    AUG_A = dict(rotation=0.08, zoom=0.12, contrast=0.12, dropout=0.35)  # matches baseline approx
    AUG_B = dict(rotation=0.12, zoom=0.15, contrast=0.15, dropout=0.40)  # stronger

    if args.candidate in ("A", "both"):
        train_candidate(args.species, seed=42, aug_params=AUG_A,
                        aug_name="policy_A_moderate", classes=classes, cand_letter="A")

    if args.candidate in ("B", "both"):
        train_candidate(args.species, seed=7, aug_params=AUG_B,
                        aug_name="policy_B_stronger", classes=classes, cand_letter="B")

    print("\nAll done! Run step3_ensemble_eval.py next.")
