"""
Evaluate baseline models on validation and test manifests.
"""
import os, json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, confusion_matrix

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PROJECT = Path(r"c:\Users\Jogen\OneDrive\Desktop\Jogendra\Projects\Animal_Disease_App")
DATA = PROJECT / "data"
MODELS = PROJECT / "models"

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

def eval_split(model, manifest_path, classes):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    cls2idx = {c: i for i, c in enumerate(classes)}
    paths, labels = [], []
    for item in manifest:
        if item['label'] in cls2idx and Path(item['path']).exists():
            paths.append(item['path'])
            labels.append(cls2idx[item['label']])
    
    def parse_img(p, l):
        raw = tf.io.read_file(p)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [224, 224])
        return tf.cast(img, tf.float32), l
    
    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).map(
        parse_img, num_parallel_calls=tf.data.AUTOTUNE).batch(32)
    preds = model.predict(ds, verbose=0)
    pred_cls = np.argmax(preds, axis=1)
    acc = accuracy_score(labels, pred_cls)
    bal = balanced_accuracy_score(labels, pred_cls)
    rep = classification_report(labels, pred_cls, target_names=classes, output_dict=True)
    cm = confusion_matrix(labels, pred_cls)
    return acc, bal, rep, cm, preds, np.array(labels)

def main():
    print("="*60)
    print("EVALUATING DOG BASELINE MODEL")
    print("="*60)
    dog_model_path = MODELS / "Dog_EfficientNetB0_balanced_best.keras"
    dog_model = tf.keras.models.load_model(str(dog_model_path))

    acc_val, bal_val, rep_val, cm_val, val_preds, val_y = eval_split(
        dog_model, DATA / "manifest_dog_val.json", DOG_CLASSES)
    print(f"Dog VAL  : Acc={acc_val*100:.2f}%, BalAcc={bal_val*100:.2f}%, "
          f"MacroF1={rep_val['macro avg']['f1-score']:.4f}, WtF1={rep_val['weighted avg']['f1-score']:.4f}")

    acc_test, bal_test, rep_test, cm_test, test_preds, test_y = eval_split(
        dog_model, DATA / "manifest_dog_test.json", DOG_CLASSES)
    print(f"Dog TEST : Acc={acc_test*100:.2f}%, BalAcc={bal_test*100:.2f}%, "
          f"MacroF1={rep_test['macro avg']['f1-score']:.4f}, WtF1={rep_test['weighted avg']['f1-score']:.4f}")

    # Save baseline metrics
    baseline_dog = {
        "val": {
            "accuracy": float(acc_val), "balanced_accuracy": float(bal_val),
            "macro_f1": float(rep_val['macro avg']['f1-score']),
            "weighted_f1": float(rep_val['weighted avg']['f1-score']),
            "report": rep_val, "cm": cm_val.tolist()
        },
        "test": {
            "accuracy": float(acc_test), "balanced_accuracy": float(bal_test),
            "macro_f1": float(rep_test['macro avg']['f1-score']),
            "weighted_f1": float(rep_test['weighted avg']['f1-score']),
            "report": rep_test, "cm": cm_test.tolist()
        }
    }
    with open(DATA / "baseline_dog_eval.json", "w") as f:
        json.dump(baseline_dog, f, indent=2)
    np.save(str(DATA / "dog_baseline_val_preds.npy"), val_preds)
    np.save(str(DATA / "dog_baseline_test_preds.npy"), test_preds)

    del dog_model

    print("\n" + "="*60)
    print("EVALUATING CAT BASELINE MODEL")
    print("="*60)
    cat_model_path = MODELS / "Cat_EfficientNetB0_stage2_best.keras"
    cat_model = tf.keras.models.load_model(str(cat_model_path))

    acc_val_cat, bal_val_cat, rep_val_cat, cm_val_cat, val_preds_cat, _ = eval_split(
        cat_model, DATA / "manifest_cat_val.json", CAT_CLASSES)
    print(f"Cat VAL  : Acc={acc_val_cat*100:.2f}%, BalAcc={bal_val_cat*100:.2f}%, "
          f"MacroF1={rep_val_cat['macro avg']['f1-score']:.4f}, WtF1={rep_val_cat['weighted avg']['f1-score']:.4f}")

    acc_test_cat, bal_test_cat, rep_test_cat, cm_test_cat, test_preds_cat, _ = eval_split(
        cat_model, DATA / "manifest_cat_test.json", CAT_CLASSES)
    print(f"Cat TEST : Acc={acc_test_cat*100:.2f}%, BalAcc={bal_test_cat*100:.2f}%, "
          f"MacroF1={rep_test_cat['macro avg']['f1-score']:.4f}, WtF1={rep_test_cat['weighted avg']['f1-score']:.4f}")

    # Save baseline metrics
    baseline_cat = {
        "val": {
            "accuracy": float(acc_val_cat), "balanced_accuracy": float(bal_val_cat),
            "macro_f1": float(rep_val_cat['macro avg']['f1-score']),
            "weighted_f1": float(rep_val_cat['weighted avg']['f1-score']),
            "report": rep_val_cat, "cm": cm_val_cat.tolist()
        },
        "test": {
            "accuracy": float(acc_test_cat), "balanced_accuracy": float(bal_test_cat),
            "macro_f1": float(rep_test_cat['macro avg']['f1-score']),
            "weighted_f1": float(rep_test_cat['weighted avg']['f1-score']),
            "report": rep_test_cat, "cm": cm_test_cat.tolist()
        }
    }
    with open(DATA / "baseline_cat_eval.json", "w") as f:
        json.dump(baseline_cat, f, indent=2)
    np.save(str(DATA / "cat_baseline_val_preds.npy"), val_preds_cat)
    np.save(str(DATA / "cat_baseline_test_preds.npy"), test_preds_cat)

    print("\nBaseline evaluation complete and saved!")

if __name__ == "__main__":
    main()

