"""
STEP 1 — Build clean train/val/test manifests for Dog and Cat.
Seeds everything at 42. Prevents leakage across splits.
Saves:
  data/manifest_dog_train.json  data/manifest_dog_val.json  data/manifest_dog_test.json
  data/manifest_cat_train.json  data/manifest_cat_val.json  data/manifest_cat_test.json
"""
import os, json, random, hashlib, shutil
from pathlib import Path
from collections import defaultdict

SEED = 42
random.seed(SEED)

PROJECT = Path(r"c:\Users\Jogen\OneDrive\Desktop\Jogendra\Projects\Animal_Disease_App")
DAML    = Path(r"C:\Users\Jogen\OneDrive\Desktop\Jogendra\DAML\DL Project")
DATA    = PROJECT / "data"
DATA.mkdir(exist_ok=True)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ── Class maps (must match production DOG_CLASSES / CAT_CLASSES order exactly) ──
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

# ── Dataset source folders → class label mapping ──
# Maps (source_folder, subfolder_name_variants) → canonical_class
DOG_SOURCES = {
    # Dataset_1_Dog_Cat
    'demodicosis':     [DAML / "Dataset_1_Dog_Cat/data/Demodicosis"],
    'dental_disease':  [DAML / "Dataset_1_Dog_Cat/data/Dental Disease in Dog"],
    'dermatitis':      [DAML / "Dataset_1_Dog_Cat/data/Dermatitis"],
    'distemper':       [DAML / "Dataset_1_Dog_Cat/data/Distemper in Dog"],
    'eye_infection':   [DAML / "Dataset_1_Dog_Cat/data/Eye Infection in Dog"],
    'fungal_infection':[DAML / "Dataset_1_Dog_Cat/data/Fungal Infection in Dog"],
    'healthy':         [DAML / "Dataset_1_Dog_Cat/data/Healthy Dog"],
    'hot_spots':       [DAML / "Dataset_1_Dog_Cat/data/Hot Spots in Dog"],
    'hypersensitivity':[DAML / "Dataset_1_Dog_Cat/data/Hypersensitivity"],
    'kennel_cough':    [DAML / "Dataset_1_Dog_Cat/data/Kennel Cough in Dog"],
    'mange':           [DAML / "Dataset_1_Dog_Cat/data/Mange in Dog"],
    'parvovirus':      [DAML / "Dataset_1_Dog_Cat/data/Parvovirus in Dog"],
    'ringworm':        [DAML / "Dataset_1_Dog_Cat/data/Ringworm in Dog"],
    'skin_allergy':    [DAML / "Dataset_1_Dog_Cat/data/Skin Allergy in Dog"],
    'tick_infestation':[DAML / "Dataset_1_Dog_Cat/data/Tick Infestation in Dog"],
    'worm_infection':  [DAML / "Dataset_1_Dog_Cat/data/Worm Infection in Dog"],
}

# Add Dataset_5_Dog multi-split sources for some classes
DS5_CLASS_MAP = {
    'demodicosis':      'demodicosis',
    'Dermatitis':       'dermatitis',
    'Fungal_infections':'fungal_infection',
    'Healthy':          'healthy',
    'Hypersensitivity': 'hypersensitivity',
    'ringworm':         'ringworm',
}
for split in ['train','valid','test']:
    for folder, cls in DS5_CLASS_MAP.items():
        p = DAML / f"Dataset_5_Dog/{split}/{folder}"
        if p.exists():
            if cls not in DOG_SOURCES:
                DOG_SOURCES[cls] = []
            DOG_SOURCES[cls].append(p)

CAT_SOURCES = {
    'dental_disease':       [DAML / "Dataset_1_Dog_Cat/data/Dental Disease in Cat"],
    'ear_mites':            [DAML / "Dataset_1_Dog_Cat/data/Ear Mites in Cat",
                             DAML / "Dataset_8_Cat/train/Ear Mites",
                             DAML / "Dataset_8_Cat/valid/Ear Mites",
                             DAML / "Dataset_8_Cat/test/Ear Mites"],
    'eye_infection':        [DAML / "Dataset_1_Dog_Cat/data/Eye Infection in Cat"],
    'feline_acne':          [DAML / "Dataset_8_Cat/train/Feline Acne",
                             DAML / "Dataset_8_Cat/valid/Feline Acne",
                             DAML / "Dataset_8_Cat/test/Feline Acne"],
    'feline_leukemia':      [DAML / "Dataset_1_Dog_Cat/data/Feline Leukemia"],
    'feline_panleukopenia': [DAML / "Dataset_1_Dog_Cat/data/Feline Panleukopenia"],
    'flea_allergy':         [DAML / "Dataset_1_Dog_Cat/data/Flea Allergy in Cat",
                             DAML / "Dataset_7_Cat/train/Flea_Allergy",
                             DAML / "Dataset_7_Cat/valid/Flea_Allergy",
                             DAML / "Dataset_7_Cat/test/Flea_Allergy"],
    'fleas':                [DAML / "Dataset_8_Cat/train/Fleas",
                             DAML / "Dataset_8_Cat/valid/Fleas",
                             DAML / "Dataset_8_Cat/test/Fleas"],
    'fungal_infection':     [DAML / "Dataset_1_Dog_Cat/data/Fungal Infection in Cat"],
    'healthy':              [DAML / "Dataset_1_Dog_Cat/data/Healthy Cat",
                             DAML / "Dataset_7_Cat/train/Health",
                             DAML / "Dataset_7_Cat/valid/Health",
                             DAML / "Dataset_7_Cat/test/Health"],
    'ringworm':             [DAML / "Dataset_1_Dog_Cat/data/Ringworm in Cat",
                             DAML / "Dataset_7_Cat/train/Ringworm",
                             DAML / "Dataset_7_Cat/valid/Ringworm",
                             DAML / "Dataset_7_Cat/test/Ringworm",
                             DAML / "Dataset_8_Cat/train/Ringworm",
                             DAML / "Dataset_8_Cat/valid/Ringworm",
                             DAML / "Dataset_8_Cat/test/Ringworm"],
    'scabies':              [DAML / "Dataset_1_Dog_Cat/data/Scabies in Cat",
                             DAML / "Dataset_7_Cat/train/Scabies",
                             DAML / "Dataset_7_Cat/valid/Scabies",
                             DAML / "Dataset_7_Cat/test/Scabies"],
    'skin_allergy':         [DAML / "Dataset_1_Dog_Cat/data/Skin Allergy in Cat"],
    'urinary_tract_infection':[DAML / "Dataset_1_Dog_Cat/data/Urinary Tract Infection in Cat"],
    'worm_infection':       [DAML / "Dataset_1_Dog_Cat/data/Worm Infection in Cat"],
}

def collect_images(sources_dict):
    """Return dict: class → list of image paths (deduplicated by md5 hash)."""
    cls2imgs = defaultdict(list)
    for cls, folders in sources_dict.items():
        seen_hashes = set()
        for folder in folders:
            folder = Path(folder)
            if not folder.exists():
                print(f"  [SKIP] Not found: {folder}")
                continue
            for f in folder.rglob("*"):
                if f.suffix.lower() in IMG_EXTS and f.is_file():
                    # deduplicate
                    h = hashlib.md5(f.read_bytes()).hexdigest()
                    if h not in seen_hashes:
                        seen_hashes.add(h)
                        cls2imgs[cls].append(str(f))
    return cls2imgs

def make_split(cls2imgs, classes, val_ratio=0.15, test_ratio=0.15):
    """
    Leakage-safe split: for each class, shuffle then slice into train/val/test.
    Guarantees no image appears in multiple splits.
    test_ratio applies to per-class images. val_ratio from remainder after test.
    Returns three dicts of {class: [paths]}.
    """
    train, val, test = defaultdict(list), defaultdict(list), defaultdict(list)
    for cls in classes:
        imgs = list(cls2imgs.get(cls, []))
        random.shuffle(imgs)
        n = len(imgs)
        n_test = max(1, int(n * test_ratio))
        n_val  = max(1, int((n - n_test) * val_ratio))
        test[cls]  = imgs[:n_test]
        val[cls]   = imgs[n_test:n_test+n_val]
        train[cls] = imgs[n_test+n_val:]
        print(f"  {cls:30s}: total={n:4d}  train={len(train[cls]):4d}  val={len(val[cls]):4d}  test={len(test[cls]):4d}")
    return train, val, test

def save_manifest(split_dict, path):
    flat = []
    for cls, paths in split_dict.items():
        for p in paths:
            flat.append({"path": p, "label": cls})
    with open(path, "w") as f:
        json.dump(flat, f, indent=2)
    print(f"  Saved {len(flat)} entries → {path}")

# ── DOG ──
print("\n=== DOG DATASET ===")
dog_imgs = collect_images(DOG_SOURCES)
# Print counts
for cls in DOG_CLASSES:
    print(f"  {cls:30s}: {len(dog_imgs.get(cls,[]))} images")
print("\nSplitting Dog...")
dog_train, dog_val, dog_test = make_split(dog_imgs, DOG_CLASSES)
save_manifest(dog_train, DATA / "manifest_dog_train.json")
save_manifest(dog_val,   DATA / "manifest_dog_val.json")
save_manifest(dog_test,  DATA / "manifest_dog_test.json")

# ── CAT ──
print("\n=== CAT DATASET ===")
cat_imgs = collect_images(CAT_SOURCES)
for cls in CAT_CLASSES:
    print(f"  {cls:30s}: {len(cat_imgs.get(cls,[]))} images")
print("\nSplitting Cat...")
cat_train, cat_val, cat_test = make_split(cat_imgs, CAT_CLASSES)
save_manifest(cat_train, DATA / "manifest_cat_train.json")
save_manifest(cat_val,   DATA / "manifest_cat_val.json")
save_manifest(cat_test,  DATA / "manifest_cat_test.json")

print("\nDone! Manifests written to data/")

