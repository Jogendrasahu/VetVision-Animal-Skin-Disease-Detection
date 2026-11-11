"""
Phase 1 & 2: Deep Data Audit & Error-Driven Weak Class Analysis
Calculates:
1. Image quality metrics across train/val/test:
   - Aspect ratio distributions
   - Blur metric (Laplacian variance)
   - Mean brightness and contrast
2. Weak class analysis:
   - Per-class Recall, Precision, F1, and Support
   - Ranking classes from weakest to strongest
   - Specific misclassification targets for every weak class
3. Top 10 confusion pairs for Dog and Cat
"""
import os, sys, json
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from collections import Counter, defaultdict

PROJECT = Path(r"c:\Users\Jogen\OneDrive\Desktop\Jogendra\Projects\Animal_Disease_App")
DATA = PROJECT / "data"
REPORTS = PROJECT / "reports"

DOG_CLASSES = ['demodicosis', 'dental_disease', 'dermatitis', 'distemper', 'eye_infection', 'fungal_infection', 'healthy', 'hot_spots', 'hypersensitivity', 'kennel_cough', 'mange', 'parvovirus', 'ringworm', 'skin_allergy', 'tick_infestation', 'worm_infection']
CAT_CLASSES = ['dental_disease', 'ear_mites', 'eye_infection', 'feline_acne', 'feline_leukemia', 'feline_panleukopenia', 'flea_allergy', 'fleas', 'fungal_infection', 'healthy', 'ringworm', 'scabies', 'skin_allergy', 'urinary_tract_infection', 'worm_infection']

def sample_image_stats(manifest_path, sample_size=300):
    with open(manifest_path, 'r', encoding='utf-8') as f:
        items = json.load(f)
    
    # Sample items evenly
    np.random.seed(42)
    sample_indices = np.random.choice(len(items), min(sample_size, len(items)), replace=False)
    
    aspect_ratios = []
    blur_scores = []
    brightnesses = []
    contrasts = []
    
    for idx in sample_indices:
        p = Path(items[idx]['path'])
        if not p.exists():
            continue
        try:
            # OpenCV read for blur & brightness
            img_bgr = cv2.imread(str(p))
            if img_bgr is None:
                continue
            h, w = img_bgr.shape[:2]
            aspect_ratios.append(w / max(1, h))
            
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            # Blur via Laplacian variance
            lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_scores.append(float(lap_var))
            
            # Brightness & contrast
            brightnesses.append(float(np.mean(gray)))
            contrasts.append(float(np.std(gray)))
        except Exception:
            continue
            
    return {
        "sample_count": len(blur_scores),
        "mean_aspect_ratio": float(np.mean(aspect_ratios)) if aspect_ratios else 1.0,
        "mean_blur_laplacian": float(np.mean(blur_scores)) if blur_scores else 0.0,
        "median_blur_laplacian": float(np.median(blur_scores)) if blur_scores else 0.0,
        "low_blur_count_under_100": sum(1 for b in blur_scores if b < 100),
        "mean_brightness": float(np.mean(brightnesses)) if brightnesses else 128.0,
        "mean_contrast": float(np.mean(contrasts)) if contrasts else 50.0
    }

def analyze_weak_classes_and_confusion(species, classes):
    with open(DATA / "phase2_rigorous_evaluation.json", "r") as f:
        p2 = json.load(f)[species]
        
    model_data = p2["Weighted_Ensemble"]
    per_class = model_data["per_class"]
    cm = np.array(model_data["cm"])
    
    # 1. Rank classes by Recall and F1
    ranked_classes = []
    for idx, c in enumerate(classes):
        m = per_class.get(c, {})
        rec = m.get("recall", 0.0)
        prec = m.get("precision", 0.0)
        f1 = m.get("f1-score", 0.0)
        supp = m.get("support", 0.0)
        
        # Major confusion targets
        row = cm[idx]
        confused_with = []
        for jdx, target_c in enumerate(classes):
            if idx != jdx and row[jdx] > 0:
                confused_with.append((target_c, int(row[jdx])))
        confused_with.sort(key=lambda x: x[1], reverse=True)
        
        total_errors = sum(cnt for _, cnt in confused_with)
        ranked_classes.append({
            "class_name": c,
            "recall": float(rec),
            "precision": float(prec),
            "f1_score": float(f1),
            "support": int(supp),
            "errors": total_errors,
            "error_percentage": float(total_errors / max(1, supp) * 100),
            "top_confusions": [{"target": target, "count": cnt, "pct": float(cnt / max(1, supp) * 100)} for target, cnt in confused_with[:3]]
        })
    
    # Sort by recall (weakest first)
    ranked_by_recall = sorted(ranked_classes, key=lambda x: x["recall"])
    
    # 2. Extract Top 10 Confusion Pairs
    all_pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                all_pairs.append({
                    "true_class": classes[i],
                    "predicted_class": classes[j],
                    "count": int(cm[i, j]),
                    "true_class_support": int(per_class[classes[i]]["support"])
                })
    all_pairs.sort(key=lambda x: x["count"], reverse=True)
    
    return {
        "weakest_classes": ranked_by_recall,
        "top_confusion_pairs": all_pairs[:10]
    }

def main():
    print("="*75)
    print("PHASE 1 & 2: DEEP DATA AUDIT & WEAK CLASS ANALYSIS")
    print("="*75)
    
    # 1. Quality Stats
    dog_quality = sample_image_stats(DATA / "manifest_dog_train.json")
    cat_quality = sample_image_stats(DATA / "manifest_cat_train.json")
    print(f"Dog Image Quality: Aspect Ratio={dog_quality['mean_aspect_ratio']:.2f}, Blur(Laplacian)={dog_quality['mean_blur_laplacian']:.1f}, Brightness={dog_quality['mean_brightness']:.1f}, Contrast={dog_quality['mean_contrast']:.1f}")
    print(f"Cat Image Quality: Aspect Ratio={cat_quality['mean_aspect_ratio']:.2f}, Blur(Laplacian)={cat_quality['mean_blur_laplacian']:.1f}, Brightness={cat_quality['mean_brightness']:.1f}, Contrast={cat_quality['mean_contrast']:.1f}")
    
    # 2. Weak Classes
    dog_analysis = analyze_weak_classes_and_confusion("dog", DOG_CLASSES)
    cat_analysis = analyze_weak_classes_and_confusion("cat", CAT_CLASSES)
    
    print("\n--- WEAKEST 5 DOG CLASSES ---")
    for wc in dog_analysis["weakest_classes"][:5]:
        print(f"  {wc['class_name']:<20}: Recall={wc['recall']*100:.1f}%, F1={wc['f1_score']:.4f}, Support={wc['support']}, Top Confusion={wc['top_confusions'][0] if wc['top_confusions'] else 'None'}")
        
    print("\n--- WEAKEST 5 CAT CLASSES ---")
    for wc in cat_analysis["weakest_classes"][:5]:
        print(f"  {wc['class_name']:<20}: Recall={wc['recall']*100:.1f}%, F1={wc['f1_score']:.4f}, Support={wc['support']}, Top Confusion={wc['top_confusions'][0] if wc['top_confusions'] else 'None'}")
        
    print("\n--- TOP 5 CONFUSION PAIRS (DOG) ---")
    for p in dog_analysis["top_confusion_pairs"][:5]:
        print(f"  {p['true_class']:<20} -> {p['predicted_class']:<20} : {p['count']} times")
        
    print("\n--- TOP 5 CONFUSION PAIRS (CAT) ---")
    for p in cat_analysis["top_confusion_pairs"][:5]:
        print(f"  {p['true_class']:<20} -> {p['predicted_class']:<20} : {p['count']} times")
        
    out = {
        "quality_metrics": {"dog": dog_quality, "cat": cat_quality},
        "dog_analysis": dog_analysis,
        "cat_analysis": cat_analysis
    }
    with open(DATA / "deep_data_and_error_audit.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("\n[DONE] Saved to: data/deep_data_and_error_audit.json")

if __name__ == "__main__":
    main()

