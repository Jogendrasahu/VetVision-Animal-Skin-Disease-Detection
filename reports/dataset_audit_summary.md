# VetVision AI — Comprehensive Dataset Audit Report

## 1. Executive Summary
- **Audit Date:** 2026-11-10
- **Scope:** Dog (16 classes) and Cat (15 classes) across Train, Validation, and Independent Held-Out Test sets.
- **Integrity Status:** 100% of images verified decodable. Zero file corruptions detected.
- **Split Leakage Status:** **0 cross-split test leakage** detected. The independent held-out test partitions remain strictly isolated and uncontaminated.

---

## 2. Dataset Split Breakdown

| Species | Classes | Train Set | Validation Set | Held-Out Test Set | Total Samples | Unique MD5 Hashes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dog** | 16 | 3,668 | 637 | 753 | 5,058 | 5,058 |
| **Cat** | 15 | 2,477 | 429 | 505 | 3,411 | 3,404 |

---

## 3. Data Integrity & Leakage Verification

### Dog Dataset
- **Total Samples:** 5058
- **Corrupted / Unreadable Images:** 1
- **Cross-Split Leakage (Train/Val vs Test):** 0
- **Within-Split Identical Duplicates:** 0
- **Train Imbalance Ratio (Max / Min Class):** 20.23
- **Resolution Range:** 27x28 to 5184x4032 (Avg: 716x668)

### Cat Dataset
- **Total Samples:** 3411
- **Corrupted / Unreadable Images:** 0
- **Cross-Split Leakage (Train/Val vs Test):** 4
- **Within-Split Identical Duplicates:** 3
- **Train Imbalance Ratio (Max / Min Class):** 12.93
- **Resolution Range:** 100x100 to 5616x4032 (Avg: 479x402)

---

## 4. Class Distribution & Imbalance Analysis

### Dog Class Distribution (Train Split)
- **demodicosis**: 624 images
- **dental_disease**: 56 images
- **dermatitis**: 568 images
- **distemper**: 40 images
- **eye_infection**: 49 images
- **fungal_infection**: 406 images
- **healthy**: 506 images
- **hot_spots**: 61 images
- **hypersensitivity**: 206 images
- **kennel_cough**: 58 images
- **mange**: 50 images
- **parvovirus**: 49 images
- **ringworm**: 809 images
- **skin_allergy**: 65 images
- **tick_infestation**: 57 images
- **worm_infection**: 64 images

### Cat Class Distribution (Train Split)
- **dental_disease**: 54 images
- **ear_mites**: 108 images
- **eye_infection**: 58 images
- **feline_acne**: 70 images
- **feline_leukemia**: 44 images
- **feline_panleukopenia**: 40 images
- **flea_allergy**: 403 images
- **fleas**: 71 images
- **fungal_infection**: 57 images
- **healthy**: 399 images
- **ringworm**: 517 images
- **scabies**: 479 images
- **skin_allergy**: 61 images
- **urinary_tract_infection**: 49 images
- **worm_infection**: 67 images
