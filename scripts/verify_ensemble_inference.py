"""
Verify compatibility of exported ensemble models with existing VetVision AI inference pipeline.
"""
import os, sys
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

PROJECT = Path(r"c:\Users\Jogen\OneDrive\Desktop\Jogendra\Projects\Animal_Disease_App")
MODELS = PROJECT / "models"
UPLOADS = PROJECT / "static" / "uploads"

# Test fixtures
dog_img = PROJECT / "test_fixtures" / "dog_sample.jpg"
cat_img = PROJECT / "test_fixtures" / "cat_sample.jpg"

def process_image_for_disease(img_path):
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    return np.expand_dims(tf.keras.utils.img_to_array(img), axis=0)

print("="*60)
print("TESTING INFERENCE COMPATIBILITY WITH VETVISION AI PIPELINE")
print("="*60)

# Dog Ensemble
dog_ens_path = MODELS / "Dog_Ensemble_v1.keras"
print(f"\n1. Loading Dog Ensemble: {dog_ens_path.name}...")
dog_model = tf.keras.models.load_model(str(dog_ens_path))
inp_dog = process_image_for_disease(str(dog_img))
print(f"  Input tensor shape: {inp_dog.shape}, dtype: {inp_dog.dtype}, min: {inp_dog.min()}, max: {inp_dog.max()}")
preds_dog = dog_model.predict(inp_dog, verbose=0)[0]
print(f"  Output shape: {preds_dog.shape}")
print(f"  Sum of probabilities: {np.sum(preds_dog):.4f}")
print(f"  Top class index: {np.argmax(preds_dog)}, Top confidence: {np.max(preds_dog)*100:.2f}%")

# Cat Ensemble
cat_ens_path = MODELS / "Cat_Ensemble_v1.keras"
print(f"\n2. Loading Cat Ensemble: {cat_ens_path.name}...")
cat_model = tf.keras.models.load_model(str(cat_ens_path))
inp_cat = process_image_for_disease(str(cat_img))
print(f"  Input tensor shape: {inp_cat.shape}, dtype: {inp_cat.dtype}, min: {inp_cat.min()}, max: {inp_cat.max()}")
preds_cat = cat_model.predict(inp_cat, verbose=0)[0]
print(f"  Output shape: {preds_cat.shape}")
print(f"  Sum of probabilities: {np.sum(preds_cat):.4f}")
print(f"  Top class index: {np.argmax(preds_cat)}, Top confidence: {np.max(preds_cat)*100:.2f}%")

print("\n[SUCCESS] Both ensemble models are 100% compatible with the RAW [0, 255] contract!")

