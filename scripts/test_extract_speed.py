"""
Test feature extraction speed for EfficientNetB0 on CPU
"""
import time, json
import numpy as np
import tensorflow as tf
from pathlib import Path

PROJECT = Path(r"c:\Users\Jogen\OneDrive\Desktop\Jogendra\Projects\Animal_Disease_App")
DATA = PROJECT / "data"

with open(DATA / "manifest_dog_train.json") as f:
    manifest = json.load(f)[:100]

base = tf.keras.applications.EfficientNetB0(
    include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')

paths = [item['path'] for item in manifest]
def parse_img(p):
    raw = tf.io.read_file(p)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [224, 224])
    return tf.cast(img, tf.float32)

ds = tf.data.Dataset.from_tensor_slices(paths).map(parse_img, num_parallel_calls=tf.data.AUTOTUNE).batch(32)

t0 = time.time()
feats = base.predict(ds, verbose=1)
elapsed = time.time() - t0
print(f"Extracted {len(feats)} features in {elapsed:.2f}s ({elapsed/len(feats)*1000:.1f} ms/img)")

