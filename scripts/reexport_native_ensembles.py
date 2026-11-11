"""
Re-export Dog and Cat unified ensembles using native tf.keras.layers.Rescaling
to ensure 100% safe native Keras serialization and deserialization.
"""
import json
import tensorflow as tf
from pathlib import Path

PROJECT = Path(r"c:\Users\Jogen\OneDrive\Desktop\Jogendra\Projects\Animal_Disease_App")
DATA = PROJECT / "data"
MODELS = PROJECT / "models"

with open(DATA / "final_model_evaluation.json", "r", encoding="utf-8") as f:
    final_eval = json.load(f)

for species in ["dog", "cat"]:
    sp_cap = species.capitalize()
    print(f"\nRe-exporting {sp_cap} Unified Ensemble with native Rescaling layers...")
    sp_data = final_eval[species]
    weights = sp_data["candidates"]["Soft Ensemble (Weighted)"]["weights"]
    w0, w1, w2, w3 = weights[0], weights[1], weights[2], weights[3]
    print(f"  Weights: Baseline={w0:.3f}, CandA={w1:.3f}, CandB={w2:.3f}, CandC={w3:.3f}")

    unified_input = tf.keras.Input(shape=(224, 224, 3), name="raw_image_input")
    
    # 0. Baseline
    base_file = MODELS / f"{'Dog_EfficientNetB0_balanced_best' if species=='dog' else 'Cat_EfficientNetB0_stage2_best'}.keras"
    m0 = tf.keras.models.load_model(str(base_file))
    m0.trainable = False
    p0 = m0(unified_input)

    # 1. Candidate A
    cand_a_file = MODELS / f"{sp_cap}_EfficientNetB0_candidate_A.keras"
    m1 = tf.keras.models.load_model(str(cand_a_file))
    m1.trainable = False
    p1 = m1(unified_input)

    # 2. Candidate B
    cand_b_file = MODELS / f"{sp_cap}_MobileNetV2_candidate_B.keras"
    m2 = tf.keras.models.load_model(str(cand_b_file))
    m2.trainable = False
    p2 = m2(unified_input)

    # 3. Candidate C
    cand_c_file = MODELS / f"{sp_cap}_DenseNet121_candidate_C.keras"
    m3 = tf.keras.models.load_model(str(cand_c_file))
    m3.trainable = False
    p3 = m3(unified_input)

    # Native Rescaling layers for scalar multiplication
    s0 = tf.keras.layers.Rescaling(scale=w0, name="weight_baseline")(p0)
    s1 = tf.keras.layers.Rescaling(scale=w1, name="weight_cand_a")(p1)
    s2 = tf.keras.layers.Rescaling(scale=w2, name="weight_cand_b")(p2)
    s3 = tf.keras.layers.Rescaling(scale=w3, name="weight_cand_c")(p3)

    unified_output = tf.keras.layers.Add(name="weighted_soft_vote")([s0, s1, s2, s3])

    unified_model = tf.keras.Model(
        inputs=unified_input, outputs=unified_output,
        name=f"{sp_cap}_Unified_Weighted_Ensemble"
    )

    out_file = MODELS / f"{sp_cap}_Ensemble_v1.keras"
    unified_model.save(str(out_file))
    print(f"  Saved {out_file.name} successfully!")

    # Verify native reload
    print(f"  Testing native reload of {out_file.name}...")
    reloaded = tf.keras.models.load_model(str(out_file))
    print(f"  [OK] Successfully reloaded {out_file.name} with standard load_model()!")

