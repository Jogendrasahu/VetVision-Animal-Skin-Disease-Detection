import os
import io
import csv
import json
import uuid
import sqlite3
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify, g, Response
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mn2_preprocess, decode_predictions

from disease_database import DISEASE_INFO, get_disease_info

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CONFIDENCE_THRESHOLD = 35.0
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.webm', '.mov', '.avi', '.mkv'}
MIN_IMAGE_DIMENSION = 32  # Minimum pixel width/height for quality check

# ── Database Configuration ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
DATABASE_PATH = os.path.join(DATA_DIR, 'anivision.db')

# ── 1. Load ML Models (Production Ensembles with Baseline Fallback) ──
print("Loading AI Models... This might take a minute.")
dog_model_path = os.path.join(BASE_DIR, 'models', 'Dog_Ensemble_v1.keras')
if not os.path.exists(dog_model_path):
    dog_model_path = os.path.join(BASE_DIR, 'models', 'Dog_EfficientNetB0_balanced_best.keras')
dog_model = tf.keras.models.load_model(dog_model_path)

cat_model_path = os.path.join(BASE_DIR, 'models', 'Cat_Ensemble_v1.keras')
if not os.path.exists(cat_model_path):
    cat_model_path = os.path.join(BASE_DIR, 'models', 'Cat_EfficientNetB0_stage2_best.keras')
cat_model = tf.keras.models.load_model(cat_model_path)

cow_model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'models', 'Cow_EfficientNetB0_stage2_best.keras'))

# Gatekeeper AI for Animal Verification
print("Loading Gatekeeper AI...")
verifier_model = MobileNetV2(weights='imagenet')
print("[OK] All Models Loaded Successfully!")

# ── 2. Disease Class Mappings (UNCHANGED) ──
DOG_CLASSES = ['demodicosis', 'dental_disease', 'dermatitis', 'distemper', 'eye_infection', 'fungal_infection', 'healthy', 'hot_spots', 'hypersensitivity', 'kennel_cough', 'mange', 'parvovirus', 'ringworm', 'skin_allergy', 'tick_infestation', 'worm_infection']
CAT_CLASSES = ['dental_disease', 'ear_mites', 'eye_infection', 'feline_acne', 'feline_leukemia', 'feline_panleukopenia', 'flea_allergy', 'fleas', 'fungal_infection', 'healthy', 'ringworm', 'scabies', 'skin_allergy', 'urinary_tract_infection', 'worm_infection']
COW_CLASSES = ['foot_and_mouth', 'healthy', 'lumpy_skin_disease']

# ── 3. Disease Information Database ──
# Imported from disease_database.py (Comprehensive 11-field clinical database)

# ── 4. Image Processing & Quality Assessment ──
def process_image_for_disease(img_path):
    img = Image.open(img_path).convert('RGB').resize((224, 224)) 
    return np.expand_dims(tf.keras.utils.img_to_array(img), axis=0)

def assess_image_quality(img_input):
    """
    Evaluates visual quality of input image (file path, PIL Image, or numpy array).
    Checks dimensions, sharpness/blur via Laplacian variance, brightness, and contrast.
    Returns (quality_score: float [0-100], metrics: dict, warnings: list[str]).
    """
    if isinstance(img_input, str):
        with Image.open(img_input) as pil_img:
            img_rgb = np.array(pil_img.convert('RGB'))
    elif isinstance(img_input, Image.Image):
        img_rgb = np.array(img_input.convert('RGB'))
    elif isinstance(img_input, np.ndarray):
        img_rgb = img_input
    else:
        raise ValueError("Unsupported image input type for quality assessment.")

    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) if (len(img_rgb.shape) == 3 and img_rgb.shape[2] == 3) else img_rgb

    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_bright = float(np.mean(gray))
    contrast_std = float(np.std(gray))

    metrics = {
        'blur_score': round(blur_var, 2),
        'brightness': round(mean_bright, 2),
        'contrast': round(contrast_std, 2),
        'width': int(w),
        'height': int(h),
        'dimensions': f"{w}x{h}"
    }

    warnings = []
    score = 100.0

    # Dimension check
    if w < 224 or h < 224:
        warnings.append(f"Image resolution ({w}x{h}px) is below optimal 224x224px.")
        score -= min(30.0, (224 - min(w, h)) * 0.15)

    # Blur assessment
    if blur_var < 30.0:
        warnings.append("Image is blurry or lacks edge detail.")
        score -= 35.0
    elif blur_var < 75.0:
        warnings.append("Image has slight blur; higher sharpness recommended.")
        score -= 15.0

    # Brightness assessment
    if mean_bright < 35.0:
        warnings.append("Image is underexposed or very dark.")
        score -= 30.0
    elif mean_bright > 225.0:
        warnings.append("Image is overexposed or washed out.")
        score -= 25.0

    # Contrast assessment
    if contrast_std < 12.0:
        warnings.append("Image lacks dynamic range/contrast or is uniform.")
        score -= 35.0

    quality_score = max(5.0, min(100.0, round(score, 1)))
    return quality_score, metrics, warnings

# ── 5. Gatekeeper AI Species Identification & Out-of-Distribution Defense ──
GATEKEEPER_MIN_CONFIDENCE = 0.22
GATEKEEPER_MIN_MARGIN = 0.05

# Official ImageNet class index mapping
IMAGENET_CLASS_PATH = os.path.join(os.path.expanduser('~'), '.keras', 'datasets', 'imagenet_class_index.json')
if not os.path.exists(IMAGENET_CLASS_PATH):
    IMAGENET_CLASS_PATH = tf.keras.utils.get_file(
        'imagenet_class_index.json',
        'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
    )

with open(IMAGENET_CLASS_PATH, 'r') as _f:
    _raw_imagenet_idx = json.load(_f)

# Explicit label extraction for canine (125), feline (13), and bovine (3)
CANINE_CLASS_LABELS = {_raw_imagenet_idx[str(i)][1] for i in range(151, 276)}
FELINE_CLASS_LABELS = {_raw_imagenet_idx[str(i)][1] for i in range(281, 294)}
BOVINE_CLASS_LABELS = {_raw_imagenet_idx[str(i)][1] for i in [345, 346, 347]}

# Complete semantic taxonomy dictionary mapping index -> metadata
IMAGENET_TAXONOMY = {}
DOG_SPECIES_INDICES = set()
CAT_SPECIES_INDICES = set()
COW_SPECIES_INDICES = set()
OTHER_ORGANISM_INDICES = set()
INANIMATE_INDICES = set()

for _idx_str, (_synset_id, _class_name) in _raw_imagenet_idx.items():
    _idx = int(_idx_str)
    if _class_name in CANINE_CLASS_LABELS:
        _cat = 'canine'
        DOG_SPECIES_INDICES.add(_idx)
    elif _class_name in FELINE_CLASS_LABELS:
        _cat = 'feline'
        CAT_SPECIES_INDICES.add(_idx)
    elif _class_name in BOVINE_CLASS_LABELS:
        _cat = 'bovine'
        COW_SPECIES_INDICES.add(_idx)
    elif _idx < 398:
        _cat = 'other_organism'
        OTHER_ORGANISM_INDICES.add(_idx)
    else:
        _cat = 'inanimate'
        INANIMATE_INDICES.add(_idx)

    IMAGENET_TAXONOMY[_idx] = {
        'synset': _synset_id,
        'label': _class_name,
        'category': _cat
    }

SUPPORTED_SPECIES_INDICES = DOG_SPECIES_INDICES | CAT_SPECIES_INDICES | COW_SPECIES_INDICES

def identify_species(img_input, selected_animal=None):
    """
    Tier-1 Gatekeeper Vision Engine:
    1. Evaluates 1000-class ImageNet probability distribution via MobileNetV2.
    2. Maps classes using explicit semantic taxonomy (canine, feline, bovine, other_organism, inanimate).
    3. Identifies whether subject is an unsupported object/screen/animal, uncertain, or verified species.
    4. Distinguishes uncalibrated softmax scores from calibrated probabilities.
    5. Validates candidate against selected animal model or enables automatic routing.
    
    Returns structured dict:
      'is_verified': bool,
      'status': 'species_match' | 'species_mismatch' | 'species_uncertain' | 'unsupported_object',
      'detected_species': 'Dog' | 'Cat' | 'Cow' | 'Unknown' | 'None',
      'confidence': float (0.0 to 100.0 uncalibrated softmax heuristic score),
      'softmax_score': float,
      'is_calibrated': bool (False),
      'calibration_disclosure': str,
      'top_label': str,
      'top_category': str,
      'decision': str,
      'message': str,
      'species_probabilities': {'dog': float, 'cat': float, 'cow': float, 'other': float}
    """
    if isinstance(img_input, str):
        img = Image.open(img_input).convert('RGB').resize((224, 224))
    elif isinstance(img_input, Image.Image):
        img = img_input.convert('RGB').resize((224, 224))
    elif isinstance(img_input, np.ndarray):
        img = Image.fromarray(img_input).convert('RGB').resize((224, 224))
    else:
        raise ValueError("Unsupported image input type for species identification.")

    x = np.expand_dims(tf.keras.utils.img_to_array(img), axis=0)
    x = mn2_preprocess(x)

    preds = verifier_model.predict(x, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    top_meta = IMAGENET_TAXONOMY.get(top_idx, {'category': 'inanimate', 'label': f'class_{top_idx}'})
    top_label = top_meta['label']
    top_category = top_meta['category']
    top_conf = float(preds[top_idx])

    dog_prob = float(np.sum(preds[list(DOG_SPECIES_INDICES)]))
    cat_prob = float(np.sum(preds[list(CAT_SPECIES_INDICES)]))
    cow_prob = float(np.sum(preds[list(COW_SPECIES_INDICES)]))
    supported_prob = dog_prob + cat_prob + cow_prob
    other_prob = max(0.0, 1.0 - supported_prob)

    scores = {'dog': dog_prob, 'cat': cat_prob, 'cow': cow_prob}
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    # Prioritize specific Top-1 ImageNet category when it directly matches a supported species
    if top_idx in COW_SPECIES_INDICES:
        top_species = 'cow'
        top_prob = cow_prob
        runner_up_species = 'dog' if dog_prob > cat_prob else 'cat'
        runner_up_prob = max(dog_prob, cat_prob)
    elif top_idx in CAT_SPECIES_INDICES:
        top_species = 'cat'
        top_prob = cat_prob
        runner_up_species = 'dog' if dog_prob > cow_prob else 'cow'
        runner_up_prob = max(dog_prob, cow_prob)
    elif top_idx in DOG_SPECIES_INDICES and dog_prob >= 0.20:
        top_species = 'dog'
        top_prob = dog_prob
        runner_up_species = 'cat' if cat_prob > cow_prob else 'cow'
        runner_up_prob = max(cat_prob, cow_prob)
    else:
        top_species, top_prob = sorted_scores[0]
        runner_up_species, runner_up_prob = sorted_scores[1]

    margin = top_prob - runner_up_prob
    is_supported_top = top_idx in SUPPORTED_SPECIES_INDICES

    # Logging
    print(f"\n[Gatekeeper] Global Top-1: {top_label} ({top_conf:.4f}, cat: {top_category}) | Supported Sum: {supported_prob:.4f}")
    print(f"[Gatekeeper] Scores -> Dog: {dog_prob:.4f}, Cat: {cat_prob:.4f}, Cow: {cow_prob:.4f}, Other: {other_prob:.4f}")
    print(f"[Gatekeeper] Top Candidate: {top_species} ({top_prob:.4f}) | Runner-Up: {runner_up_species} ({runner_up_prob:.4f}) | Margin: {margin:.4f}")

    # 1. Non-animal / Unsupported object / Unsupported animal check:
    # A) Total supported animal probability is very low (< 16%)
    # B) Or Top-1 is classified as inanimate (electronics, vehicles, tools, screenshots, clothing, furniture) with supported probability < 35%
    # C) Or Top-1 is an unsupported animal (e.g. horse 'sorrel', sheep 'ram', camel) with confidence exceeding supported candidates
    # D) Or Top-1 is not a supported species and supported animal probability has no dominant single-class signal (< 40%)
    is_unsupported = False
    if supported_prob < 0.16:
        is_unsupported = True
    elif (top_category == 'inanimate') and (supported_prob < 0.35):
        is_unsupported = True
    elif (top_category == 'other_organism') and (top_conf > top_prob * 1.1 or other_prob > 0.50):
        is_unsupported = True

    if is_unsupported:
        decision = "REJECT — UNSUPPORTED SUBJECT / NON-ANIMAL"
        msg = "No supported animal detected. Please upload a clear photo of a Dog, Cat or Cow."
        print(f"[Gatekeeper] Decision: {decision}")
        return {
            'is_verified': False,
            'status': 'unsupported_object',
            'detected_species': 'None',
            'confidence': round(supported_prob * 100, 1),
            'softmax_score': round(supported_prob * 100, 1),
            'is_calibrated': False,
            'calibration_disclosure': 'Uncalibrated ImageNet softmax heuristic score; not an independently calibrated posterior probability.',
            'top_label': top_label,
            'top_category': top_category,
            'decision': decision,
            'message': msg,
            'species_probabilities': {'dog': round(dog_prob, 4), 'cat': round(cat_prob, 4), 'cow': round(cow_prob, 4), 'other': round(other_prob, 4)}
        }

    # 2. Species Uncertainty Check:
    if (not is_supported_top and top_prob < 0.40) or top_prob < GATEKEEPER_MIN_CONFIDENCE or (top_prob < 0.38 and margin < GATEKEEPER_MIN_MARGIN):
        decision = "REJECT — SPECIES UNCERTAIN"
        msg = "Animal detected, but the species could not be identified confidently. Please upload a clearer photo showing the animal clearly."
        print(f"[Gatekeeper] Decision: {decision}")
        return {
            'is_verified': False,
            'status': 'species_uncertain',
            'detected_species': 'Unknown',
            'confidence': round(top_prob * 100, 1),
            'softmax_score': round(top_prob * 100, 1),
            'is_calibrated': False,
            'calibration_disclosure': 'Uncalibrated ImageNet softmax heuristic score; not an independently calibrated posterior probability.',
            'top_label': top_label,
            'top_category': top_category,
            'decision': decision,
            'message': msg,
            'species_probabilities': {'dog': round(dog_prob, 4), 'cat': round(cat_prob, 4), 'cow': round(cow_prob, 4), 'other': round(other_prob, 4)}
        }

    detected_name = top_species.title()

    # 3. Selected Species Validation (if a specific species was selected):
    if selected_animal and selected_animal.lower() != 'auto':
        if selected_animal.lower() != top_species:
            decision = "REJECT — SPECIES MISMATCH"
            msg = f"Species Mismatch Detected. You selected '{selected_animal.title()}', but the uploaded image appears to contain a {detected_name} (softmax score: {top_prob*100:.1f}%). Please select the correct animal model."
            print(f"[Gatekeeper] Decision: {decision}")
            return {
                'is_verified': False,
                'status': 'species_mismatch',
                'detected_species': detected_name,
                'selected_species': selected_animal.title(),
                'confidence': round(top_prob * 100, 1),
                'softmax_score': round(top_prob * 100, 1),
                'is_calibrated': False,
                'calibration_disclosure': 'Uncalibrated ImageNet softmax heuristic score; not an independently calibrated posterior probability.',
                'top_label': top_label,
                'top_category': top_category,
                'decision': decision,
                'message': msg,
                'species_probabilities': {'dog': round(dog_prob, 4), 'cat': round(cat_prob, 4), 'cow': round(cow_prob, 4), 'other': round(other_prob, 4)}
            }

    # 4. Verified Match
    decision = "ALLOW — SPECIES MATCH"
    msg = f"Verified {detected_name} (softmax score: {top_prob*100:.1f}%)."
    print(f"[Gatekeeper] Decision: {decision}")
    return {
        'is_verified': True,
        'status': 'species_match',
        'detected_species': detected_name,
        'selected_species': (selected_animal or detected_name).title(),
        'confidence': round(top_prob * 100, 1),
        'softmax_score': round(top_prob * 100, 1),
        'is_calibrated': False,
        'calibration_disclosure': 'Uncalibrated ImageNet softmax heuristic score; not an independently calibrated posterior probability.',
        'top_label': top_label,
        'top_category': top_category,
        'decision': decision,
        'message': msg,
        'species_probabilities': {'dog': round(dog_prob, 4), 'cat': round(cat_prob, 4), 'cow': round(cow_prob, 4), 'other': round(other_prob, 4)}
    }

def verify_animal(img_path, selected_animal):
    """Backward-compatible wrapper around identify_species."""
    res = identify_species(img_path, selected_animal)
    return res['is_verified'], res['status'], res['detected_species'], res['message']


# ══════════════════════════════════════════════════════════════════
# ── 6. SQLite Persistent Database ──
# ══════════════════════════════════════════════════════════════════

def init_database():
    """Create database and tables if they don't exist. Safe to call on every startup."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_history (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                animal TEXT NOT NULL,
                image_path TEXT NOT NULL,
                prediction TEXT NOT NULL,
                prediction_key TEXT NOT NULL,
                confidence REAL NOT NULL,
                top_predictions TEXT NOT NULL,
                gatekeeper_status TEXT NOT NULL DEFAULT 'species_match',
                low_confidence INTEGER NOT NULL DEFAULT 0,
                description TEXT,
                action TEXT,
                status TEXT NOT NULL DEFAULT 'completed'
            )
        ''')
        conn.commit()
        conn.close()
        print(f"[DB] Database initialized at {DATABASE_PATH}")
    except Exception as e:
        print(f"[DB] Database initialization error: {e}")

def get_db():
    """Get database connection for current request context."""
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exception):
    """Close database connection at end of request."""
    db = g.pop('db', None)
    if db is not None:
        db.close()

# Initialize database on startup
init_database()


# ══════════════════════════════════════════════════════════════════
# ── 7. Multi-Page Web Routes & Navigation ──
# ══════════════════════════════════════════════════════════════════

@app.route('/', methods=['GET'])
def index():
    """Home landing page with hero, workflow, model cards, and trust metrics."""
    return render_template('index.html')

@app.route('/detect', methods=['GET'])
def detect_page():
    """Detection studio page supporting single-image, multi-image, video, and camera workflows."""
    return render_template('detect.html')

@app.route('/detect-preview', methods=['GET'])
def detect_preview_page():
    """Diagnostic result dashboard preview with populated 11-field clinical report."""
    return render_template('detect_preview.html')

@app.route('/diseases', methods=['GET'])
def diseases_page():
    """Comprehensive disease catalog across Dog (16), Cat (15), and Cow (3) classes."""
    return render_template('diseases.html')

@app.route('/dataset', methods=['GET'])
def dataset_page():
    """Authentic dataset architecture, provenance, class distribution, and empirical benchmarks."""
    metrics_file = os.path.join(app.root_path, 'data', 'evaluation_metrics.json')
    dist_file = os.path.join(app.root_path, 'data', 'dataset_distribution.json')
    gk_file = os.path.join(app.root_path, 'gatekeeper_diversity_evaluation.json')

    metrics = {}
    distribution = {}
    gk_eval = []

    if os.path.exists(metrics_file):
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
    if os.path.exists(dist_file):
        with open(dist_file, 'r', encoding='utf-8') as f:
            distribution = json.load(f)
    if os.path.exists(gk_file):
        with open(gk_file, 'r', encoding='utf-8') as f:
            gk_eval = json.load(f)

    return render_template('dataset.html', eval_data=metrics, dist_data=distribution, gk_data=gk_eval)

@app.route('/history', methods=['GET'])
def history_page():
    """Dedicated prediction history analytics dashboard with sorting, filtering, and export."""
    return render_template('history.html')

@app.route('/guide', methods=['GET'])
def guide_page():
    """Educational photographic guide, confidence interpretation, triage indicators, and nearby vet finder."""
    return render_template('guide.html')

@app.route('/about', methods=['GET'])
def about_page():
    """Academic project overview, dual-tier architecture, MobileNetV2 Gatekeeper, and ethics framework."""
    return render_template('about.html')


# ══════════════════════════════════════════════════════════════════
# ── 8. ML Inference & Clinical Decision Engine ──
# ══════════════════════════════════════════════════════════════════

def run_disease_prediction(filepath, animal_type):
    """
    Core ML inference pipeline separated from HTTP handling:
    1. Preprocess image with RAW [0, 255] float RGB tensor (224x224).
    2. Run species-specific EfficientNetB0 model.
    3. Calculate Top-3 ranked differential predictions.
    4. Fetch standardized clinical disease profile.
    Returns dict with all clinical, diagnostic, and raw probability data.
    """
    img_array = process_image_for_disease(filepath)

    if animal_type == 'dog':
        predictions = dog_model.predict(img_array)
        classes = DOG_CLASSES
    elif animal_type == 'cat':
        predictions = cat_model.predict(img_array)
        classes = CAT_CLASSES
    elif animal_type == 'cow':
        predictions = cow_model.predict(img_array)
        classes = COW_CLASSES
    else:
        raise ValueError(f"Invalid animal type '{animal_type}'")

    probs = predictions[0]
    raw_probs = [float(p) for p in probs]
    top3_indices = np.argsort(probs)[::-1][:3]
    top3 = [
        {'disease': classes[i].replace('_', ' ').title(), 'confidence': round(float(probs[i]) * 100, 2)}
        for i in top3_indices
    ]

    class_idx = int(top3_indices[0])
    disease_key = classes[class_idx]
    confidence = round(float(probs[class_idx]) * 100, 2)

    low_confidence = confidence < CONFIDENCE_THRESHOLD
    medical_data = get_disease_info(disease_key)

    return {
        'disease_key': disease_key,
        'disease': disease_key.replace('_', ' ').title(),
        'confidence': confidence,
        'low_confidence': low_confidence,
        'top3': top3,
        'raw_probs': raw_probs,
        'classes': classes,
        'clinical_info': medical_data,
        'clinical_report': medical_data,
        'description': medical_data.get('description', ''),
        'action': medical_data.get('action', ''),
        'causes': medical_data.get('causes', ''),
        'symptoms': medical_data.get('symptoms', []),
        'affected_areas': medical_data.get('affected_areas', []),
        'contagiousness': medical_data.get('contagiousness', ''),
        'precautions': medical_data.get('precautions', []),
        'supportive_care': medical_data.get('supportive_care', ''),
        'when_to_contact_vet': medical_data.get('when_to_contact_vet', ''),
        'emergency_signs': medical_data.get('emergency_signs', []),
        'emergency_warning_signs': medical_data.get('emergency_signs', []),
        'similar_conditions': medical_data.get('similar_conditions', []),
        'next_steps': medical_data.get('next_steps', [])
    }


def save_prediction_record(animal_type, image_url, result_data):
    """Save a prediction result into SQLite database."""
    prediction_id = uuid.uuid4().hex[:16]
    timestamp = datetime.now().isoformat()
    try:
        db = get_db()
        db.execute(
            '''INSERT INTO prediction_history 
               (id, timestamp, animal, image_path, prediction, prediction_key, confidence, 
                top_predictions, gatekeeper_status, low_confidence, description, action, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                prediction_id,
                timestamp,
                animal_type.title(),
                image_url,
                result_data['disease'],
                result_data['disease_key'],
                result_data['confidence'],
                json.dumps(result_data['top3']),
                'species_match',
                1 if result_data['low_confidence'] else 0,
                result_data['description'],
                result_data['action'],
                'completed'
            )
        )
        db.commit()
        print(f"[DB] Prediction saved: {prediction_id} — {result_data['disease']} ({result_data['confidence']}%)")
        return prediction_id, timestamp
    except Exception as e:
        print(f"[DB] Error saving prediction: {e}")
        return None, timestamp


def handle_prediction_request():
    """Unified handler for single-image web frontend and mobile API requests."""
    file = request.files.get('file') or request.files.get('image')
    animal_type = (request.form.get('animal_type') or request.form.get('animal') or 'auto').lower().strip()
    
    if not file:
        return jsonify({'status': 'error', 'error': 'Missing file or image.'}), 400
    
    if animal_type not in ('dog', 'cat', 'cow', 'auto'):
        return jsonify({'status': 'error', 'error': f"Invalid animal type '{animal_type}'."}), 400
    
    if file.filename == '':
        return jsonify({'status': 'error', 'error': 'No file selected.'}), 400

    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'status': 'error', 'error': 'Unsupported file format. Please upload a JPG, JPEG, PNG, or WebP image.'}), 400

    # Generate unique filename to prevent overwrite conflicts
    unique_filename = f"{uuid.uuid4().hex[:12]}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)

    # Validate image integrity
    try:
        with Image.open(filepath) as img_check:
            img_check.verify()
    except Exception:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'status': 'error', 'error': 'The uploaded file appears to be corrupted or is not a valid image.'}), 400

    # Image quality check — minimum dimensions
    try:
        with Image.open(filepath) as img_q:
            width, height = img_q.size
    except Exception:
        if os.path.exists(filepath):
            try: os.remove(filepath)
            except OSError: pass
        return jsonify({'status': 'error', 'error': 'Unable to read image dimensions. The file may be corrupted.'}), 400

    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        if os.path.exists(filepath):
            try: os.remove(filepath)
            except OSError: pass
        return jsonify({'status': 'error', 'error': f'Image is too small ({width}×{height}px). Please upload a clearer, larger image for reliable analysis.'}), 400

    # Image Quality Evaluation
    try:
        quality_score, quality_metrics, quality_warnings = assess_image_quality(filepath)
    except Exception:
        quality_score, quality_metrics, quality_warnings = 75.0, {}, []

    # 1. Strict Gatekeeper Verification (MobileNetV2)
    res_gk = identify_species(filepath, animal_type)
    if not res_gk['is_verified']:
        return jsonify({
            'status': res_gk['status'],
            'gatekeeper_error': True,
            'selected_species': animal_type.title(),
            'detected_species': res_gk['detected_species'],
            'confidence': res_gk['confidence'],
            'message': res_gk['message'],
            'decision': res_gk['decision'],
            'species_probabilities': res_gk.get('species_probabilities', {}),
            'quality_score': quality_score,
            'quality_metrics': quality_metrics,
            'warnings': quality_warnings
        }), 200

    # Auto-route to the detected animal model if 'auto' was selected
    target_species = res_gk['detected_species'].lower() if animal_type == 'auto' else animal_type

    # 2. Run Disease Model ONLY IF Gatekeeper verified
    try:
        result_data = run_disease_prediction(filepath, target_species)
    except ValueError as ve:
        return jsonify({'status': 'error', 'error': str(ve)}), 400
    except Exception as e:
        print(f"[ML] Prediction execution error: {e}")
        return jsonify({'status': 'error', 'error': 'An error occurred during disease model analysis.'}), 500

    image_url = '/' + filepath.replace('\\', '/')

    # 3. Auto-save to SQLite prediction history
    prediction_id, timestamp = save_prediction_record(target_species, image_url, result_data)

    return jsonify({
        'status': 'success',
        'animal': target_species.title(),
        'auto_routed': (animal_type == 'auto'),
        'disease': result_data['disease'],
        'disease_key': result_data['disease_key'],
        'confidence': result_data['confidence'],
        'low_confidence': result_data['low_confidence'],
        'top3': result_data['top3'],
        'description': result_data['description'],
        'action': result_data['action'],
        'clinical_info': result_data.get('clinical_info', {}),
        'clinical_report': {
            'causes': result_data.get('causes', ''),
            'symptoms': result_data.get('symptoms', []),
            'affected_areas': result_data.get('affected_areas', []),
            'contagiousness': result_data.get('contagiousness', ''),
            'precautions': result_data.get('precautions', []),
            'supportive_care': result_data.get('supportive_care', ''),
            'when_to_contact_vet': result_data.get('when_to_contact_vet', ''),
            'emergency_signs': result_data.get('emergency_signs', []),
            'similar_conditions': result_data.get('similar_conditions', []),
            'next_steps': result_data.get('next_steps', [])
        },
        'quality_score': quality_score,
        'quality_metrics': quality_metrics,
        'warnings': quality_warnings,
        'image_url': image_url,
        'prediction_id': prediction_id,
        'timestamp': timestamp,
        'gatekeeper': {
            'status': 'species_match',
            'verified_species': target_species.title()
        }
    }), 200


# ══════════════════════════════════════════════════════════════════
# ── 9. Multi-Image Ingestion & Disagreement Analysis Engine ──
# ══════════════════════════════════════════════════════════════════

@app.route('/api/predict/multi', methods=['POST'])
def api_predict_multi():
    """
    Multi-image diagnostic screening with per-image species verification,
    ensemble mean probability distribution, and explicit disagreement detection.
    """
    animal_type = (request.form.get('animal_type') or request.form.get('animal') or 'auto').lower().strip()
    if not animal_type or animal_type not in ('dog', 'cat', 'cow', 'auto'):
        return jsonify({'status': 'error', 'error': 'Missing animal type selection (dog, cat, cow, auto).'}), 400

    uploaded_files = request.files.getlist('files') or request.files.getlist('files[]') or request.files.getlist('images') or request.files.getlist('images[]')
    if not uploaded_files or len(uploaded_files) == 0:
        return jsonify({'status': 'error', 'error': 'No images were provided for multi-image analysis.'}), 400

    if len(uploaded_files) > 5:
        return jsonify({'status': 'error', 'error': 'Maximum 5 images permitted per multi-view analysis.'}), 400

    verified_results = []
    rejected_reasons = []
    all_evaluations = []
    detected_species_counts = {}

    for idx, file in enumerate(uploaded_files):
        if not file or file.filename == '':
            continue

        filename = secure_filename(file.filename)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            reason = f"Unsupported file format '{ext}'."
            rejected_reasons.append(f"Image {idx+1} ({filename}): {reason}")
            all_evaluations.append({
                'index': idx + 1,
                'filename': filename,
                'image_url': '',
                'status': 'invalid_format',
                'detected_species': 'Unknown',
                'selected_species': animal_type.title(),
                'confidence': 0.0,
                'message': reason,
                'is_analyzed': False
            })
            continue

        unique_filename = f"multi_{uuid.uuid4().hex[:10]}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        rel_url = '/' + filepath.replace('\\', '/')

        # Integrity & dimension verification
        try:
            with Image.open(filepath) as img_check:
                img_check.verify()
            with Image.open(filepath) as img_q:
                w, h = img_q.size
        except Exception:
            reason = "Corrupted file or unreadable image data."
            rejected_reasons.append(f"Image {idx+1} ({filename}): {reason}")
            all_evaluations.append({
                'index': idx + 1,
                'filename': filename,
                'image_url': rel_url,
                'status': 'corrupted',
                'detected_species': 'Unknown',
                'selected_species': animal_type.title(),
                'confidence': 0.0,
                'message': reason,
                'is_analyzed': False
            })
            if os.path.exists(filepath):
                try: os.remove(filepath)
                except OSError: pass
            continue

        if w < MIN_IMAGE_DIMENSION or h < MIN_IMAGE_DIMENSION:
            reason = f"Resolution too small ({w}×{h}px). Minimum required is {MIN_IMAGE_DIMENSION}×{MIN_IMAGE_DIMENSION}px."
            rejected_reasons.append(f"Image {idx+1} ({filename}): {reason}")
            all_evaluations.append({
                'index': idx + 1,
                'filename': filename,
                'image_url': rel_url,
                'status': 'low_resolution',
                'detected_species': 'Unknown',
                'selected_species': animal_type.title(),
                'confidence': 0.0,
                'message': reason,
                'is_analyzed': False
            })
            if os.path.exists(filepath):
                try: os.remove(filepath)
                except OSError: pass
            continue

        # Gatekeeper per-image species identification
        res_gk = identify_species(filepath, animal_type)
        det_species = res_gk['detected_species']
        if det_species not in ('None', 'Unknown'):
            detected_species_counts[det_species] = detected_species_counts.get(det_species, 0) + 1

        eval_item = {
            'index': idx + 1,
            'filename': filename,
            'image_url': rel_url,
            'filepath': filepath,
            'status': res_gk['status'],
            'detected_species': det_species,
            'selected_species': animal_type.title(),
            'confidence': res_gk['confidence'],
            'message': res_gk['message'],
            'top_label': res_gk.get('top_label', ''),
            'is_analyzed': False
        }

        if not res_gk['is_verified']:
            rejected_reasons.append(f"Image {idx+1} ({filename}): {res_gk['message']}")
            all_evaluations.append(eval_item)
            continue

        target_species = det_species.lower() if animal_type == 'auto' else animal_type

        # Run disease model inference ONLY on verified species
        try:
            res = run_disease_prediction(filepath, target_species)
            eval_item.update({
                'is_analyzed': True,
                'target_species': target_species.title(),
                'disease': res['disease'],
                'disease_key': res['disease_key'],
                'disease_confidence': res['confidence'],
                'raw_probs': res['raw_probs'],
                'classes': res['classes']
            })
            verified_results.append({
                'index': idx + 1,
                'filename': filename,
                'image_url': rel_url,
                'filepath': filepath,
                'species': target_species,
                'disease': res['disease'],
                'disease_key': res['disease_key'],
                'confidence': res['confidence'],
                'raw_probs': res['raw_probs'],
                'classes': res['classes'],
                'full_res': res
            })
        except Exception as e:
            eval_item['status'] = 'inference_error'
            eval_item['message'] = f"Disease inference error: {e}"
            rejected_reasons.append(f"Image {idx+1} ({filename}): Inference error ({e}).")

        all_evaluations.append(eval_item)

    distinct_verified = set(r['species'] for r in verified_results)

    # If NO images could be verified for the selected animal:
    if not verified_results:
        unsupported_count = sum(1 for e in all_evaluations if e.get('status') == 'unsupported_object')
        mismatch_count = sum(1 for e in all_evaluations if e.get('status') == 'species_mismatch')
        uncertain_count = sum(1 for e in all_evaluations if e.get('status') == 'species_uncertain')
        distinct_detected = list(detected_species_counts.keys())

        if unsupported_count == len(all_evaluations):
            rejection_status = 'unsupported_object'
            rejection_msg = "No supported animal detected in any uploaded image. Please upload clear photos of a Dog, Cat, or Cow."
        elif len(distinct_detected) > 1:
            rejection_status = 'mixed_species'
            rejection_msg = f"Multiple distinct animal species ({', '.join(distinct_detected)}) were detected. VetVision AI does not combine disease probabilities across different species. Please submit images of a single animal species."
        elif mismatch_count > 0:
            rejection_status = 'species_mismatch'
            majority_species = max(detected_species_counts.items(), key=lambda x: x[1])[0] if detected_species_counts else 'another animal'
            rejection_msg = f"Species Mismatch Detected. You selected '{animal_type.title()}', but uploaded images were identified as {majority_species}. Please switch to the {majority_species} model."
        elif uncertain_count > 0:
            rejection_status = 'species_uncertain'
            rejection_msg = "Animal detected, but species could not be identified with sufficient confidence across the uploaded images. Please upload clearer photos."
        else:
            rejection_status = 'species_mismatch'
            rejection_msg = "None of the uploaded images could be verified for the selected species model."

        return jsonify({
            'status': rejection_status,
            'gatekeeper_error': True,
            'message': rejection_msg,
            'detected_species': distinct_detected[0] if len(distinct_detected) == 1 else ('None' if rejection_status == 'unsupported_object' else 'Unknown'),
            'selected_species': animal_type.title(),
            'total_images_submitted': len(uploaded_files),
            'images_analyzed': 0,
            'details': rejected_reasons,
            'image_breakdown': all_evaluations
        }), 200

    # Strict Mixed Species Isolation: If images from multiple species were verified in 'auto' mode
    if len(distinct_verified) > 1:
        return jsonify({
            'status': 'mixed_species_isolated',
            'gatekeeper_error': False,
            'message': f"Multiple distinct species detected ({', '.join(s.title() for s in distinct_verified)}). In accordance with clinical safety principles, VetVision AI will not average disease probabilities across different species. Each image has been evaluated strictly using its corresponding species classifier.",
            'total_images_submitted': len(uploaded_files),
            'images_analyzed': len(verified_results),
            'has_cross_species_consensus': False,
            'image_breakdown': all_evaluations
        }), 200

    effective_species = list(distinct_verified)[0]
    # Probability ensemble aggregation ONLY among verified images of the matching species
    classes = verified_results[0]['classes']
    prob_matrix = np.array([r['raw_probs'] for r in verified_results])
    mean_probs = np.mean(prob_matrix, axis=0)

    top3_indices = np.argsort(mean_probs)[::-1][:3]
    consensus_top3 = [
        {'disease': classes[i].replace('_', ' ').title(), 'confidence': round(float(mean_probs[i]) * 100, 2)}
        for i in top3_indices
    ]

    primary_idx = int(top3_indices[0])
    primary_disease_key = classes[primary_idx]
    primary_disease = primary_disease_key.replace('_', ' ').title()
    primary_confidence = round(float(mean_probs[primary_idx]) * 100, 2)

    # Disagreement & Consensus Metric Analysis among verified images
    num_verified = len(verified_results)
    agreeing_images = [r for r in verified_results if r['disease_key'] == primary_disease_key]
    agreement_count = len(agreeing_images)
    agreement_rate = round((agreement_count / num_verified) * 100, 1)

    has_disagreement = (agreement_count < num_verified)
    disagreement_details = []

    if has_disagreement:
        for r in verified_results:
            if r['disease_key'] != primary_disease_key:
                disagreement_details.append(f"Image {r['index']} suggested {r['disease']} ({r['confidence']}%)")

    # Check for mixed species / excluded images among total uploads
    has_mixed_species = len(verified_results) < len(all_evaluations)
    mixed_species_notice = ""
    if has_mixed_species:
        num_excluded = len(all_evaluations) - len(verified_results)
        mixed_species_notice = (
            f"Notice: {num_excluded} of {len(all_evaluations)} image(s) did not match the verified {effective_species.title()} "
            f"species and were strictly excluded from clinical consensus. Only {num_verified} verified {effective_species.title()} "
            f"image(s) were analyzed."
        )

    # Select representative primary preview image (highest confidence for winning class)
    best_item = max(verified_results, key=lambda x: x['raw_probs'][primary_idx])
    primary_image_url = best_item['image_url']

    medical_data = get_disease_info(primary_disease_key)
    result_data = {
        'disease': primary_disease,
        'disease_key': primary_disease_key,
        'confidence': primary_confidence,
        'low_confidence': primary_confidence < CONFIDENCE_THRESHOLD,
        'top3': consensus_top3,
        'description': medical_data['description'],
        'action': medical_data['action']
    }

    prediction_id, timestamp = save_prediction_record(effective_species, primary_image_url, result_data)

    per_image_breakdown = [
        {
            'index': r['index'],
            'image_url': r['image_url'],
            'disease': r['disease'],
            'confidence': r['confidence'],
            'agrees_with_consensus': (r['disease_key'] == primary_disease_key),
            'status': 'verified',
            'detected_species': effective_species.title()
        }
        for r in verified_results
    ]

    return jsonify({
        'status': 'success',
        'mode': 'multi_image',
        'animal': effective_species.title(),
        'auto_routed': (animal_type == 'auto'),
        'disease': primary_disease,
        'disease_key': primary_disease_key,
        'confidence': primary_confidence,
        'low_confidence': result_data['low_confidence'],
        'top3': consensus_top3,
        'image_url': primary_image_url,
        'images_analyzed': num_verified,
        'total_images_submitted': len(uploaded_files),
        'consensus_agreement': agreement_rate,
        'agreement_score': round(agreement_rate / 100.0, 2),
        'consensus_status': 'consensus' if not has_disagreement else 'divergent',
        'has_disagreement': has_disagreement,
        'disagreement_details': disagreement_details,
        'has_mixed_species': has_mixed_species,
        'mixed_species_notice': mixed_species_notice,
        'image_breakdown': all_evaluations,
        'verified_breakdown': per_image_breakdown,
        'frame_predictions': per_image_breakdown,
        'rejected_reasons': rejected_reasons,
        'description': medical_data['description'],
        'action': medical_data['action'],
        'clinical_report': {
            'causes': medical_data.get('causes', ''),
            'symptoms': medical_data.get('symptoms', []),
            'affected_areas': medical_data.get('affected_areas', []),
            'contagiousness': medical_data.get('contagiousness', ''),
            'precautions': medical_data.get('precautions', []),
            'supportive_care': medical_data.get('supportive_care', ''),
            'when_to_contact_vet': medical_data.get('when_to_contact_vet', ''),
            'emergency_signs': medical_data.get('emergency_signs', []),
            'similar_conditions': medical_data.get('similar_conditions', []),
            'next_steps': medical_data.get('next_steps', [])
        },
        'prediction_id': prediction_id,
        'timestamp': timestamp,
        'gatekeeper': {
            'status': 'species_match',
            'verified_species': animal_type.title()
        }
    }), 200


# ══════════════════════════════════════════════════════════════════
# ── 10. Video Ingestion & Controlled Keyframe Temporal Pooling ──
# ══════════════════════════════════════════════════════════════════

@app.route('/api/predict/video', methods=['POST'])
def api_predict_video():
    """
    Video screening engine:
    1. Uniformly extracts keyframes across the video stream.
    2. Filters out motion blur via Laplacian variance.
    3. Executes Gatekeeper species verification per candidate frame.
    4. Computes temporal consensus probability distribution.
    5. Returns transparent screening report with explicit video diagnostic caveats.
    """
    if 'video' not in request.files and 'file' not in request.files:
        return jsonify({'status': 'error', 'error': 'No video file provided.'}), 400

    video_file = request.files.get('video') or request.files.get('file')
    animal_type = (request.form.get('animal_type') or request.form.get('animal') or '').lower().strip()

    if not animal_type or animal_type not in ('dog', 'cat', 'cow'):
        return jsonify({'status': 'error', 'error': 'Valid animal type (dog, cat, cow) is required.'}), 400

    if video_file.filename == '':
        return jsonify({'status': 'error', 'error': 'No video file selected.'}), 400

    filename = secure_filename(video_file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({'status': 'error', 'error': f"Unsupported video format '{ext}'. Allowed: MP4, WebM, MOV, AVI."}), 400

    unique_video_name = f"vid_{uuid.uuid4().hex[:10]}_{filename}"
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_video_name)
    video_file.save(video_path)

    # OpenCV Video Capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if os.path.exists(video_path):
            try: os.remove(video_path)
            except OSError: pass
        return jsonify({'status': 'error', 'error': 'Failed to decode video file. The codec or container may be unreadable or corrupt.'}), 400

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    duration_sec = total_frames / fps if fps > 0 else 0

    if total_frames < 5:
        cap.release()
        if os.path.exists(video_path):
            try: os.remove(video_path)
            except OSError: pass
        return jsonify({'status': 'error', 'error': 'Video contains insufficient frames for diagnostic analysis (minimum 5 frames required).'}), 400

    # Uniform keyframe sampling (Sample 8 to 12 frames across video duration)
    sample_count = min(12, max(6, total_frames // 10))
    sample_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

    valid_frame_results = []
    frame_evaluations = []
    blurry_frames = []
    detected_species_set = set()
    unsupported_frames_count = 0
    mismatch_frames_count = 0

    for idx, f_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        timestamp_sec = round(float(f_idx) / fps, 2)

        # Blur assessment: Laplacian variance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_var < 20.0:
            blurry_frames.append({'frame': idx + 1, 'timestamp': f"{timestamp_sec}s", 'blur_var': round(blur_var, 1)})
            frame_evaluations.append({
                'frame': idx + 1,
                'timestamp': f"{timestamp_sec}s",
                'status': 'blurry',
                'detected_species': 'Unknown',
                'message': f"Frame skipped due to excessive motion blur (sharpness: {blur_var:.1f})."
            })
            continue

        # Save extracted frame as JPEG
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        frame_filename = f"frame_{unique_video_name}_{idx}.jpg"
        frame_path = os.path.join(app.config['UPLOAD_FOLDER'], frame_filename)
        pil_frame.save(frame_path, 'JPEG', quality=85)
        rel_frame_url = '/' + frame_path.replace('\\', '/')

        # Gatekeeper per-frame verification
        res_gk = identify_species(frame_path, animal_type)
        det_species = res_gk['detected_species']
        if det_species not in ('None', 'Unknown'):
            detected_species_set.add(det_species)

        frame_info = {
            'frame': idx + 1,
            'timestamp': f"{timestamp_sec}s",
            'image_url': rel_frame_url,
            'status': res_gk['status'],
            'detected_species': det_species,
            'confidence': res_gk['confidence'],
            'top_label': res_gk.get('top_label', ''),
            'message': res_gk['message']
        }

        if res_gk['status'] == 'unsupported_object':
            unsupported_frames_count += 1
            frame_evaluations.append(frame_info)
            continue
        elif res_gk['status'] == 'species_mismatch':
            mismatch_frames_count += 1
            frame_evaluations.append(frame_info)
            continue
        elif not res_gk['is_verified']:
            frame_evaluations.append(frame_info)
            continue

        # Run disease model inference on verified frame
        try:
            res = run_disease_prediction(frame_path, animal_type)
            frame_info.update({
                'predicted_disease': res['disease'],
                'disease_key': res['disease_key'],
                'confidence': res['confidence']
            })
            valid_frame_results.append({
                'frame_idx': idx + 1,
                'timestamp_sec': timestamp_sec,
                'frame_url': rel_frame_url,
                'frame_path': frame_path,
                'disease': res['disease'],
                'disease_key': res['disease_key'],
                'confidence': res['confidence'],
                'raw_probs': res['raw_probs'],
                'classes': res['classes']
            })
            frame_evaluations.append(frame_info)
        except Exception as e:
            frame_info['status'] = 'error'
            frame_info['message'] = str(e)
            frame_evaluations.append(frame_info)

    cap.release()

    # 1. Multi-Species Conflict Check across video frames:
    # If distinct supported animal species appear in the video, reject immediately
    if len(detected_species_set) > 1:
        return jsonify({
            'status': 'mixed_species',
            'gatekeeper_error': True,
            'message': f"Multiple different animal species ({', '.join(sorted(detected_species_set))}) detected across video frames. VetVision AI does not combine disease probabilities across different species. Please submit video footage focused on a single animal.",
            'detected_species': list(sorted(detected_species_set)),
            'frames_sampled': len(sample_indices),
            'frames_usable': 0,
            'frame_breakdown': frame_evaluations,
            'blur_count': len(blurry_frames)
        }), 200

    # 2. If insufficient verified frames, categorize rejection accurately
    if len(valid_frame_results) < 2:
        # A. Non-animal / screen recording / unsupported subject in video
        if unsupported_frames_count > 0 and len(detected_species_set) == 0:
            return jsonify({
                'status': 'unsupported_object',
                'gatekeeper_error': True,
                'message': 'No supported animal detected in this video. Please upload a clear video of a Dog, Cat or Cow.',
                'detected_species': 'None',
                'frames_sampled': len(sample_indices),
                'frames_usable': 0,
                'frame_breakdown': frame_evaluations,
                'blur_count': len(blurry_frames)
            }), 200

        # 3. Species mismatch (e.g. video shows a Cat when Dog model was selected)
        if mismatch_frames_count >= 2 or (len(detected_species_set) == 1 and animal_type.title() not in detected_species_set):
            maj_species = list(detected_species_set)[0] if detected_species_set else 'another animal'
            return jsonify({
                'status': 'species_mismatch',
                'gatekeeper_error': True,
                'detected_species': maj_species,
                'selected_species': animal_type.title(),
                'message': f"Species Mismatch Detected. You selected '{animal_type.title()}', but video frames consistently show a {maj_species}. Please switch to the {maj_species} model.",
                'frames_sampled': len(sample_indices),
                'frames_usable': 0,
                'frame_breakdown': frame_evaluations,
                'blur_count': len(blurry_frames)
            }), 200

        # 4. Excessive motion blur
        if len(blurry_frames) >= max(3, len(sample_indices) // 2):
            return jsonify({
                'status': 'species_uncertain',
                'gatekeeper_error': True,
                'message': f"Video contained excessive motion blur ({len(blurry_frames)} of {len(sample_indices)} frames discarded). Please record a steady, well-lit video with the animal clearly in frame.",
                'detected_species': 'Unknown',
                'frames_sampled': len(sample_indices),
                'frames_usable': len(valid_frame_results),
                'frame_breakdown': frame_evaluations,
                'blur_count': len(blurry_frames)
            }), 200

        # 5. Generic insufficient usable frames
        return jsonify({
            'status': 'species_uncertain',
            'gatekeeper_error': True,
            'message': f"Video quality was insufficient for reliable diagnosis. Fewer than 2 clear frames could be verified for {animal_type.title()}. Please record a steady, well-lit video with the animal clearly in frame.",
            'detected_species': 'Unknown',
            'frames_sampled': len(sample_indices),
            'frames_usable': len(valid_frame_results),
            'frame_breakdown': frame_evaluations,
            'blur_count': len(blurry_frames)
        }), 200

    # Aggregate valid frame probabilities (Temporal Mean Pooling)
    classes = valid_frame_results[0]['classes']
    prob_matrix = np.array([r['raw_probs'] for r in valid_frame_results])
    mean_probs = np.mean(prob_matrix, axis=0)

    top3_indices = np.argsort(mean_probs)[::-1][:3]
    video_top3 = [
        {'disease': classes[i].replace('_', ' ').title(), 'confidence': round(float(mean_probs[i]) * 100, 2)}
        for i in top3_indices
    ]

    primary_idx = int(top3_indices[0])
    primary_disease_key = classes[primary_idx]
    primary_disease = primary_disease_key.replace('_', ' ').title()
    primary_confidence = round(float(mean_probs[primary_idx]) * 100, 2)

    # Calculate frame agreement
    agreeing_frames = [r for r in valid_frame_results if r['disease_key'] == primary_disease_key]
    frame_agreement_pct = round((len(agreeing_frames) / len(valid_frame_results)) * 100, 1)

    # Best representative keyframe
    best_frame = max(valid_frame_results, key=lambda x: x['raw_probs'][primary_idx])
    preview_image_url = best_frame['frame_url']

    medical_data = get_disease_info(primary_disease_key)
    result_data = {
        'disease': primary_disease,
        'disease_key': primary_disease_key,
        'confidence': primary_confidence,
        'low_confidence': primary_confidence < CONFIDENCE_THRESHOLD,
        'top3': video_top3,
        'description': medical_data['description'],
        'action': medical_data['action']
    }

    prediction_id, timestamp = save_prediction_record(animal_type, preview_image_url, result_data)

    return jsonify({
        'status': 'success',
        'mode': 'video',
        'animal': animal_type.title(),
        'disease': primary_disease,
        'disease_key': primary_disease_key,
        'confidence': primary_confidence,
        'low_confidence': result_data['low_confidence'],
        'top3': video_top3,
        'image_url': preview_image_url,
        'video_duration_sec': round(duration_sec, 1),
        'frames_sampled': len(sample_indices),
        'frames_verified': len(valid_frame_results),
        'total_frames_extracted': len(valid_frame_results),
        'frame_agreement_pct': frame_agreement_pct,
        'agreement_score': round(frame_agreement_pct / 100.0, 2),
        'consensus_status': 'high_confidence' if frame_agreement_pct >= 70 else 'moderate_confidence',
        'frame_breakdown': frame_evaluations,
        'verified_frames': [
            {
                'frame': r['frame_idx'],
                'timestamp': f"{r['timestamp_sec']}s",
                'image_url': r['frame_url'],
                'predicted_disease': r['disease'],
                'confidence': r['confidence']
            }
            for r in valid_frame_results
        ],
        'clinical_disclaimer': 'Video screening provides an aggregated multi-frame impression. It is an experimental screening tool and does not guarantee definitive diagnostic coverage of uninspected lesions.',
        'description': medical_data['description'],
        'action': medical_data['action'],
        'clinical_report': {
            'causes': medical_data.get('causes', ''),
            'symptoms': medical_data.get('symptoms', []),
            'affected_areas': medical_data.get('affected_areas', []),
            'contagiousness': medical_data.get('contagiousness', ''),
            'precautions': medical_data.get('precautions', []),
            'supportive_care': medical_data.get('supportive_care', ''),
            'when_to_contact_vet': medical_data.get('when_to_contact_vet', ''),
            'emergency_signs': medical_data.get('emergency_signs', []),
            'similar_conditions': medical_data.get('similar_conditions', []),
            'next_steps': medical_data.get('next_steps', [])
        },
        'prediction_id': prediction_id,
        'timestamp': timestamp,
        'gatekeeper': {
            'status': 'species_match',
            'verified_species': animal_type.title()
        }
    }), 200

@app.route('/api/verify_species', methods=['POST'])
def api_verify_species():
    """Direct species identification endpoint without running disease inference."""
    if 'file' not in request.files and 'image' not in request.files:
        return jsonify({'status': 'error', 'error': 'No image file provided.'}), 400
    file = request.files.get('file') or request.files.get('image')
    selected = (request.form.get('animal_type') or request.form.get('animal') or 'auto').lower().strip()
    
    unique_filename = f"verify_{uuid.uuid4().hex[:8]}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    try:
        res = identify_species(filepath, selected)
        return jsonify(res), 200
    finally:
        if os.path.exists(filepath):
            try: os.remove(filepath)
            except OSError: pass


# ══════════════════════════════════════════════════════════════════
# ── 11. Legacy Endpoints & API Metadata ──
# ══════════════════════════════════════════════════════════════════

@app.route('/predict', methods=['POST'])
def predict():
    """Web UI prediction endpoint."""
    return handle_prediction_request()

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Mobile REST API prediction endpoint."""
    return handle_prediction_request()

@app.route('/api/diseases', methods=['GET'])
def api_diseases():
    """Public catalog endpoint returning all 34 classes across Dog, Cat, and Cow."""
    return jsonify({
        'status': 'success',
        'species_catalog': {
            'dog': {'count': len(DOG_CLASSES), 'classes': DOG_CLASSES},
            'cat': {'count': len(CAT_CLASSES), 'classes': CAT_CLASSES},
            'cow': {'count': len(COW_CLASSES), 'classes': COW_CLASSES}
        },
        'diseases': DISEASE_INFO
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check & configuration metadata endpoint for clients."""
    return jsonify({
        'status': 'healthy',
        'service': 'VetVision AI API',
        'version': '1.0.0',
        'database': os.path.exists(DATABASE_PATH),
        'supported_animals': {
            'dog': {'classes_count': len(DOG_CLASSES), 'classes': DOG_CLASSES},
            'cat': {'classes_count': len(CAT_CLASSES), 'classes': CAT_CLASSES},
            'cow': {'classes_count': len(COW_CLASSES), 'classes': COW_CLASSES}
        },
        'gatekeeper': 'MobileNetV2',
        'confidence_threshold': CONFIDENCE_THRESHOLD
    })

@app.route('/api/evaluation/metrics', methods=['GET'])
def api_evaluation_metrics():
    """Authentic empirical evaluation metrics and dataset distribution endpoint."""
    metrics_file = os.path.join(app.root_path, 'data', 'evaluation_metrics.json')
    dist_file = os.path.join(app.root_path, 'data', 'dataset_distribution.json')
    gk_file = os.path.join(app.root_path, 'gatekeeper_diversity_evaluation.json')

    metrics = {}
    distribution = {}
    gk_eval = []

    if os.path.exists(metrics_file):
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
    if os.path.exists(dist_file):
        with open(dist_file, 'r', encoding='utf-8') as f:
            distribution = json.load(f)
    if os.path.exists(gk_file):
        with open(gk_file, 'r', encoding='utf-8') as f:
            gk_eval = json.load(f)

    return jsonify({
        'status': 'success',
        'metrics': metrics,
        'distribution': distribution,
        'gatekeeper_evaluation': gk_eval,
        'provenance': 'Preliminary Local-Dataset Evaluation (pre-split source pools)',
        'disclosure': 'Metrics reflect authentic empirical evaluation across 34 classes (718 test images) on pre-split local datasets. Not held-out clinical trial results.'
    }), 200


# ══════════════════════════════════════════════════════════════════
# ── 12. History API Endpoints & CSV Export ──
# ══════════════════════════════════════════════════════════════════

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get prediction history with optional search/filter."""
    try:
        db = get_db()
        
        search = request.args.get('search', '').strip()
        animal_filter = request.args.get('animal', '').strip().lower()
        confidence_filter = request.args.get('confidence', '').strip().lower()
        
        query = 'SELECT * FROM prediction_history WHERE status = ?'
        params = ['completed']
        
        if search:
            query += ' AND (prediction LIKE ? OR animal LIKE ? OR timestamp LIKE ?)'
            search_term = f'%{search}%'
            params.extend([search_term, search_term, search_term])
        
        if animal_filter and animal_filter in ('dog', 'cat', 'cow'):
            query += ' AND LOWER(animal) = ?'
            params.append(animal_filter)
        
        query += ' ORDER BY timestamp DESC'
        
        rows = db.execute(query, params).fetchall()
        
        history = []
        for row in rows:
            item = dict(row)
            item['top_predictions'] = json.loads(item['top_predictions'])
            
            # Apply confidence filter in Python for flexibility
            if confidence_filter == 'high' and item['confidence'] < CONFIDENCE_THRESHOLD:
                continue
            if confidence_filter == 'low' and item['confidence'] >= CONFIDENCE_THRESHOLD:
                continue
            
            history.append(item)
        
        return jsonify({'status': 'success', 'history': history, 'total': len(history)})
    except Exception as e:
        print(f"[DB] Error fetching history: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to retrieve prediction history.', 'history': []})

@app.route('/api/history/export', methods=['GET'])
def export_history_csv():
    """Export prediction history records as a downloadable CSV file."""
    try:
        db = get_db()
        rows = db.execute('SELECT * FROM prediction_history WHERE status = ? ORDER BY timestamp DESC', ('completed',)).fetchall()
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Record ID', 'Timestamp', 'Species', 'Predicted Condition', 'Confidence (%)', 'Gatekeeper Status', 'Low Confidence Flag', 'Image Path'])
        
        for r in rows:
            writer.writerow([
                r['id'],
                r['timestamp'],
                r['animal'],
                r['prediction'],
                f"{r['confidence']}%",
                r['gatekeeper_status'],
                'YES' if r['low_confidence'] else 'NO',
                r['image_path']
            ])
            
        csv_data = output.getvalue()
        output.close()
        
        filename = f"vetvision_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return Response(
            csv_data,
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename={filename}'}
        )
    except Exception as e:
        print(f"[DB] Error exporting history CSV: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to generate history CSV export.'}), 500

@app.route('/api/history/<prediction_id>', methods=['GET'])
def get_history_item(prediction_id):
    """Get a single prediction history item by ID."""
    try:
        db = get_db()
        row = db.execute('SELECT * FROM prediction_history WHERE id = ?', (prediction_id,)).fetchone()
        
        if row is None:
            return jsonify({'status': 'error', 'error': 'Prediction not found.'}), 404
        
        item = dict(row)
        item['top_predictions'] = json.loads(item['top_predictions'])
        
        return jsonify({'status': 'success', 'prediction': item})
    except Exception as e:
        print(f"[DB] Error fetching prediction: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to retrieve prediction.'}), 500

@app.route('/api/history/<prediction_id>', methods=['DELETE'])
def delete_history_item(prediction_id):
    """Delete a single prediction history item."""
    try:
        db = get_db()
        result = db.execute('DELETE FROM prediction_history WHERE id = ?', (prediction_id,))
        db.commit()
        
        if result.rowcount == 0:
            return jsonify({'status': 'error', 'error': 'Prediction not found.'}), 404
        
        print(f"[DB] Prediction deleted: {prediction_id}")
        return jsonify({'status': 'success', 'message': 'Prediction deleted from history.'})
    except Exception as e:
        print(f"[DB] Error deleting prediction: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to delete prediction.'}), 500

@app.route('/api/history', methods=['DELETE'])
def clear_history():
    """Delete all prediction history."""
    try:
        db = get_db()
        result = db.execute('DELETE FROM prediction_history')
        db.commit()
        
        deleted_count = result.rowcount
        print(f"[DB] All history cleared ({deleted_count} records)")
        return jsonify({'status': 'success', 'message': f'All prediction history cleared ({deleted_count} records removed).'})
    except Exception as e:
        print(f"[DB] Error clearing history: {e}")
        return jsonify({'status': 'error', 'error': 'Failed to clear history.'}), 500


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)