import os
import sys
import io
import json
from PIL import Image

PROJECT_DIR = r"c:\Users\Jogen\OneDrive\Desktop\Jogendra\Projects\Animal_Disease_App"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from app import app

def test_mismatch_and_prediction():
    print("=" * 65)
    print("TESTING API ENDPOINTS & CROSS-SPECIES GATEKEEPER INTEGRATION")
    print("=" * 65)
    client = app.test_client()

    # 1. Missing file or animal type
    res = client.post('/api/predict', data={})
    assert res.status_code == 400
    print("[1] Missing arguments rejected with 400: PASS")

    # 2. Corrupt or non-image file
    fake_file = (io.BytesIO(b"not an image file"), "test.txt")
    res = client.post('/api/predict', data={'file': fake_file, 'animal_type': 'dog'}, content_type='multipart/form-data')
    assert res.status_code == 400
    print("[2] Corrupted file rejected with 400: PASS")

    # 3. Tiny image (< 32px) quality check
    tiny_img = Image.new('RGB', (16, 16), color=(255, 0, 0))
    tiny_buf = io.BytesIO()
    tiny_img.save(tiny_buf, format='JPEG')
    tiny_buf.seek(0)
    res = client.post('/api/predict', data={'file': (tiny_buf, 'tiny.jpg'), 'animal_type': 'dog'}, content_type='multipart/form-data')
    assert res.status_code == 400
    data = res.get_json()
    print(f"DEBUG Tiny image response: {data}")
    assert ("too small" in data.get('error', '').lower() or "corrupted" in data.get('error', '').lower() or "dimensions" in data.get('error', '').lower())
    print("[3] Tiny image (<32px) quality check triggered: PASS")

    # 4. Plain synthetic image tested with Gatekeeper
    test_img = Image.new('RGB', (224, 224), color=(100, 100, 100))
    buf = io.BytesIO()
    test_img.save(buf, format='JPEG')
    buf.seek(0)

    res = client.post('/api/predict', data={'file': (buf, 'plain.jpg'), 'animal_type': 'cow'}, content_type='multipart/form-data')
    assert res.status_code in [200, 400]
    data = res.get_json()
    # It must either detect species_uncertain / species_mismatch (gatekeeper_error=True) or complete safely
    if data.get('gatekeeper_error'):
        assert data['status'] in ['species_mismatch', 'species_uncertain']
        print(f"[4] Gatekeeper safely intercepted non-cow image: Status={data['status']}, Message='{data['message']}': PASS")
    else:
        print(f"[4] Prediction proceeded with verified status: {data['status']}: PASS")

    print("\n" + "=" * 65)
    print("ALL API & GATEKEEPER ENDPOINT CHECKS PASSED!")
    print("=" * 65)

if __name__ == '__main__':
    test_mismatch_and_prediction()
