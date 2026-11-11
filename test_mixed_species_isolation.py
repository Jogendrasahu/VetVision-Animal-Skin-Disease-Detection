"""
Test suite verifying species isolation and mixed-species handling in VetVision AI.
"""
import io
import os
import json
from app import app

client = app.test_client()

def test_mixed_species_isolation():
    print("=" * 65)
    print("TESTING MIXED-SPECIES MULTI-UPLOAD ISOLATION & ROUTING")
    print("=" * 65)

    dog_file_path = 'static/uploads/D_F.jpeg'
    cat_file_path = 'static/uploads/Cat_Sample.jpg'
    cow_file_path = 'static/uploads/C_H.jpg'

    assert os.path.exists(dog_file_path), "Dog test file missing"
    assert os.path.exists(cat_file_path), "Cat test file missing"
    assert os.path.exists(cow_file_path), "Cow test file missing"

    # --- Case 1: Same species multi-upload (2 dogs) with auto-routing ---
    print("\n[Case 1] Uploading 2 Dogs with animal_type='auto'...")
    with open(dog_file_path, 'rb') as f1, open(dog_file_path, 'rb') as f2:
        data = {
            'animal_type': 'auto',
            'files': [
                (io.BytesIO(f1.read()), 'dog1.jpeg'),
                (io.BytesIO(f2.read()), 'dog2.jpeg')
            ]
        }
        res = client.post('/api/predict/multi', data=data, content_type='multipart/form-data')
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    res_json = res.get_json()
    assert res_json['status'] == 'success'
    assert res_json['animal'] == 'Dog'
    assert res_json['images_analyzed'] == 2
    assert res_json['total_images_submitted'] == 2
    assert len(res_json['top3']) == 3
    print("  -> PASS: 2 dogs successfully identified, routed to Dog model, consensus calculated.")

    # --- Case 2: Mixed species (Dog + Cat) with animal_type='auto' ---
    print("\n[Case 2] Uploading mixed species (1 Dog + 1 Cat) with animal_type='auto'...")
    with open(dog_file_path, 'rb') as f1, open(cat_file_path, 'rb') as f2:
        data = {
            'animal_type': 'auto',
            'files': [
                (io.BytesIO(f1.read()), 'dog1.jpeg'),
                (io.BytesIO(f2.read()), 'cat1.jpg')
            ]
        }
        res = client.post('/api/predict/multi', data=data, content_type='multipart/form-data')
    assert res.status_code == 200
    res_json = res.get_json()
    assert res_json['status'] == 'mixed_species_isolated'
    assert res_json['has_cross_species_consensus'] is False
    breakdown = res_json['image_breakdown']
    assert len(breakdown) == 2
    assert breakdown[0]['detected_species'] == 'Dog' and breakdown[0]['target_species'] == 'Dog'
    assert breakdown[1]['detected_species'] == 'Cat' and breakdown[1]['target_species'] == 'Cat'
    print(f"  -> PASS: Mixed species recognized. No cross-species averaging! Breakdown: {breakdown[0]['detected_species']} & {breakdown[1]['detected_species']}.")

    # --- Case 3: Mixed species (Dog + Cat) with user-selected animal_type='dog' ---
    print("\n[Case 3] Uploading mixed species (1 Dog + 1 Cat) with animal_type='dog'...")
    with open(dog_file_path, 'rb') as f1, open(cat_file_path, 'rb') as f2:
        data = {
            'animal_type': 'dog',
            'files': [
                (io.BytesIO(f1.read()), 'dog1.jpeg'),
                (io.BytesIO(f2.read()), 'cat1.jpg')
            ]
        }
        res = client.post('/api/predict/multi', data=data, content_type='multipart/form-data')
    assert res.status_code == 200
    res_json = res.get_json()
    assert res_json['status'] == 'success'
    assert res_json['animal'] == 'Dog'
    assert res_json['images_analyzed'] == 1
    assert 'excluded' in res_json.get('mixed_species_notice', '')
    print(f"  -> PASS: Cat image strictly excluded from Dog model. Notice: {res_json['mixed_species_notice']}")

    # --- Case 4: Complete Mismatch (2 Cats uploaded when user selected animal_type='dog') ---
    print("\n[Case 4] Uploading 2 Cats when user selected animal_type='dog'...")
    with open(cat_file_path, 'rb') as f1, open(cat_file_path, 'rb') as f2:
        data = {
            'animal_type': 'dog',
            'files': [
                (io.BytesIO(f1.read()), 'cat1.jpg'),
                (io.BytesIO(f2.read()), 'cat2.jpg')
            ]
        }
        res = client.post('/api/predict/multi', data=data, content_type='multipart/form-data')
    assert res.status_code == 200
    res_json = res.get_json()
    assert res_json['status'] == 'species_mismatch'
    assert res_json['gatekeeper_error'] is True
    assert res_json['images_analyzed'] == 0
    print(f"  -> PASS: Zero cats routed to Dog model. Gatekeeper rejected: {res_json['message']}")

    # --- Case 5: Single-image auto-routing ---
    print("\n[Case 5] Testing single-image auto-routing with Dog...")
    with open(dog_file_path, 'rb') as f:
        data = {'animal_type': 'auto', 'file': (io.BytesIO(f.read()), 'dog_single.jpeg')}
        res = client.post('/predict', data=data, content_type='multipart/form-data')
    assert res.status_code == 200
    res_json = res.get_json()
    assert res_json['status'] == 'success'
    assert res_json['animal'] == 'Dog'
    assert res_json['auto_routed'] is True
    print(f"  -> PASS: Single-image auto-routed to {res_json['animal']} model ({res_json['disease']}).")

    print("\n" + "=" * 65)
    print("ALL MIXED-SPECIES ISOLATION TESTS PASSED WITH 100% COMPLIANCE!")
    print("=" * 65)

if __name__ == '__main__':
    test_mixed_species_isolation()
