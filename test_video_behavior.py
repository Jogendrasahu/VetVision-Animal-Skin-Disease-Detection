"""
Comprehensive video screening evaluation script.
Tests:
1. Valid Dog video -> Keyframe extraction, Gatekeeper verification, temporal consensus.
2. Multi-species video (Dog + Cat) -> Multiple species detected, mixed-species rejection.
3. Excessive motion blur video -> Laplacian thresholding flags blurry frames.
4. Species mismatch video (Dog video submitted to Cat model) -> Rejected as species mismatch.
"""
import io
import os
import json
from app import app

client = app.test_client()

def test_video_screening():
    print("=" * 65)
    print("TESTING VIDEO SCREENING BEHAVIOR & SAFETY CAVEATS")
    print("=" * 65)

    dog_vid = 'test_fixtures/valid_dog_video.mp4'
    cat_vid = 'test_fixtures/valid_cat_video.mp4'
    multi_vid = 'test_fixtures/multi_species_video.mp4'
    blurry_vid = 'test_fixtures/blurry_video.mp4'

    assert os.path.exists(dog_vid), "Dog video missing"
    assert os.path.exists(cat_vid), "Cat video missing"
    assert os.path.exists(multi_vid), "Multi-species video missing"
    assert os.path.exists(blurry_vid), "Blurry video missing"

    # Test 1: Valid Dog video with dog model
    print("\n[Test 1] Testing Valid Dog Video with Dog Model...")
    with open(dog_vid, 'rb') as f:
        data = {'animal_type': 'dog', 'file': (io.BytesIO(f.read()), 'dog_video.mp4')}
        res = client.post('/api/predict/video', data=data, content_type='multipart/form-data')
    assert res.status_code == 200
    res_json = res.get_json()
    assert res_json['status'] == 'success'
    assert res_json['animal'] == 'Dog'
    assert 'experimental screening tool' in res_json['clinical_disclaimer']
    print(f"  -> PASS: Valid dog video processed. Verified frames: {res_json['frames_verified']}/{res_json['frames_sampled']}.")
    print(f"     Consensus: {res_json['disease']} ({res_json['confidence']}%), Agreement: {res_json['frame_agreement_pct']}%.")
    print(f"     Caveat: '{res_json['clinical_disclaimer'][:60]}...'")

    # Test 2: Multi-species video (frames transition from dog to cat)
    print("\n[Test 2] Testing Multi-Species Video (Dog + Cat in same footage)...")
    with open(multi_vid, 'rb') as f:
        data = {'animal_type': 'dog', 'file': (io.BytesIO(f.read()), 'multi_species.mp4')}
        res = client.post('/api/predict/video', data=data, content_type='multipart/form-data')
    assert res.status_code == 200
    res_json = res.get_json()
    assert res_json['status'] == 'mixed_species'
    assert res_json['gatekeeper_error'] is True
    print(f"  -> PASS: Multi-species conflict caught across frames. Gatekeeper rejected: {res_json['message']}")

    # Test 3: Species mismatch in video (Dog video submitted with animal_type='cat')
    print("\n[Test 3] Testing Species Mismatch (Dog video with Cat model)...")
    with open(dog_vid, 'rb') as f:
        data = {'animal_type': 'cat', 'file': (io.BytesIO(f.read()), 'dog_as_cat.mp4')}
        res = client.post('/api/predict/video', data=data, content_type='multipart/form-data')
    assert res.status_code == 200
    res_json = res.get_json()
    assert res_json['status'] == 'species_mismatch'
    assert res_json['gatekeeper_error'] is True
    print(f"  -> PASS: Video species mismatch rejected: {res_json['message']}")

    # Test 4: Blurry video (severe motion blur)
    print("\n[Test 4] Testing Blurry Video (Excessive camera shake)...")
    with open(blurry_vid, 'rb') as f:
        data = {'animal_type': 'dog', 'file': (io.BytesIO(f.read()), 'blurry.mp4')}
        res = client.post('/api/predict/video', data=data, content_type='multipart/form-data')
    assert res.status_code == 200
    res_json = res.get_json()
    assert res_json['gatekeeper_error'] is True
    print(f"  -> PASS: Low stability / blurry video handled safely. Status: {res_json['status']} ({res_json['message']})")

    print("\n" + "=" * 65)
    print("ALL VIDEO SCREENING SAFETY TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 65)

if __name__ == '__main__':
    test_video_screening()
