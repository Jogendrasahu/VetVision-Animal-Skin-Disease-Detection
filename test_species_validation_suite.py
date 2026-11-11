"""
Automated 14-Scenario Species Validation Test Suite for VetVision AI.
"""
import requests
import json
import os
import sys

BASE_URL = "http://127.0.0.1:5000"
FIXTURE_DIR = "test_fixtures"

test_results = []

def record(test_num, description, detected_species, inference_allowed, user_facing_result, status):
    test_results.append({
        'test_num': test_num,
        'description': description,
        'detected_species': detected_species,
        'inference_allowed': "YES" if inference_allowed else "NO (BLOCKED)",
        'user_facing_result': user_facing_result,
        'status': status
    })
    status_icon = "[PASS]" if status == "PASS" else "[FAIL]"
    print(f"\n{status_icon} Test #{test_num}: {description}")
    print(f"   Detected Species: {detected_species}")
    print(f"   Inference Allowed: {'YES' if inference_allowed else 'NO (BLOCKED)'}")
    print(f"   User-Facing Result: {user_facing_result[:110]}...")

# ── Test 1: Dog image selected as Dog ──
with open(os.path.join(FIXTURE_DIR, "dog_sample.jpg"), "rb") as f:
    r = requests.post(f"{BASE_URL}/predict", data={'animal_type': 'dog'}, files={'image': f})
d = r.json()
is_pass = (r.status_code == 200 and d.get('status') == 'success' and 'disease' in d)
record(1, "Dog image selected as Dog", d.get('gatekeeper', {}).get('verified_species', 'Dog'), is_pass, f"Success: {d.get('disease')} ({d.get('confidence')}%)", "PASS" if is_pass else "FAIL")

# ── Test 2: Cat image selected as Dog ──
with open(os.path.join(FIXTURE_DIR, "cat_sample.jpg"), "rb") as f:
    r = requests.post(f"{BASE_URL}/predict", data={'animal_type': 'dog'}, files={'image': f})
d = r.json()
is_pass = (r.status_code == 200 and d.get('status') == 'species_mismatch' and d.get('detected_species') == 'Cat')
record(2, "Cat image selected as Dog", d.get('detected_species'), not d.get('gatekeeper_error', False), d.get('message', ''), "PASS" if is_pass else "FAIL")

# ── Test 3: Cow image selected as Dog ──
with open(os.path.join(FIXTURE_DIR, "cow_sample.jpg"), "rb") as f:
    r = requests.post(f"{BASE_URL}/predict", data={'animal_type': 'dog'}, files={'image': f})
d = r.json()
is_pass = (r.status_code == 200 and d.get('status') == 'species_mismatch' and d.get('detected_species') == 'Cow')
record(3, "Cow image selected as Dog", d.get('detected_species'), not d.get('gatekeeper_error', False), d.get('message', ''), "PASS" if is_pass else "FAIL")

# ── Test 4: Unsupported human/object/screen image ──
with open(os.path.join(FIXTURE_DIR, "screen_sample.png"), "rb") as f:
    r = requests.post(f"{BASE_URL}/predict", data={'animal_type': 'dog'}, files={'image': f})
d = r.json()
is_pass = (r.status_code == 200 and d.get('status') == 'unsupported_object' and "No supported animal detected" in d.get('message', ''))
record(4, "Unsupported screen/object image", d.get('detected_species'), not d.get('gatekeeper_error', False), d.get('message', ''), "PASS" if is_pass else "FAIL")

# ── Test 5: Unclear or low-quality animal image ──
with open(os.path.join(FIXTURE_DIR, "blurry_dog.jpg"), "rb") as f:
    r = requests.post(f"{BASE_URL}/predict", data={'animal_type': 'dog'}, files={'image': f})
d = r.json()
# Heavily blurred image should trigger species_uncertain OR quality warnings OR low confidence
is_pass = (d.get('status') in ('species_uncertain', 'unsupported_object') or d.get('low_confidence') == True or len(d.get('warnings', [])) > 0)
record(5, "Unclear or low-quality animal image", d.get('detected_species', 'Unknown'), d.get('status') == 'success', d.get('message') or f"Quality Warning: {d.get('warnings')}", "PASS" if is_pass else "FAIL")

# ── Test 6: Multiple images containing different animal species ──
with open(os.path.join(FIXTURE_DIR, "dog_sample.jpg"), "rb") as f1, open(os.path.join(FIXTURE_DIR, "cat_sample.jpg"), "rb") as f2:
    r = requests.post(f"{BASE_URL}/api/predict/multi", data={'animal_type': 'dog'}, files=[('files', f1), ('files', f2)])
d = r.json()
# Cat should be excluded from consensus, only Dog analyzed, notice present
is_pass = (r.status_code == 200 and d.get('has_mixed_species') == True and d.get('images_analyzed') == 1 and d.get('status') == 'success')
record(6, "Multiple images containing different species (Dog + Cat)", "Dog (verified) & Cat (excluded)", True, d.get('mixed_species_notice', ''), "PASS" if is_pass else "FAIL")

# ── Test 7: Multiple images containing only one species ──
with open(os.path.join(FIXTURE_DIR, "dog_sample.jpg"), "rb") as f1, open(os.path.join(FIXTURE_DIR, "dog_sample.jpg"), "rb") as f2:
    r = requests.post(f"{BASE_URL}/api/predict/multi", data={'animal_type': 'dog'}, files=[('files', f1), ('files', f2)])
d = r.json()
is_pass = (r.status_code == 200 and d.get('status') == 'success' and d.get('images_analyzed') == 2 and not d.get('has_mixed_species', False))
record(7, "Multiple images containing single species (2 Dogs)", "Dog (both verified)", True, f"Consensus: {d.get('disease')} ({d.get('confidence')}%, agreement: {d.get('consensus_agreement')}%)", "PASS" if is_pass else "FAIL")

# ── Test 8: Valid Dog video ──
with open(os.path.join(FIXTURE_DIR, "valid_dog_video.mp4"), "rb") as f:
    r = requests.post(f"{BASE_URL}/api/predict/video", data={'animal_type': 'dog'}, files={'video': f})
d = r.json()
is_pass = (r.status_code == 200 and d.get('status') == 'success' and d.get('frames_verified', 0) >= 2)
record(8, "Valid Dog video", "Dog", is_pass, f"Video consensus: {d.get('disease')} ({d.get('confidence')}%, {d.get('frames_verified')} frames)", "PASS" if is_pass else "FAIL")

# ── Test 9: Valid Cat video ──
with open(os.path.join(FIXTURE_DIR, "valid_cat_video.mp4"), "rb") as f:
    r = requests.post(f"{BASE_URL}/api/predict/video", data={'animal_type': 'cat'}, files={'video': f})
d = r.json()
is_pass = (r.status_code == 200 and d.get('status') == 'success' and d.get('frames_verified', 0) >= 2)
record(9, "Valid Cat video", "Cat", is_pass, f"Video consensus: {d.get('disease')} ({d.get('confidence')}%, {d.get('frames_verified')} frames)", "PASS" if is_pass else "FAIL")

# ── Test 10: Valid Cow video ──
with open(os.path.join(FIXTURE_DIR, "valid_cow_video.mp4"), "rb") as f:
    r = requests.post(f"{BASE_URL}/api/predict/video", data={'animal_type': 'cow'}, files={'video': f})
d = r.json()
is_pass = (r.status_code == 200 and d.get('status') == 'success' and d.get('frames_verified', 0) >= 2)
record(10, "Valid Cow video", "Cow", is_pass, f"Video consensus: {d.get('disease')} ({d.get('confidence')}%, {d.get('frames_verified')} frames)", "PASS" if is_pass else "FAIL")

# ── Test 11: Screen-recording video with no animal ──
screen_vid_path = "static/uploads/vid_0720a75e61_Screen_Recording_2026-11-03_151156.mp4"
with open(screen_vid_path, "rb") as f:
    r = requests.post(f"{BASE_URL}/api/predict/video", data={'animal_type': 'dog'}, files={'video': f})
d = r.json()
is_pass = (r.status_code == 200 and d.get('status') == 'unsupported_object' and "No supported animal detected" in d.get('message', ''))
record(11, "Screen-recording video with no animal", d.get('detected_species'), not d.get('gatekeeper_error', False), d.get('message', ''), "PASS" if is_pass else "FAIL")

# ── Test 12: Corrupted or empty video ──
with open(os.path.join(FIXTURE_DIR, "corrupt_video.mp4"), "rb") as f:
    r = requests.post(f"{BASE_URL}/api/predict/video", data={'animal_type': 'dog'}, files={'video': f})
d = r.json()
is_pass = (r.status_code == 400 and ('Failed to decode' in d.get('error', '') or 'insufficient' in d.get('error', '')))
record(12, "Corrupted or empty video", "None (Decode Failure)", False, d.get('error', ''), "PASS" if is_pass else "FAIL")

# ── Test 13: Video with multiple species (Dog + Cat) ──
with open(os.path.join(FIXTURE_DIR, "multi_species_video.mp4"), "rb") as f:
    r = requests.post(f"{BASE_URL}/api/predict/video", data={'animal_type': 'dog'}, files={'video': f})
d = r.json()
is_pass = (r.status_code == 200 and d.get('status') == 'mixed_species' and d.get('gatekeeper_error') == True)
record(13, "Video with multiple species (Dog + Cat)", f"{d.get('detected_species')}", not d.get('gatekeeper_error', False), d.get('message', ''), "PASS" if is_pass else "FAIL")

# ── Test 14: Video with insufficient usable frames (Blurry) ──
with open(os.path.join(FIXTURE_DIR, "blurry_video.mp4"), "rb") as f:
    r = requests.post(f"{BASE_URL}/api/predict/video", data={'animal_type': 'dog'}, files={'video': f})
d = r.json()
is_pass = (r.status_code == 200 and d.get('status') == 'species_uncertain' and d.get('gatekeeper_error') == True and ('motion blur' in d.get('message', '') or 'insufficient' in d.get('message', '')))
record(14, "Video with excessive motion blur / insufficient usable frames", d.get('detected_species'), not d.get('gatekeeper_error', False), d.get('message', ''), "PASS" if is_pass else "FAIL")

print("\n" + "=" * 80)
total = len(test_results)
passed = sum(1 for t in test_results if t['status'] == 'PASS')
print(f"RESULTS: {passed}/{total} Tests Passed ({passed/total*100:.1f}%)")
print("=" * 80)

# Save test results JSON
with open("test_species_validation_results.json", "w") as f:
    json.dump(test_results, f, indent=2)

if passed < total:
    sys.exit(1)

