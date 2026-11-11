"""
test_cross_species_matrix.py
Comprehensive testing of all 9 species matching and mismatching permutations:
1. Dog image + Dog selected -> MATCH -> Disease Prediction Returned
2. Cat image + Cat selected -> MATCH -> Disease Prediction Returned
3. Cow image + Cow selected -> MATCH -> Disease Prediction Returned
4. Dog image + Cat selected -> MISMATCH -> Error returned, NO disease prediction
5. Dog image + Cow selected -> MISMATCH -> Error returned, NO disease prediction
6. Cat image + Dog selected -> MISMATCH -> Error returned, NO disease prediction
7. Cat image + Cow selected -> MISMATCH -> Error returned, NO disease prediction
8. Cow image + Dog selected -> MISMATCH -> Error returned, NO disease prediction
9. Cow image + Cat selected -> MISMATCH -> Error returned, NO disease prediction
"""

import os
import sys
import io
import json

PROJECT_DIR = r"c:\Users\Jogen\OneDrive\Desktop\Jogendra\Projects\Animal_Disease_App"
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from app import app

def run_cross_species_matrix():
    print("=" * 70)
    print("ANIVISION AI — COMPREHENSIVE CROSS-SPECIES MATRIX VERIFICATION")
    print("=" * 70)

    client = app.test_client()

    dog_img_path = os.path.join(PROJECT_DIR, 'static', 'uploads', 'D_F.jpeg')
    cat_img_path = os.path.join(PROJECT_DIR, 'static', 'uploads', 'Cat_Sample.jpg')
    cow_img_path = os.path.join(PROJECT_DIR, 'static', 'uploads', 'C_H.jpg')

    assert os.path.exists(dog_img_path), f"Missing Dog sample image: {dog_img_path}"
    assert os.path.exists(cat_img_path), f"Missing Cat sample image: {cat_img_path}"
    assert os.path.exists(cow_img_path), f"Missing Cow sample image: {cow_img_path}"

    test_matrix = [
        # (Image Name, Image Path, Selected Animal, Expected Match, Expected Status)
        ("Dog Image", dog_img_path, "dog", True, "success"),
        ("Cat Image", cat_img_path, "cat", True, "success"),
        ("Cow Image", cow_img_path, "cow", True, "success"),
        
        ("Dog Image", dog_img_path, "cat", False, "species_mismatch"),
        ("Dog Image", dog_img_path, "cow", False, "species_mismatch"),
        
        ("Cat Image", cat_img_path, "dog", False, "species_mismatch"),
        ("Cat Image", cat_img_path, "cow", False, "species_mismatch"),
        
        ("Cow Image", cow_img_path, "dog", False, "species_mismatch"),
        ("Cow Image", cow_img_path, "cat", False, "species_mismatch"),
    ]

    all_passed = True
    results_summary = []

    for idx, (img_desc, img_path, selected_animal, expect_match, expected_status) in enumerate(test_matrix, 1):
        print(f"\n[Test {idx}/9] {img_desc} + Selected Animal: '{selected_animal.upper()}'")
        
        with open(img_path, 'rb') as f:
            file_bytes = f.read()

        data = {
            'file': (io.BytesIO(file_bytes), os.path.basename(img_path)),
            'animal_type': selected_animal
        }

        res = client.post('/predict', data=data, content_type='multipart/form-data')
        assert res.status_code == 200, f"Expected HTTP 200, got {res.status_code}"
        res_json = res.get_json()

        if expect_match:
            # Must return disease prediction
            is_success = (res_json.get('status') == 'success' and 
                          'disease' in res_json and 
                          'confidence' in res_json and 
                          'top3' in res_json and 
                          not res_json.get('gatekeeper_error'))
            if is_success:
                print(f"  -> SUCCESS (MATCH): Predicted '{res_json['disease']}' with {res_json['confidence']}% confidence.")
                print(f"  -> Top-3 Differential: {[item['disease'] for item in res_json['top3']]}")
                results_summary.append((idx, f"{img_desc} + {selected_animal}", "PASS", f"Matched -> {res_json['disease']} ({res_json['confidence']}%)"))
            else:
                print(f"  -> FAILED: Expected success match, got {res_json}")
                all_passed = False
                results_summary.append((idx, f"{img_desc} + {selected_animal}", "FAIL", str(res_json)))
        else:
            # Must detect mismatch and NEVER return disease prediction
            is_mismatch = (res_json.get('gatekeeper_error') is True and 
                           res_json.get('status') in ['species_mismatch', 'species_uncertain'] and 
                           'disease' not in res_json and 
                           'top3' not in res_json)
            if is_mismatch:
                print(f"  -> SUCCESS (MISMATCH STOPPED): Gatekeeper blocked model execution.")
                print(f"  -> Status: {res_json.get('status')}")
                print(f"  -> Message: {res_json.get('message')}")
                print(f"  -> Detected Species: {res_json.get('detected_species')}")
                results_summary.append((idx, f"{img_desc} + {selected_animal}", "PASS", f"Gatekeeper Intercepted ({res_json.get('status')})"))
            else:
                print(f"  -> FAILED: Expected mismatch rejection, but got: {res_json}")
                all_passed = False
                results_summary.append((idx, f"{img_desc} + {selected_animal}", "FAIL", "Forced prediction on mismatch!"))

    print("\n" + "=" * 70)
    print("CROSS-SPECIES MATRIX SUMMARY:")
    print("=" * 70)
    for idx, name, status, detail in results_summary:
        print(f"  [{idx}] {name:30} : {status:4} | {detail}")

    assert all_passed, "Some cross-species matrix tests failed!"
    print("\n>>> ALL 9 CROSS-SPECIES SAFETY PERMUTATIONS PASSED FLAWLESSLY! <<<")
    print("=" * 70)

if __name__ == '__main__':
    run_cross_species_matrix()
