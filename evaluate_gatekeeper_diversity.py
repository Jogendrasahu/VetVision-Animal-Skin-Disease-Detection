"""
Comprehensive Real-World Diversity & Edge-Case Evaluation for VetVision AI Gatekeeper.
Evaluates 27 diverse real-world images across breeds, poses, backgrounds, lighting,
occlusions, multiple animals, screenshots, humans, vehicles, and unrelated objects.
"""
import os
import json
import app

EVAL_CASES = [
    # ── Category 1: Diverse Domestic Dogs ──
    {
        'id': 'DOG-01',
        'category': 'Canine Breed (Long Coat)',
        'filepath': 'static/uploads/D_F.jpeg',
        'expected_ground_truth': 'Dog',
        'description': 'Shetland Sheepdog with full body, long sable coat, daylight'
    },
    {
        'id': 'DOG-02',
        'category': 'Canine Breed (White / Large)',
        'filepath': 'static/uploads/D_U1.webp',
        'expected_ground_truth': 'Dog',
        'description': 'Kuvasz (large white livestock guardian dog), angled posture'
    },
    {
        'id': 'DOG-03',
        'category': 'Canine Breed (Spitz / Wolf-type)',
        'filepath': 'static/uploads/D_U2.jpg',
        'expected_ground_truth': 'Dog',
        'description': 'Siberian Husky / Wolf-like canine, outdoor terrain'
    },
    {
        'id': 'DOG-04',
        'category': 'Canine Breed (Short Coat)',
        'filepath': 'test_fixtures/car_test.jpg',
        'expected_ground_truth': 'Dog',
        'description': 'PyTorch standard retriever sample dog on lawn'
    },
    # ── Category 2: Domestic Cats ──
    {
        'id': 'CAT-01',
        'category': 'Feline Breed (Tabby / Egyptian)',
        'filepath': 'static/uploads/Cat_Sample.jpg',
        'expected_ground_truth': 'Cat',
        'description': 'Tabby/Egyptian domestic cat, domestic indoor setting'
    },
    # ── Category 3: Bovines (Cows / Cattle) ──
    {
        'id': 'COW-01',
        'category': 'Bovine Breed (Domestic Cattle / Ox)',
        'filepath': 'static/uploads/C_H.jpg',
        'expected_ground_truth': 'Cow',
        'description': 'Domestic cattle / ox standing, natural pasture background'
    },
    {
        'id': 'COW-02',
        'category': 'Bovine Lesion View',
        'filepath': 'static/uploads/C_L.webp',
        'expected_ground_truth': 'Cow',
        'description': 'Cattle dermatological lesion close-up view'
    },
    {
        'id': 'COW-03',
        'category': 'Bovine Standing Profile',
        'filepath': 'static/uploads/C_S.jpg',
        'expected_ground_truth': 'Cow',
        'description': 'Side profile of standing dairy/beef cow'
    },
    {
        'id': 'COW-04',
        'category': 'Bovine Agricultural Sample',
        'filepath': 'static/uploads/C_No1.png',
        'expected_ground_truth': 'Cow',
        'description': 'Agricultural livestock holding view'
    },
    {
        'id': 'COW-05',
        'category': 'Bovine Clinical Pathology',
        'filepath': 'static/uploads/C_UH1.jpg',
        'expected_ground_truth': 'Cow',
        'description': 'Unhealthy bovine with clinical skin pathology'
    },
    # ── Category 4: Unsupported Animals ──
    {
        'id': 'UNSUP-ANIMAL-01',
        'category': 'Unsupported Equine',
        'filepath': 'test_fixtures/horse_test.jpg',
        'expected_ground_truth': 'Unsupported Animal (Horse)',
        'description': 'Horse (equine) outdoors'
    },
    {
        'id': 'UNSUP-ANIMAL-02',
        'category': 'Unsupported Avian',
        'filepath': 'test_fixtures/bird_test.jpg',
        'expected_ground_truth': 'Unsupported Animal (Bird)',
        'description': 'Bird perched on branch'
    },
    {
        'id': 'UNSUP-ANIMAL-03',
        'category': 'Unsupported Insect',
        'filepath': 'test_fixtures/butterfly_test.jpg',
        'expected_ground_truth': 'Unsupported Organism (Butterfly)',
        'description': 'Butterfly on leaf'
    },
    # ── Category 5: Humans & Portraits ──
    {
        'id': 'HUMAN-01',
        'category': 'Human Portrait (Uniform)',
        'filepath': 'test_fixtures/grace_hopper_test.jpg',
        'expected_ground_truth': 'Unsupported (Human)',
        'description': 'Admiral Grace Hopper portrait in military uniform'
    },
    {
        'id': 'HUMAN-02',
        'category': 'Human Face (Standard Benchmark)',
        'filepath': 'test_fixtures/cat_sample2_test.jpg',
        'expected_ground_truth': 'Unsupported (Human)',
        'description': 'Standard Lena human portrait benchmark'
    },
    # ── Category 6: Vehicles & Transportation ──
    {
        'id': 'VEHICLE-01',
        'category': 'Vehicle / Automobile',
        'filepath': 'test_fixtures/smart_car_test.jpg',
        'expected_ground_truth': 'Unsupported (Vehicle)',
        'description': 'Automobile on street'
    },
    # ── Category 7: Architecture, Food & Plants ──
    {
        'id': 'SCENE-01',
        'category': 'Architecture / Structure',
        'filepath': 'test_fixtures/china_architecture.jpg',
        'expected_ground_truth': 'Unsupported (Architecture)',
        'description': 'Chinese pagoda architectural landscape'
    },
    {
        'id': 'OBJECT-01',
        'category': 'Food / Fruit Basket',
        'filepath': 'test_fixtures/fruits_test.jpg',
        'expected_ground_truth': 'Unsupported (Food)',
        'description': 'Still life fruit basket'
    },
    {
        'id': 'PLANT-01',
        'category': 'Botany / Flowers',
        'filepath': 'test_fixtures/flower_plant.jpg',
        'expected_ground_truth': 'Unsupported (Plant)',
        'description': 'Flowering plant foliage'
    },
    # ── Category 8: Screenshots & Digital Artifacts ──
    {
        'id': 'SCREEN-01',
        'category': 'Browser Desktop Screenshot',
        'filepath': 'static/uploads/Screenshot_2026-09-27_221918.png',
        'expected_ground_truth': 'Unsupported (Screenshot)',
        'description': 'Web browser desktop interface screenshot'
    },
    {
        'id': 'SCREEN-02',
        'category': 'Software Application Screenshot',
        'filepath': 'static/uploads/Screenshot_2026-11-03_103606.png',
        'expected_ground_truth': 'Unsupported (Screenshot)',
        'description': 'Desktop software interface'
    },
    {
        'id': 'OBJECT-02',
        'category': 'Metallic Hardware',
        'filepath': 'static/uploads/blank.jpg',
        'expected_ground_truth': 'Unsupported (Hardware)',
        'description': 'Metallic screw / fastener hardware'
    },
    # ── Category 9: Challenging Edge Cases ──
    {
        'id': 'EDGE-01',
        'category': 'Multiple Animals (Composite)',
        'filepath': 'test_fixtures/both_dog_and_cat.jpg',
        'expected_ground_truth': 'Multiple Animals (Dog + Cat)',
        'description': 'Composite image containing both Dog and Cat side by side'
    },
    {
        'id': 'EDGE-02',
        'category': 'Partial Occlusion',
        'filepath': 'test_fixtures/dog_occluded.jpg',
        'expected_ground_truth': 'Dog (50% Occluded)',
        'description': 'Dog image with 50% obstruction by barrier'
    },
    {
        'id': 'EDGE-03',
        'category': 'Low Lighting / Underexposed',
        'filepath': 'test_fixtures/cat_dark_lighting.jpg',
        'expected_ground_truth': 'Cat (75% Dark)',
        'description': 'Cat with 75% reduced illumination (severe underexposure)'
    },
    {
        'id': 'EDGE-04',
        'category': 'Extreme Close-Up (No Facial Anatomy)',
        'filepath': 'test_fixtures/dog_fur_extreme_closeup.jpg',
        'expected_ground_truth': 'Dog Fur Texture Only',
        'description': 'Isolated close-up patch of dog fur with no facial landmarks'
    },
    {
        'id': 'EDGE-05',
        'category': 'Severe Motion Blur',
        'filepath': 'test_fixtures/blurry_dog.jpg',
        'expected_ground_truth': 'Dog (Severe Blur)',
        'description': 'Dog image subject to radius=18 Gaussian blur'
    },
    {
        'id': 'EDGE-06',
        'category': 'Ambiguous Bovine Angle',
        'filepath': 'static/uploads/C_UH.jpg',
        'expected_ground_truth': 'Cow (Difficult Angle)',
        'description': 'Cattle lesion with atypical angle resembling camel profile'
    }
]

results = []

print("=" * 125)
print(f"{'ID':<15} | {'CATEGORY':<28} | {'GROUND TRUTH':<24} | {'GATEKEEPER STATUS':<20} | {'PREDICTED':<10} | {'SCORE':<7} | {'TOP LABEL'}")
print("=" * 125)

for case in EVAL_CASES:
    p = case['filepath']
    if not os.path.exists(p):
        print(f"Skipping missing file: {p}")
        continue
    
    # Run Gatekeeper without selected animal (unbiased species discovery)
    res = app.identify_species(p, selected_animal=None)
    status = res['status']
    det = res['detected_species']
    score = f"{res['softmax_score']:.1f}%"
    top_lbl = res.get('top_label', 'N/A')
    cat = res.get('top_category', 'N/A')
    
    # Classify outcome
    gt = case['expected_ground_truth']
    if 'Dog' in gt and 'Occluded' not in gt and 'Texture' not in gt and 'Blur' not in gt:
        outcome = 'CORRECT_ACCEPT' if (status == 'species_match' and det == 'Dog') else ('UNCERTAIN' if status == 'species_uncertain' else 'ERROR')
    elif 'Cat' in gt and 'Dark' not in gt:
        outcome = 'CORRECT_ACCEPT' if (status == 'species_match' and det == 'Cat') else ('UNCERTAIN' if status == 'species_uncertain' else 'ERROR')
    elif 'Cow' in gt and 'Difficult' not in gt:
        outcome = 'CORRECT_ACCEPT' if (status == 'species_match' and det == 'Cow') else ('UNCERTAIN' if status == 'species_uncertain' else 'ERROR')
    elif 'Unsupported' in gt:
        outcome = 'CORRECT_REJECT' if status == 'unsupported_object' else ('UNCERTAIN' if status == 'species_uncertain' else 'FALSE_ACCEPTANCE')
    else:
        # Edge cases
        if status in ('species_uncertain', 'unsupported_object'):
            outcome = 'UNCERTAIN_OR_REJECTED (SAFE)'
        else:
            outcome = f'ACCEPTED_AS_{det}'
    
    results.append({
        'id': case['id'],
        'category': case['category'],
        'ground_truth': gt,
        'filepath': p,
        'status': status,
        'detected_species': det,
        'softmax_score': res['softmax_score'],
        'top_label': top_lbl,
        'top_category': cat,
        'decision': res['decision'],
        'outcome': outcome
    })
    
    print(f"{case['id']:<15} | {case['category'][:26]:<28} | {gt[:22]:<24} | {status:<20} | {det:<10} | {score:<7} | {top_lbl} ({cat})")

print("=" * 125)

# Save evaluation output
with open("gatekeeper_diversity_evaluation.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nEvaluation successfully written to gatekeeper_diversity_evaluation.json")

