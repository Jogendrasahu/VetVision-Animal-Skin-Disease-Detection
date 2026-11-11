import requests

def test_dataset_dashboard():
    r = requests.get('http://127.0.0.1:5000/dataset')
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    html = r.text
    print(f"Dataset page loaded: {len(html)} bytes")

    checks = [
        "Preliminary Local-Dataset Evaluation Disclosure",
        "Dog Classifier Accuracy",
        "Cat Classifier Accuracy",
        "Cow Classifier Accuracy",
        "81.88%",
        "80.89%",
        "97.14%",
        "0.9714",
        "Dataset Statistics",
        "15,694 Total Images",
        "Neural Architecture",
        "Training Accuracy",
        "Training history source: Not available from current verified evaluation artifacts",
        "Multi-Model Performance Comparison",
        "Confusion Matrices",
        "Diagonal values represent correctly classified samples. Off-diagonal values represent classification errors.",
        "Per-Class Precision, Recall",
        "Pathology Coverage Matrix (34 Total Classes)",
        "Detailed Diagnostic Breakdown",
        "Clinical Limitations",
        "chart.umd.min.js"
    ]

    for item in checks:
        assert item in html, f"Missing required text: {item}"
        print(f"  [PASS] Found: '{item[:50]}'")

    # Check API endpoint
    api_r = requests.get('http://127.0.0.1:5000/api/evaluation/metrics')
    assert api_r.status_code == 200, f"API returned {api_r.status_code}"
    data = api_r.json()
    assert 'metrics' in data
    assert 'cow' in data['metrics']
    assert 'dog' in data['metrics']
    assert 'cat' in data['metrics']
    assert round(data['metrics']['cow']['accuracy'] * 100, 2) == 97.14
    assert round(data['metrics']['dog']['accuracy'] * 100, 2) == 81.88
    assert round(data['metrics']['cat']['accuracy'] * 100, 2) == 80.89
    print("  [PASS] /api/evaluation/metrics returns authentic verified metrics")

    # Check Home Page Hero Title
    home_r = requests.get('http://127.0.0.1:5000/')
    assert home_r.status_code == 200
    assert "Animal Skin Disease Detection Using Deep Learning" in home_r.text
    print("  [PASS] Home page contains exact requested Hero Title")

    print("\nALL DATASET DASHBOARD CHECKS PASSED SUCCESSFULLY!")

if __name__ == '__main__':
    test_dataset_dashboard()
