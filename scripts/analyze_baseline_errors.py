import json

for species in ['dog', 'cat']:
    with open(f'data/baseline_{species}_eval.json') as f:
        data = json.load(f)
    print('='*60)
    print(f'{species.upper()} BASELINE ERROR ANALYSIS (TEST SET)')
    print('='*60)
    rep = data['test']['report']
    classes = [k for k in rep.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    sorted_by_f1 = sorted(classes, key=lambda c: rep[c]['f1-score'])
    print('Bottom 5 classes by F1-score:')
    for c in sorted_by_f1[:5]:
        p = rep[c]['precision']
        r = rep[c]['recall']
        f1 = rep[c]['f1-score']
        sup = rep[c]['support']
        print(f"  {c:25s}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}, Support={sup}")
    print('\nTop 5 classes by F1-score:')
    for c in sorted_by_f1[-5:]:
        p = rep[c]['precision']
        r = rep[c]['recall']
        f1 = rep[c]['f1-score']
        sup = rep[c]['support']
        print(f"  {c:25s}: P={p:.3f}, R={r:.3f}, F1={f1:.3f}, Support={sup}")

