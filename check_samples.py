import os
import app

sample_files = [
    'static/uploads/D_F.jpeg',
    'static/uploads/D_U1.webp',
    'static/uploads/D_U2.jpg',
    'static/uploads/dog.jpg',
    'static/uploads/Cat_Sample.jpg',
    'static/uploads/C_H.jpg',
    'static/uploads/C_L.webp',
    'static/uploads/C_No1.png',
    'static/uploads/C_S.jpg',
    'static/uploads/C_UH.jpg',
    'static/uploads/C_UH1.jpg',
    'static/uploads/Screenshot_2026-09-27_221918.png',
    'static/uploads/plain.jpg',
    'static/uploads/blank.jpg',
    'static/uploads/ss.png',
    'static/uploads/i.jpg',
    'static/uploads/img.jpg',
]

print("-" * 115)
print(f"{'FILE PATH':<45} | {'GATEKEEPER STATUS':<22} | {'DETECTED':<8} | {'CONF':<7} | {'TOP-1 IMAGENET LABEL'}")
print("-" * 115)

for p in sample_files:
    if os.path.exists(p):
        try:
            res = app.identify_species(p)
            status = res['status']
            species = res['detected_species']
            conf = f"{res['confidence']:.1f}%"
            top_lbl = res.get('top_label', 'N/A')
            print(f"{p:<45} | {status:<22} | {species:<8} | {conf:<7} | {top_lbl}")
        except Exception as e:
            print(f"{p:<45} | ERROR: {e}")
print("-" * 115)

