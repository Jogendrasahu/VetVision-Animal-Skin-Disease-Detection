"""
Generate fixtures for the 14-scenario species validation test suite.
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageFilter

FIXTURE_DIR = "test_fixtures"
os.makedirs(FIXTURE_DIR, exist_ok=True)

# Sources
DOG_SRC = "static/uploads/D_F.jpeg"
CAT_SRC = "static/uploads/Cat_Sample.jpg"
COW_SRC = "static/uploads/C_H.jpg"
SCREEN_SRC = "static/uploads/Screenshot_2026-09-27_221918.png"

# 1. Base test images
img_dog = cv2.imread(DOG_SRC)
img_cat = cv2.imread(CAT_SRC)
img_cow = cv2.imread(COW_SRC)
img_screen = cv2.imread(SCREEN_SRC)

cv2.imwrite(os.path.join(FIXTURE_DIR, "dog_sample.jpg"), img_dog)
cv2.imwrite(os.path.join(FIXTURE_DIR, "cat_sample.jpg"), img_cat)
cv2.imwrite(os.path.join(FIXTURE_DIR, "cow_sample.jpg"), img_cow)
cv2.imwrite(os.path.join(FIXTURE_DIR, "screen_sample.png"), img_screen)

# 2. Blurry / Unclear animal image
pil_dog = Image.open(DOG_SRC)
blurry_img = pil_dog.filter(ImageFilter.GaussianBlur(radius=18))
blurry_img.save(os.path.join(FIXTURE_DIR, "blurry_dog.jpg"), quality=30)

# Helper to write MP4
def write_video(filename, frames, fps=10):
    path = os.path.join(FIXTURE_DIR, filename)
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    print(f"Generated video: {path} ({len(frames)} frames)")
    return path

# 3. Valid Dog video (25 frames, subtle pan)
dog_frames = []
h, w = img_dog.shape[:2]
for i in range(25):
    dx = int(np.sin(i / 3.0) * 8)
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    shifted = cv2.warpAffine(img_dog, M, (w, h))
    dog_frames.append(shifted)
write_video("valid_dog_video.mp4", dog_frames)

# 4. Valid Cat video (25 frames)
cat_frames = []
h, w = img_cat.shape[:2]
for i in range(25):
    dx = int(np.sin(i / 3.0) * 8)
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    shifted = cv2.warpAffine(img_cat, M, (w, h))
    cat_frames.append(shifted)
write_video("valid_cat_video.mp4", cat_frames)

# 5. Valid Cow video (25 frames)
cow_frames = []
h, w = img_cow.shape[:2]
for i in range(25):
    dx = int(np.sin(i / 3.0) * 8)
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    shifted = cv2.warpAffine(img_cow, M, (w, h))
    cow_frames.append(shifted)
write_video("valid_cow_video.mp4", cow_frames)

# 6. Multi-species video (15 Dog frames + 15 Cat frames resized to same dimension)
target_size = (300, 300)
dog_resized = cv2.resize(img_dog, target_size)
cat_resized = cv2.resize(img_cat, target_size)
multi_frames = [dog_resized] * 15 + [cat_resized] * 15
write_video("multi_species_video.mp4", multi_frames)

# 7. Insufficient usable frames video (all heavily blurred frames)
blurred_frame = cv2.GaussianBlur(dog_resized, (51, 51), 0)
blurry_frames = [blurred_frame] * 20
write_video("blurry_video.mp4", blurry_frames)

# 8. Corrupted / empty video
with open(os.path.join(FIXTURE_DIR, "corrupt_video.mp4"), "wb") as f:
    f.write(b"NOT_A_VALID_MP4_HEADER_CORRUPTED_FILE_DATA")

print("All 14 test scenario fixtures successfully created in test_fixtures/")

