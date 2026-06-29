"""
create_test_data.py
Generate dataset sintetis kecil untuk keperluan CI testing.
Membuat 5 kelas beras dengan 50 gambar dummy per kelas.
Tidak perlu dataset asli — murni untuk memvalidasi pipeline CI.
"""
import os
import random
import numpy as np
from pathlib import Path
from PIL import Image

CLASSES     = ["Arborio", "Basmati", "Ipsala", "Jasmine", "Karacadag"]
IMG_SIZE    = (64, 64)
N_PER_CLASS = 50           # cukup untuk CI, cepat selesai
SPLIT_RATIO = (0.7, 0.15, 0.15)
TARGET_DIR  = Path("MLProject/data_split")

print("=" * 50)
print("Creating synthetic test dataset for CI...")
print("=" * 50)

# Warna berbeda per kelas supaya model bisa belajar
CLASS_COLORS = {
    "Arborio"   : (220, 200, 150),
    "Basmati"   : (240, 230, 180),
    "Ipsala"    : (200, 180, 120),
    "Jasmine"   : (230, 215, 160),
    "Karacadag" : (190, 170, 110),
}

def make_fake_image(color, size=IMG_SIZE):
    """Buat gambar solid dengan noise kecil."""
    arr = np.full((*size, 3), color, dtype=np.uint8)
    noise = np.random.randint(-15, 15, arr.shape, dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

# Bersihkan & buat folder
if TARGET_DIR.exists():
    import shutil
    shutil.rmtree(TARGET_DIR)

for cls in CLASSES:
    images = []
    for i in range(N_PER_CLASS):
        img = make_fake_image(CLASS_COLORS[cls])
        images.append(img)

    # Split
    random.shuffle(images)
    n_train = int(SPLIT_RATIO[0] * N_PER_CLASS)
    n_val   = int(SPLIT_RATIO[1] * N_PER_CLASS)

    splits = {
        "train" : images[:n_train],
        "val"   : images[n_train:n_train + n_val],
        "test"  : images[n_train + n_val:],
    }

    for split_name, imgs in splits.items():
        folder = TARGET_DIR / split_name / cls
        folder.mkdir(parents=True, exist_ok=True)
        for idx, img in enumerate(imgs):
            img.save(folder / f"{cls}_{idx:04d}.jpg")

    print(f"  {cls}: {N_PER_CLASS} images created")

# Verifikasi
print("\nVerifikasi:")
for split in ["train", "val", "test"]:
    count = len(list((TARGET_DIR / split).glob("**/*.jpg")))
    print(f"  {split}: {count} images")

print("\nDone! Synthetic dataset ready for CI.")
