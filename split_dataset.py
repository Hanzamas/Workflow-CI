"""
split_dataset.py
Script untuk split Rice Image Dataset jadi train/val/test
Dijalankan di CI setelah download dari Kaggle.
"""
import os
import shutil
import random
from pathlib import Path

# Konfigurasi
SOURCE_DIR  = Path("MLProject/data_raw/Rice_Image_Dataset")
TARGET_DIR  = Path("MLProject/data_split")
SPLIT_RATIO = (0.7, 0.15, 0.15)
MAX_PER_CLASS = 200   # Batasi per kelas biar CI cepat (total 1000 gambar)
SEED = 42

random.seed(SEED)

print("=" * 50)
print("Splitting Rice Image Dataset for CI...")
print(f"Source : {SOURCE_DIR}")
print(f"Target : {TARGET_DIR}")
print(f"Max/class: {MAX_PER_CLASS}")
print("=" * 50)

if not SOURCE_DIR.exists():
    print(f"ERROR: {SOURCE_DIR} tidak ditemukan!")
    exit(1)

# Bersihkan target
if TARGET_DIR.exists():
    shutil.rmtree(TARGET_DIR)

classes = [d.name for d in SOURCE_DIR.iterdir() if d.is_dir()]
print(f"Kelas ditemukan: {classes}\n")

for cls in sorted(classes):
    src_cls = SOURCE_DIR / cls
    images  = list(src_cls.glob("*.jpg")) + list(src_cls.glob("*.png"))
    random.shuffle(images)
    images  = images[:MAX_PER_CLASS]   # ambil maksimal N gambar

    n       = len(images)
    n_train = int(SPLIT_RATIO[0] * n)
    n_val   = int(SPLIT_RATIO[1] * n)

    splits = {
        "train": images[:n_train],
        "val"  : images[n_train:n_train + n_val],
        "test" : images[n_train + n_val:],
    }

    for split_name, imgs in splits.items():
        out_dir = TARGET_DIR / split_name / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy2(img, out_dir / img.name)

    print(f"  {cls:<12}: {len(splits['train'])} train | {len(splits['val'])} val | {len(splits['test'])} test")

total = len(list(TARGET_DIR.rglob("*.jpg"))) + len(list(TARGET_DIR.rglob("*.png")))
print(f"\nTotal gambar di data_split: {total}")
print("Done!")
