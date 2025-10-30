import json
import os
import shutil
import numpy as np
from random import shuffle

# --- Base Paths ---
base_path = './train-sample-videos/'
dataset_path = './prepared-dataset/'
tmp_fake_path = './tmp-fake-faces'

print(f'Creating Directory: {dataset_path}')
os.makedirs(dataset_path, exist_ok=True)

print(f'Creating Directory: {tmp_fake_path}')
os.makedirs(tmp_fake_path, exist_ok=True)


def get_filename_only(file_path):
    """Extract filename without extension."""
    return os.path.splitext(os.path.basename(file_path))[0]


# --- Load Metadata ---
metadata_path = os.path.join(base_path, 'metadata.json')
if not os.path.exists(metadata_path):
    print(f"❌ metadata.json not found at {metadata_path}")
    exit(1)

with open(metadata_path) as metadata_json:
    metadata = json.load(metadata_json)
    print(f"Total video entries found: {len(metadata)}")

# --- Create Real/Fake Directories ---
real_path = os.path.join(dataset_path, 'real')
fake_path = os.path.join(dataset_path, 'fake')

for path in [real_path, fake_path]:
    print(f'Creating Directory: {path}')
    os.makedirs(path, exist_ok=True)

# --- Copy Faces Based on Metadata ---
for filename, info in metadata.items():
    label = info.get('label', '').upper()
    tmp_path = os.path.join(base_path, get_filename_only(filename), 'faces')

    if not os.path.exists(tmp_path):
        print(f"⚠️ Skipping {filename} (no 'faces' folder found)")
        continue

    print(f"\nProcessing {filename} ({label})")

    dest_dir = real_path if label == 'REAL' else tmp_fake_path if label == 'FAKE' else None
    if not dest_dir:
        print(f"⚠️ Unknown label for {filename}, skipping...")
        continue

    for item in os.listdir(tmp_path):
        src = os.path.join(tmp_path, item)
        dst = os.path.join(dest_dir, item)
        try:
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        except Exception as e:
            print(f"⚠️ Error copying {item}: {e}")

# --- Count and Balance Dataset ---
all_real_faces = [f for f in os.listdir(real_path) if os.path.isfile(os.path.join(real_path, f))]
all_fake_faces = [f for f in os.listdir(tmp_fake_path) if os.path.isfile(os.path.join(tmp_fake_path, f))]

print(f"\nTotal Real Faces: {len(all_real_faces)}")
print(f"Total Fake Faces (before balancing): {len(all_fake_faces)}")

# Down-sample fake faces to match real count
if len(all_fake_faces) > len(all_real_faces):
    random_faces = np.random.choice(all_fake_faces, len(all_real_faces), replace=False)
else:
    random_faces = all_fake_faces

for fname in random_faces:
    src = os.path.join(tmp_fake_path, fname)
    dst = os.path.join(fake_path, fname)
    shutil.copyfile(src, dst)

print("✅ Down-sampling Done!")

# --- Split Dataset into Train/Val/Test ---
def split_data(src_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=1377):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    for category in ['real', 'fake']:
        category_path = os.path.join(src_dir, category)
        files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
        shuffle(files)
        n_total = len(files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        splits = {
            'train': files[:n_train],
            'val': files[n_train:n_train + n_val],
            'test': files[n_train + n_val:]
        }

        for split_name, split_files in splits.items():
            split_folder = os.path.join(output_dir, split_name, category)
            os.makedirs(split_folder, exist_ok=True)
            for f in split_files:
                shutil.copy2(os.path.join(category_path, f), os.path.join(split_folder, f))

    print("✅ Train/Val/Test Split Done!")

split_data(dataset_path, 'split_dataset', train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
