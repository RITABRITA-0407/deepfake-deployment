#
# import os
# import shutil
# import pandas as pd
# from pathlib import Path
# import random
#
# # ===== CONFIGURATION =====
# SOURCE_DATA_DIR = r"C:\Users\weddi\Downloads\archive\real_vs_fake\real-vs-fake"  # Change this to your local dataset path
# PROJECT_FOLDER = os.getcwd()
# # PROJECT_FOLDER = r"C:\Users\weddi\PycharmProjects\PythonProject"  # Your project folder
# RANDOM_SEED = 42
#
# # Sample sizes per class (real/fake)
# SAMPLE_SIZES = {
#     'train': 500,  # 500 real + 500 fake = 1000 total
#     'valid': 250,  # 250 real + 250 fake = 500 total
# }
#
# # ===== SETUP =====
# random.seed(RANDOM_SEED)
# os.makedirs(PROJECT_FOLDER, exist_ok=True)
#
# # Image extensions to look for
# IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
#
# # ===== STEP 1: SCAN FOLDERS AND CREATE CSV =====
# print("🔍 Scanning image folders...")
# data_records = []
#
# for split in ['train', 'valid']:
#     split_dir = os.path.join(SOURCE_DATA_DIR, split)
#
#     if not os.path.exists(split_dir):
#         print(f"⚠️  {split_dir} not found, skipping...")
#         continue
#
#     # Scan for real and fake subfolders
#     for class_name in ['real', 'fake']:
#         class_path = os.path.join(split_dir, class_name)
#
#         if not os.path.isdir(class_path):
#             print(f"⚠️  {class_path} not found, skipping...")
#             continue
#
#         # Find all images in this class folder
#         image_count = 0
#         for image_file in os.listdir(class_path):
#             if Path(image_file).suffix.lower() in IMAGE_EXTENSIONS:
#                 image_path = os.path.join(class_path, image_file)
#                 data_records.append({
#                     'split': split,
#                     'class': class_name,
#                     'image_name': image_file,
#                     'image_path': image_path
#                 })
#                 image_count += 1
#
#         print(f"  Found {image_count} images in {split}/{class_name}")
#
# print(f"\n✅ Total images scanned: {len(data_records)}")
#
# # ===== STEP 2: RANDOM SAMPLING PER SPLIT & CLASS =====
# print(f"\n🎲 Randomly sampling images...")
# sampled_records = []
#
# for split in ['train', 'valid']:
#     sample_size = SAMPLE_SIZES.get(split, 0)
#
#     for class_name in ['real', 'fake']:
#         # Filter records for this split and class
#         split_class_records = [r for r in data_records
#                                if r['split'] == split and r['class'] == class_name]
#
#         if len(split_class_records) == 0:
#             print(f"⚠️  No images found for {split}/{class_name}")
#             continue
#
#         # Random sample
#         sample = random.sample(split_class_records,
#                                min(sample_size, len(split_class_records)))
#         sampled_records.extend(sample)
#         print(f"  {split}/{class_name}: {len(sample)} images sampled (out of {len(split_class_records)})")
#
# print(f"\n📊 Total sampled: {len(sampled_records)} images")
#
# # ===== STEP 3: CREATE CSV FROM SAMPLED DATA =====
# csv_path = os.path.join(PROJECT_FOLDER, 'dataset.csv')
# df = pd.DataFrame(sampled_records)
# df.to_csv(csv_path, index=False)
# print(f"\n✅ CSV created: {csv_path}")
# print(f"\nCSV Preview (first 10 rows):")
# print(df.head(10))
#
# # ===== STATISTICS =====
# print(f"\n📊 Dataset Statistics:")
# for split in ['train', 'valid']:
#     split_df = df[df['split'] == split]
#     if len(split_df) > 0:
#         real_count = len(split_df[split_df['class'] == 'real'])
#         fake_count = len(split_df[split_df['class'] == 'fake'])
#         print(f"  {split}: {real_count} real + {fake_count} fake = {len(split_df)} total")
#
# # ===== STEP 4: COPY IMAGES TO PROJECT FOLDER =====
# print(f"\n📁 Copying {len(sampled_records)} images to {PROJECT_FOLDER}...")
#
# failed_copies = 0
# for idx, row in df.iterrows():
#     split = row['split']
#     class_name = row['class']
#     image_name = row['image_name']
#     source_path = row['image_path']
#
#     # Create directory structure: project_data/split/class/
#     target_dir = os.path.join(PROJECT_FOLDER, split, class_name)
#     os.makedirs(target_dir, exist_ok=True)
#
#     target_path = os.path.join(target_dir, image_name)
#
#     try:
#         shutil.copy2(source_path, target_path)
#         if (idx + 1) % 100 == 0:
#             print(f"  Copied {idx + 1}/{len(sampled_records)} images...")
#     except Exception as e:
#         print(f"⚠️  Failed to copy {image_name}: {e}")
#         failed_copies += 1
#
# print(f"✅ Copy complete! ({failed_copies} failed)")
#
# # ===== SUMMARY =====
# print("\n" + "=" * 60)
# print("📋 FINAL SUMMARY")
# print("=" * 60)
# print(f"CSV file: {csv_path}")
# print(f"Images location: {PROJECT_FOLDER}")
# print(f"Total images processed: {len(sampled_records)}")
# print(f"\nBreakdown:")
# print(f"  Train: 500 real + 500 fake = 1000 images")
# print(f"  Valid: 250 real + 250 fake = 500 images")
# print(f"  TOTAL: 1500 images")
# print("=" * 60)
# print(f"\nProject structure created:")
# print(f"{PROJECT_FOLDER}/")
# print(f"├── dataset.csv")
# print(f"├── train/")
# print(f"│   ├── real/ (500 images)")
# print(f"│   └── fake/ (500 images)")
# print(f"├── valid/")
# print(f"    ├── real/ (250 images)")
# print(f"    └── fake/ (250 images)")
# print("=" * 60)

# import os
# import shutil
# import pandas as pd
# from pathlib import Path
# import random
#
# # ===== CONFIGURATION =====
# SOURCE_DATA_DIR = r"C:\Users\RITABRITA\Downloads\archive (7)\real_vs_fake\real-vs-fake"
# PROJECT_FOLDER = os.getcwd()
# RANDOM_SEED = 42
#
# # Sample sizes PER CLASS (real/fake)
# # train: 1000 real + 1000 fake = 2000
# # valid: 500 real + 500 fake = 1000
# SAMPLE_SIZES = {
#     'train': 1000,
#     'valid': 500,
# }
#
# # ===== SETUP =====
# random.seed(RANDOM_SEED)
# os.makedirs(PROJECT_FOLDER, exist_ok=True)
#
# IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
#
# # ===== STEP 1: SCAN FOLDERS =====
# print("🔍 Scanning image folders...")
# data_records = []
#
# for split in ['train', 'valid']:
#     split_dir = os.path.join(SOURCE_DATA_DIR, split)
#
#     if not os.path.exists(split_dir):
#         print(f"⚠️  {split_dir} not found, skipping...")
#         continue
#
#     for class_name in ['real', 'fake']:
#         class_path = os.path.join(split_dir, class_name)
#
#         if not os.path.isdir(class_path):
#             print(f"⚠️  {class_path} not found, skipping...")
#             continue
#
#         image_count = 0
#         for image_file in os.listdir(class_path):
#             if Path(image_file).suffix.lower() in IMAGE_EXTENSIONS:
#                 image_path = os.path.join(class_path, image_file)
#                 data_records.append({
#                     'split': split,
#                     'class': class_name,
#                     'image_name': image_file,
#                     'image_path': image_path
#                 })
#                 image_count += 1
#
#         print(f"  Found {image_count} images in {split}/{class_name}")
#
# print(f"\n✅ Total images scanned: {len(data_records)}")
#
# # ===== STEP 2: RANDOM SAMPLING =====
# print(f"\n🎲 Randomly sampling images...")
# sampled_records = []
#
# for split in ['train', 'valid']:
#     sample_size = SAMPLE_SIZES.get(split, 0)
#
#     for class_name in ['real', 'fake']:
#         split_class_records = [
#             r for r in data_records
#             if r['split'] == split and r['class'] == class_name
#         ]
#
#         if len(split_class_records) == 0:
#             print(f"⚠️  No images found for {split}/{class_name}")
#             continue
#
#         sample = random.sample(
#             split_class_records,
#             min(sample_size, len(split_class_records))
#         )
#
#         sampled_records.extend(sample)
#
#         print(f"  {split}/{class_name}: "
#               f"{len(sample)} sampled (out of {len(split_class_records)})")
#
# print(f"\n📊 Total sampled: {len(sampled_records)} images")
#
# # ===== STEP 3: CREATE CSV =====
# csv_path = os.path.join(PROJECT_FOLDER, 'dataset.csv')
# df = pd.DataFrame(sampled_records)
# df.to_csv(csv_path, index=False)
#
# print(f"\n✅ CSV created: {csv_path}")
# print("\nCSV Preview:")
# print(df.head())
#
# # ===== DATASET STATISTICS =====
# print("\n📊 Dataset Statistics:")
# for split in SAMPLE_SIZES.keys():
#     split_df = df[df['split'] == split]
#     real_count = len(split_df[split_df['class'] == 'real'])
#     fake_count = len(split_df[split_df['class'] == 'fake'])
#
#     print(f"  {split}: {real_count} real + {fake_count} fake = {len(split_df)} total")
#
# # ===== STEP 4: COPY IMAGES =====
# print(f"\n📁 Copying {len(sampled_records)} images...")
#
# failed_copies = 0
#
# for idx, row in df.iterrows():
#     target_dir = os.path.join(PROJECT_FOLDER, row['split'], row['class'])
#     os.makedirs(target_dir, exist_ok=True)
#
#     target_path = os.path.join(target_dir, row['image_name'])
#
#     try:
#         shutil.copy2(row['image_path'], target_path)
#
#         if (idx + 1) % 200 == 0:
#             print(f"  Copied {idx + 1}/{len(sampled_records)} images...")
#
#     except Exception as e:
#         print(f"⚠️  Failed to copy {row['image_name']}: {e}")
#         failed_copies += 1
#
# print(f"✅ Copy complete! ({failed_copies} failed)")
#
# # ===== FINAL SUMMARY =====
# print("\n" + "=" * 60)
# print("📋 FINAL SUMMARY")
# print("=" * 60)
#
# total_train = SAMPLE_SIZES['train'] * 2
# total_valid = SAMPLE_SIZES['valid'] * 2
# total_images = total_train + total_valid
#
# print(f"Train: {SAMPLE_SIZES['train']} real + {SAMPLE_SIZES['train']} fake = {total_train}")
# print(f"Valid: {SAMPLE_SIZES['valid']} real + {SAMPLE_SIZES['valid']} fake = {total_valid}")
# print(f"TOTAL: {total_images} images")
# print("=" * 60)

import os
import shutil
import pandas as pd
from pathlib import Path
import random

# ===== CONFIGURATION =====
SOURCE_DATA_DIR = r"C:\Users\RITABRITA\Downloads\archive (7)\real_vs_fake\real-vs-fake"
PROJECT_FOLDER  = os.getcwd()
RANDOM_SEED     = 42

# Sample sizes PER CLASS (real/fake)
# train: 5250 real + 5250 fake = 10500
# valid: 1125  real + 1125  fake = 2250
# test:  1125  real + 1125  fake = 2250
SAMPLE_SIZES = {
    'train': 5250,
    'valid': 1125,
    'test' : 1125,
}

# ===== SETUP =====
random.seed(RANDOM_SEED)
os.makedirs(PROJECT_FOLDER, exist_ok=True)

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

# ===== STEP 1: SCAN FOLDERS =====
print("🔍 Scanning image folders...")
data_records = []

for split in SAMPLE_SIZES:
    split_dir = os.path.join(SOURCE_DATA_DIR, split)

    if not os.path.exists(split_dir):
        print(f"⚠️  {split_dir} not found, skipping...")
        continue

    for class_name in ['real', 'fake']:
        class_path = os.path.join(split_dir, class_name)

        if not os.path.isdir(class_path):
            print(f"⚠️  {class_path} not found, skipping...")
            continue

        image_count = 0
        for image_file in os.listdir(class_path):
            if Path(image_file).suffix.lower() in IMAGE_EXTENSIONS:
                image_path = os.path.join(class_path, image_file)
                data_records.append({
                    'split':      split,
                    'class':      class_name,
                    'image_name': image_file,
                    'image_path': image_path
                })
                image_count += 1

        print(f"  Found {image_count} images in {split}/{class_name}")

print(f"\n✅ Total images scanned: {len(data_records)}")

# ===== STEP 2: RANDOM SAMPLING =====
print(f"\n🎲 Randomly sampling images...")
sampled_records = []

for split, sample_size in SAMPLE_SIZES.items():
    for class_name in ['real', 'fake']:
        split_class_records = [
            r for r in data_records
            if r['split'] == split and r['class'] == class_name
        ]

        if len(split_class_records) == 0:
            print(f"⚠️  No images found for {split}/{class_name}")
            continue

        sample = random.sample(
            split_class_records,
            min(sample_size, len(split_class_records))
        )

        sampled_records.extend(sample)
        print(f"  {split}/{class_name}: "
              f"{len(sample)} sampled (out of {len(split_class_records)})")

print(f"\n📊 Total sampled: {len(sampled_records)} images")

# ===== STEP 3: CREATE CSV =====
csv_path = os.path.join(PROJECT_FOLDER, 'dataset_2.csv')
df = pd.DataFrame(sampled_records)
df.to_csv(csv_path, index=False)

print(f"\n✅ CSV created: {csv_path}")
print("\nCSV Preview:")
print(df.head())

# ===== DATASET STATISTICS =====
print("\n📊 Dataset Statistics:")
for split in SAMPLE_SIZES:
    split_df   = df[df['split'] == split]
    real_count = len(split_df[split_df['class'] == 'real'])
    fake_count = len(split_df[split_df['class'] == 'fake'])
    print(f"  {split}: {real_count} real + {fake_count} fake = {len(split_df)} total")

# ===== STEP 4: COPY IMAGES =====
print(f"\n📁 Copying {len(sampled_records)} images...")
failed_copies = 0

for idx, row in df.iterrows():
    target_dir  = os.path.join(PROJECT_FOLDER, row['split'], row['class'])
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, row['image_name'])

    try:
        shutil.copy2(row['image_path'], target_path)
        if (idx + 1) % 200 == 0:
            print(f"  Copied {idx + 1}/{len(sampled_records)} images...")
    except Exception as e:
        print(f"⚠️  Failed to copy {row['image_name']}: {e}")
        failed_copies += 1

print(f"✅ Copy complete! ({failed_copies} failed)")

# ===== FINAL SUMMARY =====
print("\n" + "=" * 60)
print("📋 FINAL SUMMARY")
print("=" * 60)

total_all = 0
for split, size in SAMPLE_SIZES.items():
    total = size * 2
    total_all += total
    print(f"{split.capitalize():6}: {size} real + {size} fake = {total}")

print(f"{'TOTAL':6}: {total_all} images")
print("=" * 60)