import os
import shutil
import random
from tqdm import tqdm

# === 1️⃣ Set Paths ===
original_dataset_dir = r"C:\Users\ezwan\Documents\DEGREE\FINAL YEAR PROEJCT\animals\animals"
base_dir = "cat_vs_noncat_dataset"        # New folder to save the split dataset
os.makedirs(base_dir, exist_ok=True)

# === 2️⃣ Create Subdirectories ===
splits = ['train', 'val', 'test']
classes = ['cat', 'non_cat']

for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(base_dir, split, cls), exist_ok=True)

# === 3️⃣ Define Split Ratios ===
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# === 4️⃣ Get All Animal Folders ===
animal_classes = sorted(os.listdir(original_dataset_dir))

# === 5️⃣ Process Each Animal Folder ===
for animal in tqdm(animal_classes, desc="Processing animal classes"):
    animal_folder = os.path.join(original_dataset_dir, animal)
    if not os.path.isdir(animal_folder):
        continue

    # Get all image files for this animal
    images = [f for f in os.listdir(animal_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)

    # Split the data
    n_total = len(images)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)

    train_files = images[:n_train]
    val_files = images[n_train:n_train + n_val]
    test_files = images[n_train + n_val:]

    # Classify as 'cat' or 'non_cat'
    class_folder = 'cat' if animal.lower() == 'cat' else 'non_cat'

    # Copy files into new directories
    for split_name, file_list in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
        for file_name in file_list:
            src = os.path.join(animal_folder, file_name)
            dst = os.path.join(base_dir, split_name, class_folder, file_name)
            shutil.copy(src, dst)

print("\n✅ Dataset successfully split and organized!")
print(f"📂 Structure created in: {base_dir}")

