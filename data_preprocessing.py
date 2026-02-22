import os
import shutil
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 🐱 Step 1: Define Paths
dataset_path = 'oxford iiit pet dataset/images'  # Adjust to your local path
output_dir = 'cat_dataset_split'  # Output folder

# Step 2: Setup Breed List and Folders
cat_breeds = [
    'Abyssinian', 'Bengal', 'Birman', 'Bombay',
    'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon',
    'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese',
    'Sphynx'
]
cat_breeds_lower = [b.lower() for b in cat_breeds]
splits = ['train', 'val', 'test']
IMG_SIZE = (224, 224)
split_ratio = {'train': 0.7, 'val': 0.15, 'test': 0.15}

for split in splits:
    for breed in cat_breeds:
        os.makedirs(os.path.join(output_dir, split, breed), exist_ok=True)

#Step 3: Filter Cat Images
cat_images = {breed: [] for breed in cat_breeds}

for filename in tqdm(os.listdir(dataset_path), desc="Filtering Cat Breeds"):
    if not filename.endswith('.jpg'):
        continue
    breed_name = '_'.join(filename.split('_')[:-1])
    breed_name_lower = breed_name.lower()
 
    if breed_name_lower in cat_breeds_lower:
        matched_breed = cat_breeds[cat_breeds_lower.index(breed_name_lower)]
        cat_images[matched_breed].append(filename)

# 🧼 Step 4: Resize, Normalize, and Split
def resize_and_normalize(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        img = img.resize(IMG_SIZE)
        img_array = np.array(img) / 255.0  # Normalize to 0–1
        return Image.fromarray((img_array * 255).astype(np.uint8))
    except:
        return None

for breed, filenames in tqdm(cat_images.items(), desc="Processing & Splitting"):
    random.shuffle(filenames)
    total = len(filenames)
    train_end = int(split_ratio['train'] * total)
    val_end = train_end + int(split_ratio['val'] * total)

    split_map = {
        'train': filenames[:train_end],
        'val': filenames[train_end:val_end],
        'test': filenames[val_end:]
    }

    for split, files in split_map.items():
        for f in files:
            src = os.path.join(dataset_path, f)
            dst = os.path.join(output_dir, split, breed, f)
            processed = resize_and_normalize(src)
            if processed:
                processed.save(dst)

# 🔁 Step 5: Data Augmentation (Train Set Only)
datagen = ImageDataGenerator(
    rotation_range=20,
    horizontal_flip=True,
    fill_mode='nearest'
)

AUG_PER_IMG = 2

for breed in tqdm(cat_breeds, desc="Augmenting Training Images"):
    folder_path = os.path.join(output_dir, 'train', breed)
    images = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

    for img_file in images:
        img_path = os.path.join(folder_path, img_file)
        img = Image.open(img_path).convert('RGB')
        img = img.resize(IMG_SIZE)
        x = np.expand_dims(np.array(img) / 255.0, axis=0)

        aug_iter = datagen.flow(x, batch_size=1)
        for i in range(AUG_PER_IMG):
            aug_img = next(aug_iter)[0]
            aug_img = (aug_img * 255).astype(np.uint8)
            aug_img = Image.fromarray(aug_img)
            aug_img.save(os.path.join(folder_path, f"{img_file.split('.')[0]}_aug{i+1}.jpg"))

print("\n✅ All done! Processed dataset is saved in ./cat_dataset_split/")
