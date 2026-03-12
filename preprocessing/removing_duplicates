import os
import cv2
import numpy as np
import multiprocessing
from tqdm import tqdm
from imagehash import phash
from PIL import Image
from itertools import combinations

# Path to your image folder
IMAGE_FOLDER = "image_path"

# Load all image file paths
image_paths = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Function to compute perceptual hash
def compute_phash(image_path):
    try:
        image = Image.open(image_path).convert('L').resize((8, 8))
        return image_path, phash(image)
    except Exception as e:
        return image_path, None  # Return None if there's an error

# Function to compute 2D correlation coefficient for three RGB channels
def compute_correlation(pair):
    img1_path, img2_path = pair
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        return (img1_path, img2_path, 0)  # Skip invalid images

    # Ensure images have the same resolution
    if img1.shape[:2] != img2.shape[:2]:
        return (img1_path, img2_path, 0)

    corr = []
    for i in range(3):  # Check R, G, B channels separately
        channel1 = img1[:, :, i].astype(np.float32)
        channel2 = img2[:, :, i].astype(np.float32)

        # Compute mean of each channel
        mean1, mean2 = np.mean(channel1), np.mean(channel2)

        # Compute correlation coefficient
        numerator = np.sum((channel1 - mean1) * (channel2 - mean2))
        denominator = np.sqrt(np.sum((channel1 - mean1) ** 2) * np.sum((channel2 - mean2) ** 2))

        corr_value = numerator / (denominator + 1e-8)  # Avoid division by zero
        corr.append(corr_value)

    # If all channels have correlation close to 0.9995, mark as duplicate
    is_duplicate = int(all(c >= 0.9995 for c in corr))
    return (img1_path, img2_path, is_duplicate)

# Step 1: Compute pHash for all images (Parallel Processing)
with multiprocessing.Pool(processes=8) as pool:
    phash_results = list(tqdm(pool.imap_unordered(compute_phash, image_paths), total=len(image_paths)))

# Filter out failed images
phash_results = [res for res in phash_results if res[1] is not None]
image_hashes = {path: hash_val for path, hash_val in phash_results}

# Step 2: Find similar images using pHash (Hamming distance <= 5)
similar_images = [(p1, p2) for (p1, h1), (p2, h2) in combinations(image_hashes.items(), 2) if h1 - h2 <= 5]

# Step 3: Compute full 2D correlation only on filtered pairs (Parallel Processing)
with multiprocessing.Pool(processes=8) as pool:
    duplicate_results = list(tqdm(pool.imap_unordered(compute_correlation, similar_images), total=len(similar_images)))

# Step 4: Save duplicate images
duplicates = [(img1, img2) for img1, img2, is_dup in duplicate_results if is_dup]
np.savetxt("0.9995_duplicates.txt", duplicates, fmt="%s")

print(f"Found {len(duplicates)} duplicate image pairs! Results saved in 'duplicates.txt'.")
