import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def ben_graham_clahe_preprocessing(image_path):
    # Step 1: Read the image
    img = cv2.imread(image_path)

    # Step 2: Apply Gaussian Blur (Ben Graham Method)
    blurred_img = cv2.GaussianBlur(img, (0, 0), 30)  # Kernel size (0,0), SigmaX=10

    # Blend original and blurred images
    blend_img = cv2.addWeighted(img, 4, blurred_img, -4, 128)

    # Convert to grayscale (Ben Graham method results in grayscale)
    gray_img = cv2.cvtColor(blend_img, cv2.COLOR_BGR2GRAY)

    # Step 3: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray_img)

    # Convert grayscale back to 3 channels (since CLAHE works best on BGR)
    final_img = cv2.merge([clahe_img, clahe_img, clahe_img])

    # Step 4: Resize to 224x224
    final_img = cv2.resize(final_img, (224, 224))

    # Step 5: Normalize (Scale pixel values from [0,255] to [0,1])
    final_img = final_img.astype(np.float32) / 255.0

    return final_img
