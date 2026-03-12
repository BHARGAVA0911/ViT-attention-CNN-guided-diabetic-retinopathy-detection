# ViT-attention-CNN-guided-diabetic-retinopathy-detection

Diabetic retinopathy (DR) is a top contributor to visual impairment, for which early diagnosis is essential to impact effectively. State-of-the-art deep learning methods are impeded by low interpretability and unbalanced class handling, thereby impairing clinical feasibility. 
In this work, we propose an innovative deep learning framework that combines Vision Transformers (ViT) with Convolutional Neural Networks (CNNs) to automate the initial screening procedure for diabetic retinopathy. The goal is to improve the diagnostic efficiency and accuracy, thereby helping ophthalmologist in making faster and more reliable choices. 
The proposed model uses attention maps from ViT and integrates heterogeneous activation functions within CNN to enable effective extraction of features. The dataset featured in Aptos Kaggle competition was used for this study. The hybrid model was evaluated using accuracy, AUC-ROC, sensitivity, and specificity. 
The model performed remarkably well demonstrating that our approach enhances classification accuracy and interpretability, offering a robust solution for automated DR screening. It can be derived that the model can reliably identify and distinguish among severities of DR.

# Dataset
The dataset is obtained from Asia Pacific Tele-Ophthalmology Society (APTOS)· It consists of large set (3428) of retina images taken using fundus photography under a variety of imaging conditions. A clinician has rated each image for the severity of diabetic retinopathy on a scale of 0 to 4:

0 - No DR
1 - Mild
2 - Moderate
3 - Severe
4 - Proliferative DR

# Methodology

I.  Handling Class imbalance
 
1. The dataset shows extreme class imbalance tilted towards the non - DR class with fewer instances in the more extreme classes. This tends to lead the model being biased toward the majority class (No DR) and exhibiting poor performance on minority extreme cases. 
2. We employed a Weighted Random Sampler that guarantees that each class will contribute proportionally to training without undersampling or oversampling, which is done explicitly. This method adds weights to every sample with the inverse of the class frequency.
   
II. Handling duplicate data
 
1. The data contains minimal repeated samples, but most of them labeled with varied name and labels. 
2. These repeated samples have the tendency to cause label noise, confusing the model when it is being trained and affecting performance. 
3. At first, we calculated perceptual hashes for all the images. Images with a Hamming distance of ≤ 5 were flagged as possible duplicates. Next, in order to validate duplication, we computed pixel-wise correlation coefficient over RGB channels. 
4. Image pairs with a correlation coefficient of ≥ 0.9995 for all three channels were marked as duplicates. This technique efficiently removed near-duplicate images with low false positives. 

III. Data pre-processing

The shape of the images in the dataset are irregular, as they were collected from different sources. To improve the quality of fundus images and enhance model performance, preprocessing methods such as gaussian blur with CLAHE (Contrast Limited Adaptive Histogram Equalization) was used.

IV. ViTs for Extracting attention maps:

1. The model extracts spatial attention maps which represent regions in the image that the ViT considers most relevant for classification.
2. This map is added as a fourth channel to the standard RGB image. The resulting 4-channel input is passed into a custom CNN with heterogeneous activation functions. The custom CNN model has its first convolutional layer configured to accept 4 input channels. 

V.  CNN for Classification:

1. The custom CNN model has its first convolutional layer configured to accept 4 input channels. 
2. The CNN simultaneously learns to extract features from both, raw pixel values (RGB channels), and the ViT-derived semantic attention map, which highlights diagnostically relevant regions. 

# System Diagram
![Untitled](https://github.com/user-attachments/assets/d8c8e4d4-14da-419d-b8a4-32d1ecdb1654)

#




