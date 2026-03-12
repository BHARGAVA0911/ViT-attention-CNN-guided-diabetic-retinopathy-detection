# ViT-attention-CNN-guided-diabetic-retinopathy-detection

This paper proposes a new ViT+ CNN hybrid model for DR classification by utilizing CNNs heterogeneous activation facility for feature extraction and ViTs attention maps to remain aware of the global context. 

Weighted Random Sampler is used to effectively counter class imbalance and to improve minority class representation and stabilize training. 

A Vision Transformer (ViT) model is pre-trained on fundus images to produce attention maps that are added as extra input channels to CNN-based training to improve interpretability and robustness. 

# Dataset
The dataset is obtained from Asia Pacific Tele-Ophthalmology Society (APTOS)· It consists of large set (3428) of retina images taken using fundus photography under a variety of imaging conditions. A clinician has rated each image for the severity of diabetic retinopathy on a scale of 0 to 4:

0 - No DR
1 - Mild
2 - Moderate
3 - Severe
4 - Proliferative DR

# Baseline Models

I. ViTs for Extracting attention maps:

1. The model extracts spatial attention maps which represent regions in the image that the ViT considers most relevant for classification.
2. This map is added as a fourth channel to the standard RGB image. The resulting 4-channel input is passed into a custom CNN with heterogeneous activation functions. The custom CNN model has its first convolutional layer configured to accept 4 input channels. 

II.  CNN for Classification:

1. The custom CNN model has its first convolutional layer configured to accept 4 input channels. 
2. The CNN simultaneously learns to extract features from both, raw pixel values (RGB channels), and the ViT-derived semantic attention map, which highlights diagnostically relevant regions. 
3. This design preserves simplicity while improving model explainability and convergence.

# System Diagram
![Untitled](https://github.com/user-attachments/assets/d8c8e4d4-14da-419d-b8a4-32d1ecdb1654)




