import torch
import matplotlib.pyplot as plt
from torchcam.methods import SmoothGradCAMpp
import numpy as np
import cv2


def generate_gradcam(model, dataset, device, save_path="results/gradcam_example.png"):

    model.eval()

    # Get sample
    image_with_attn, label = dataset[1]

    original_image = image_with_attn[:3]
    image_with_attn = image_with_attn.unsqueeze(0).to(device)

    cam_extractor = SmoothGradCAMpp(model, target_layer="conv5")

    # Forward pass
    output = model(image_with_attn)
    _, predicted_class = torch.max(output, 1)

    print(f"Predicted class: {predicted_class.item()}, Actual class: {label}")

    # Generate activation map
    activation_map = cam_extractor(predicted_class.item(), output)[0]

    activation_map = activation_map.squeeze().cpu().numpy()

    activation_map = (activation_map - activation_map.min()) / (
        activation_map.max() - activation_map.min()
    )

    activation_map = cv2.resize(activation_map, (256, 256))

    threshold = 0.7
    binary_mask = (activation_map > threshold).astype(np.uint8) * 255

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(original_image.cpu().permute(1, 2, 0).numpy())
    ax[0].set_title("Original Image")

    ax[1].imshow(activation_map, cmap="jet")
    ax[1].set_title("GradCAM Heatmap")

    ax[2].imshow(binary_mask, cmap="gray")
    ax[2].set_title("Binary Lesion Mask")

    plt.savefig(save_path)
    plt.close()

    print("GradCAM saved to:", save_path)
