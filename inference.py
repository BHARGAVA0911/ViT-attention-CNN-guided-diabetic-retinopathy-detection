import torch
import numpy as np
from sklearn.metrics import (classification_report, confusion_matrix,
                            roc_auc_score, cohen_kappa_score,
                            mean_absolute_error)
import matplotlib.pyplot as plt
import seaborn as sns
from gradcam import generate_gradcam

generate_gradcam(model, test_dataset, device)

model.eval()
all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

# Convert to numpy arrays
y_true = np.array(all_labels)
y_pred = np.array(all_predictions)
y_probs = np.array(all_probabilities)

# 1. Standard Classification Metrics
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# 2. Ordinal-Specific Metrics
print("\nOrdinal Evaluation Metrics:")
# Quadratic Weighted Kappa (penalizes severity misclassifications)
kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
print(f"Quadratic Weighted Kappa: {kappa:.4f}")

# Mean Absolute Error (measures how far predictions are from true severity)
mae = mean_absolute_error(y_true, y_pred)
print(f"Mean Absolute Error: {mae:.4f}")

# 3. Enhanced Confusion Matrix with severity distance coloring
conf_matrix = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"fontsize":10})
plt.xlabel('Predicted Severity')
plt.ylabel('True Severity')
plt.title('Confusion Matrix (Warmer colors = larger severity misclassification)')
plt.show()

# 4. Class-wise ROC Curves with AUC
print("\nClass-wise AUC-ROC:")
plt.figure(figsize=(10, 8))
for i in range(len(class_names)):
    fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_probs[:, i])
    auc = roc_auc_score((y_true == i).astype(int), y_probs[:, i])
    plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC={auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves by DR Severity')
plt.legend(loc="lower right")
plt.show()

# 5. Ordinal Classification Analysis
print("\nSeverity Misclassification Analysis:")
for true_sev in range(len(class_names)):
    for pred_sev in range(len(class_names)):
        count = conf_matrix[true_sev, pred_sev]
        if true_sev != pred_sev and count > 0:
            severity_diff = abs(true_sev - pred_sev)
            print(f"True {class_names[true_sev]} → Pred {class_names[pred_sev]}: "
                  f"{count} cases ({severity_diff}-level error)")
#6. Gradcam visualization
print(generate_gradcam(model, test_dataset, device))
