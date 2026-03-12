import torch
from torch.utils.data import WeightedRandomSampler

# 🔹 Get label counts
class_counts = train_df['diagnosis'].value_counts().sort_index().values

# 🔹 Compute class weights (inverse frequency)
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)

# 🔹 Assign sample weights based on class labels
sample_weights = train_df['diagnosis'].map(lambda x: class_weights[x]).to_numpy(dtype=np.float32)

# 🔹 Convert sample_weights to a PyTorch tensor
sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

# 🔹 Create sampler
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
