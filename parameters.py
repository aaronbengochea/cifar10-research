import torch
from models import ResNet18

# Initialize your model
model = ResNet18()

# (Optional) Load your best checkpoint to ensure the same architecture
checkpoint = torch.load("./checkpoint/ckpt.pth", map_location="cpu")
model.load_state_dict(checkpoint["net"])

# Count the parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")