import os
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from models import ResNet18
from PIL import Image
import numpy as np

# Set target directory for prediction results
predictions_filename = "evals/vanilla_resnet_18v02.csv"

# Set special testset path
testset_path = 'testset/cifar_test_nolabel.pkl'

# Load the best model checkpoint.
checkpoint_path = 'checkpoint/ckpt.pth'


# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Function to load a CIFAR batch from a pickle file
def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        batch = pickle.load(fo, encoding='bytes')
    return batch

# Load the test batch from the same directory
cifar10_batch = load_cifar_batch(testset_path)

# Extract images; the test data is already in (N x W x H x C) format.
# The images are stored under the key b'data'
images = cifar10_batch[b'data']

# If images is a numpy array, ensure its type is uint8 (if needed)
if isinstance(images, np.ndarray) and images.dtype != 'uint8':
    images = images.astype('uint8')

# Define the test transform: convert images to PIL images, then to tensors with normalization.
transform_test = transforms.Compose([
    transforms.ToPILImage(),  # convert numpy array (H x W x C) to PIL Image
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Create a custom Dataset for the test images.
class TestDataset(Dataset):
    def __init__(self, images, transform=None):
        """
        images: numpy array of shape (N, W, H, C)
        transform: torchvision transform to apply to each image
        """
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image, idx  # use the index as the image ID

# Create the test dataset and DataLoader.
test_dataset = TestDataset(images, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Load the best model checkpoint.
assert os.path.isfile(checkpoint_path), "Checkpoint file not found!"
checkpoint = torch.load(checkpoint_path, map_location=device)

# Initialize the model and load the state.
model = ResNet18().to(device)
model.load_state_dict(checkpoint['net'])
model.eval()  # set model to evaluation mode

# Generate predictions on the test set.
predictions = []
with torch.no_grad():
    for imgs, ids in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        _, preds = outputs.max(1)
        preds = preds.cpu().numpy()
        for image_id, pred in zip(ids.numpy(), preds):
            predictions.append({'ID': image_id, 'Label': int(pred)})

# Convert predictions to a DataFrame and save as CSV.
df_predictions = pd.DataFrame(predictions)
df_predictions.to_csv(predictions_filename, index=False)
print(f"Predictions saved to {predictions_filename}")

