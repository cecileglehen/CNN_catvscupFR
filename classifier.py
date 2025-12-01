import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# --------------------------
# 1) Config
# --------------------------
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Device utilisé :", device)

# --------------------------
# 2) Transformations images
# --------------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # plus petit pour moins de mémoire
    transforms.ToTensor()
])

# --------------------------
# 3) Dataset personnalisé
# --------------------------
class ImageDataset(Dataset):
    def __init__(self, folder_dict, transform):
        self.images = []
        self.labels = []
        self.transform = transform
        for label, folder in enumerate(folder_dict):
            for file in os.listdir(folder_dict[folder]):
                if file.lower().endswith(('.png','.jpg','.jpeg')):
                    self.images.append(os.path.join(folder_dict[folder], file))
                    self.labels.append(label)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        return self.transform(img), self.labels[idx], self.images[idx]

# --------------------------
# 4) Préparer DataLoader
# --------------------------
train_folders = {"cup":"train/cup", "cat":"train/cat"}
test_folders = {"cup":"test/cup", "cat":"test/cat"}

train_dataset = ImageDataset(train_folders, transform)
test_dataset = ImageDataset(test_folders, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# --------------------------
# 5) Modèle CNN from scratch
# --------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64 -> 32x32

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 2 classes : tasse / chat
        )
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --------------------------
# 6) Entraînement
# --------------------------
epochs = 5
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels, _ in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

print("Entraînement terminé !")

# --------------------------
# 7) Évaluation
# --------------------------
model.eval()
all_preds = []
all_images = []

with torch.no_grad():
    for inputs, labels, paths in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_images.extend(paths)

# --------------------------
# 8) Affichage résultats
# --------------------------
classes = ["cup", "cat"]

for img_path, pred_label in zip(all_images, all_preds):
    folder_name = os.path.basename(os.path.dirname(img_path))
    file_name = os.path.basename(img_path)
    print(f"{folder_name}/{file_name} => {classes[pred_label]}")