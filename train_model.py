import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_loader import MinecraftDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 18 * 13, 128), nn.ReLU(),
            nn.Linear(128, 6),
            nn.Sigmoid()  # потому что мультиметка от 0 до 1
        )

    def forward(self, x):
        return self.net(x)

dataset = MinecraftDataset("dataset/images", "dataset/actions")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss()

print("train start")

for epoch in range(10):  # потом увеличу
    total_loss = 0

    for images, actions in loader:
        images, actions = images.to(device), actions.to(device)

        pred = model(images)
        loss = criterion(pred, actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss / len(loader):.4f}")

torch.save(model.state_dict(), "minecraft_model.pth")
print("model saved")
