import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Sandbox network only allows certain domains; point at a GitHub-hosted mirror
datasets.MNIST.mirrors = ["https://raw.githubusercontent.com/fgnt/mnist/master/"]

train_data = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

test_data = datasets.MNIST(root='mnist_data', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(800, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 800)
        x = self.fc1(x)
        return x

model = SimpleCNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 5
for epoch in range(epochs):
    for images, labels in train_loader:
        pred = model(images)
        loss = loss_fn(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch:{epoch}, Loss:{loss.item()}")

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        pred = model(images)
        number_predicted = torch.argmax(pred, dim=1)
        correct += (number_predicted == labels).sum()
        total += labels.size(0)
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

# Single-image inference on a real MNIST test image
image, label = test_data[0]
images = image.view(1, 1, 28, 28)
pred = model(images)
pred_image = torch.argmax(pred)
print(f'predicted digit:{pred_image},actual digit:{label}')
