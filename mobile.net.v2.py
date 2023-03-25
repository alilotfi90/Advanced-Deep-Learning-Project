from zipfile import ZipFile
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet

# Extract the training and test data
train_zip = '/content/drive/MyDrive/images/train.zip'
test_zip = '/content/drive/MyDrive/images/test.zip'
with ZipFile(train_zip, 'r') as zip:
    zip.extractall()
with ZipFile(test_zip, 'r') as zip:
    zip.extractall()

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# Set up the dataset and data loaders
train_path = "/content/train"
test_path = "/content/test"

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_data = datasets.ImageFolder(train_path, transform=transform)
test_data = datasets.ImageFolder(test_path, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

# Load MobileNetV2 model
mobilenet = models.mobilenet_v2(pretrained=True)

# Replace the last fully connected layer with our own
num_ftrs = mobilenet.classifier[-1].in_features
mobilenet.classifier[-1] = nn.Linear(num_ftrs, 100)

optimizer = optim.Adam(mobilenet.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

max_epoch = [10, 20, 40]
for e in max_epoch:
    # Train the model
    for epoch in range(e):
        print(epoch)
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = mobilenet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
    # Test the model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = mobilenet(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print("for number of epochs being:", e)
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
