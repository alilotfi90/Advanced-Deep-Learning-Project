from zipfile import ZipFile
file_name='/content/drive/MyDrive/images/train.zip'
with ZipFile(file_name,'r') as zip:
  zip.extractall()
  print('done')

from zipfile import ZipFile
file_name='/content/drive/MyDrive/images/test.zip'
with ZipFile(file_name,'r') as zip:
  zip.extractall()
  print('done')

#!pip install efficientnet_pytorch
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
import torch.optim as optim
import torch.nn as nn

train_path = "/content/train"
test_path = "/content/test"

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_data = ImageFolder(train_path, transform=transform)
test_data = ImageFolder(test_path, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=100)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

max_epoch=[10,20,30]
for e in max_epoch:
  for epoch in range(e):
      print(epoch)
      running_loss = 0.0
      for i, data in enumerate(train_loader, 0):
          inputs, labels = data
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
          if i % 100 == 99:
              print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
              running_loss = 0.0

  correct = 0
  total = 0

  with torch.no_grad():
      for data in test_loader:
          images, labels = data
          outputs = model(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
  print("for number of epochs being:",e)
  print('Accuracy of the network on the test images: %d %%' % (
      100 * correct / total))

torch.save(model.state_dict(), "/content/drive/MyDrive/images/efficient30")

model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=100)

# load the weights from a file
state_dict = torch.load('/content/drive/MyDrive/images/efficient30')

# set the model's parameters to the loaded weights
model.load_state_dict(state_dict)


for epoch in range(10):
      print(epoch)
      running_loss = 0.0
      for i, data in enumerate(train_loader, 0):
          inputs, labels = data
          optimizer.zero_grad()
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
          if i % 100 == 99:
              print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
              running_loss = 0.0

correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print("for number of epochs being:",e)
print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))