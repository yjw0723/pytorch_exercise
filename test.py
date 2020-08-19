import torch
import torchvision
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

train_dir = './redness/train'
test_dir = './redness/test'
device = torch.device("cuda")

data_transforms = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
train_data = datasets.ImageFolder(train_dir, transform=data_transforms)
test_data = datasets.ImageFolder(test_dir, transform=data_transforms)
train_idxs = list(range(len(train_data)))
test_idxs = list(range(len(test_data)))
np.random.shuffle(train_idxs)
np.random.shuffle(test_idxs)
train_sampler = SubsetRandomSampler(train_idxs)
test_sampler = SubsetRandomSampler(test_idxs)

trainloader = torch.utils.data.DataLoader(train_data,
                                          sampler=train_sampler, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data,
                                         sampler=test_sampler, batch_size=32)

model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(2048, 512),
                         nn.ReLU(),
                         nn.Dropout(0.2),
                         nn.Linear(512, 4))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
model.to(device)

print(len(trainloader))
for epoch in range(10):   # 데이터셋을 수차례 반복합니다.
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 통계를 출력합니다.
        running_loss += loss.item()
        # print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
        if i % 100 == 99:

            running_loss = 0.0

print('Finished Training')
torch.save(model, 'aerialmodel.pth')