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
                                          sampler=train_sampler, batch_size=1)
testloader = torch.utils.data.DataLoader(test_data,
                                         sampler=test_sampler, batch_size=1)

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

TOTAL_EPOCH = 10
train_avg_loss = []
test_avg_loss = []
for epoch in range(10):   # 데이터셋을 수차례 반복합니다.
    train_loss_list = []
    test_loss_list = []
    for i, data in enumerate(trainloader, 0):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        print('outputs:', outputs.type(), outputs)
        print('labels:',labels.type(), labels)
        loss.backward()
        optimizer.step()

        # 통계를 출력합니다.
        train_loss_list.append(loss.item())
        print(f'iteration:{i}/{len(trainloader)}|loss:{round(loss.item(),3)}')

    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        test_loss_list.append(criterion(logps, labels).item())
    print(f'EPOCH:{epoch}/{TOTAL_EPOCH}|TRAIN LOSS(AVG):{round(np.mean(train_loss_list),3)}|VALIDATION LOSS(AVG):{round(np.mean(test_loss_list),3)}')

print('Finished Training')
torch.save(model, 'redness.pth')