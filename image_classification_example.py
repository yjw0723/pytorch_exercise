import torch
from torchvision import datasets, transforms, models
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

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

net = models.resnet50(pretrained=True)
for param in net.parameters():
    param.requires_grad = False

net.fc = nn.Sequential(nn.Linear(2048, 512),
                       nn.ReLU(),
                       nn.Dropout(0.2),
                       nn.Linear(512, 4),
                       nn.Softmax())

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)
net.to(device)

TOTAL_EPOCH = 10
train_avg_loss = []
test_avg_loss = []
train_accuracy_list = []
test_accuracy_list = []

for epoch in range(10):   # 데이터셋을 수차례 반복합니다.
    train_loss_list = []
    test_loss_list = []

    total = 0
    correct = 0
    print(f'START TRAINING EPOCH:{epoch}')
    for i, data in enumerate(tqdm(trainloader), 0):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = net.forward(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 통계를 출력합니다.
        train_loss_list.append(loss.item())
    train_accuracy = 100*correct/total
    train_accuracy_list.append(train_accuracy)
    train_avg_loss.append(round(np.mean(train_loss_list), 3))

    print(f'START VALIDATION EPOCH:{epoch}')
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs =net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            test_loss_list.append(criterion(outputs, labels).item())
        test_accuracy = 100 * correct / total
        test_accuracy_list.append(test_accuracy)
        test_avg_loss.append(round(np.mean(test_loss_list), 3))
    print(f'EPOCH:{epoch}/{TOTAL_EPOCH}'
          f'|TRAIN LOSS(AVG):{round(np.mean(train_loss_list),3)}'
          f'|TRAIN ACCURACY:{train_accuracy}'
          f'|VALIDATION LOSS(AVG):{round(np.mean(test_loss_list),3)}'
          f'|VALIDATION ACCURACY:{test_accuracy}')

print('Finished Training')
torch.save(net, 'redness.pth')

df = pd.DataFrame({'epoch': list(range(TOTAL_EPOCH)),
                   'train_loss': train_avg_loss,
                   'train_accuracy': train_accuracy_list,
                   'test_loss':test_avg_loss,
                   'test_accuracy':test_accuracy_list},
                  columns=['epoch', 'train_loss', 'train_accuracy', 'test_loss', 'test_accuracy'])
df_save_path = './redness_loss.csv'
df.to_csv(df_save_path, index=False, encoding='euc-kr')

plt.plot(list(range(TOTAL_EPOCH)), train_avg_loss, 'b', label='Training Loss')
plt.plot(list(range(TOTAL_EPOCH)), test_avg_loss, 'r', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
save_path = 'redness_loss.png'
plt.savefig(save_path)
plt.cla()

plt.plot(list(range(TOTAL_EPOCH)), train_accuracy_list, 'b', label='Training Accuracy')
plt.plot(list(range(TOTAL_EPOCH)), test_accuracy_list, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.legend()
save_path = 'redness_accuracy.png'
plt.savefig(save_path)
plt.cla()