from pytorch_ops import *
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt #모형 학습시 accuracy와 loss를 저장하기 위한 라이브러리입니다.

CSV_PATH = 'E:/data/multi-label_classification_FASHION/labels.csv'
IMG_DIR = 'E:/data/multi-label_classification_FASHION/imgs'
device = torch.device("cuda")

transformed_dataset = readDataset(csv_file_path=CSV_PATH,
                                  root_dir=IMG_DIR,
                                  disciriminator='_',
                                  transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()]))
dataloader = DataLoader(transformed_dataset, batch_size=16,
                        shuffle=True, num_workers=0)


model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(2048, 512),
                         nn.ReLU(),
                         nn.Dropout(0.2),
                         nn.Linear(512, 6))
criterion = nn.BCELoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
model.to(device)

TOTAL_EPOCH = 50
train_avg_loss = []
m = nn.Sigmoid()
for epoch in range(TOTAL_EPOCH):
    loss_list = []

    for i, data in enumerate(dataloader, 0):
        # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
        inputs, labels = data['image'], data['label']
        inputs = torch.where(torch.isnan(inputs), torch.zeros_like(inputs), inputs)
        inputs = inputs.type(torch.cuda.FloatTensor)
        labels = labels.type(torch.cuda.FloatTensor)
        inputs, labels = inputs.to(device), labels.to(device)

        # 변화도(Gradient) 매개변수를 0으로 만들고
        optimizer.zero_grad()

        # 순전파 + 역전파 + 최적화를 한 후
        outputs = model.forward(inputs)
        outputs = m(outputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # 통계를 출력합니다.
        loss_list.append(loss.item())
        print(f'iteration:{i}/{len(dataloader)}|loss:{round(loss.item(),3)}')

    train_avg_loss.append(round(np.mean(loss_list),3))

df = pd.DataFrame({'epoch': list(range(TOTAL_EPOCH)), 'train_accuracy': train_avg_loss},
                  columns=['epoch', 'train_loss'])
df_save_path = './loss.csv'
df.to_csv(df_save_path, index=False, encoding='euc-kr')

plt.plot(list(range(TOTAL_EPOCH)), train_avg_loss, 'b', label='Training Loss')
plt.title('Training and validation accuracy')
plt.legend()
save_path = 'loss.png'
plt.savefig(save_path)
plt.cla()