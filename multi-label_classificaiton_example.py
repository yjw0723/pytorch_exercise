from pytorch_ops import *
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt #모형 학습시 accuracy와 loss를 저장하기 위한 라이브러리입니다.
import torch.backends.cudnn as cudnn

CSV_PATH = 'E:/data/multi-label_classification_FASHION/labels.csv'
IMG_DIR = 'E:/data/multi-label_classification_FASHION/imgs_resized'
device = torch.device("cuda")

# transformed_dataset = readDataset(csv_file_path=CSV_PATH,
#                                   root_dir=IMG_DIR,
#                                   disciriminator='_',
#                                   transform=transforms.Compose([
#                                                Rescale(256),
#                                                RandomCrop(224),
#                                                ToTensor()]))
transformed_dataset = readDataset(csv_file_path=CSV_PATH,
                                  root_dir=IMG_DIR,
                                  disciriminator='_',
                                  transform=transforms.Compose([
                                               ToTensor()]))

dataloader = DataLoader(transformed_dataset, batch_size=192,
                        shuffle=True, num_workers=0)

model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(2048, 512),
                         nn.ReLU(),
                         nn.Dropout(0.2),
                         nn.Linear(512, 6))
criterion = WBCEloss
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
model.cuda()
m = nn.Sigmoid()
TOTAL_EPOCH = 100
DATA_LENGTH = len(transformed_dataset.df['FILENAME'].tolist())
train_avg_loss = []
train_avg_acc = []
cudnn.benchmark = True

for epoch in range(TOTAL_EPOCH):
    loss_list = []
    accuracy = 0
    with torch.autograd.detect_anomaly():
        for i, data in enumerate(dataloader, 0):
            # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
            inputs, labels, weight_p, weight_n = data['image'], data['label'], data['weight_p'], data['weight_n']
            inputs = torch.where(torch.isnan(inputs), torch.zeros_like(inputs), inputs)
            inputs = inputs.type(torch.cuda.FloatTensor)
            labels = labels.type(torch.cuda.FloatTensor)
            weight_p = weight_p.type(torch.cuda.FloatTensor)
            weight_n = weight_n.type(torch.cuda.FloatTensor)
            inputs, labels, weight_p, weight_n = inputs.cuda(), labels.cuda(), weight_p.cuda(), weight_n.cuda()

            # 변화도(Gradient) 매개변수를 0으로 만들고
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화를 한 후
            outputs = model(inputs)
            outputs = m(outputs)
            loss = criterion(labels, outputs, weight_p, weight_n)
            loss.backward()
            optimizer.step()

            outputs = torch.round(outputs)
            outputs = outputs.detach().cpu().numpy().tolist()
            labels = labels.cpu().numpy().tolist()
            for label, output in zip(labels, outputs):
                if label == output:
                    accuracy += 1.

            # 통계를 출력합니다.
            loss_list.append(loss.item())
            print(f'iteration:{i}/{len(dataloader)}|loss:{loss.item()}')

        train_avg_loss.append(np.mean(loss_list))
        train_avg_acc.append(accuracy / DATA_LENGTH)
        print(f'EPOCH:{epoch}/{TOTAL_EPOCH}|Average Loss:{np.mean(loss_list)}|Accuracy:{accuracy/DATA_LENGTH}')

PATH = './fashion_multilabel_classification_with_WBCEloss_only_resize_100.pth'
torch.save(model.state_dict(), PATH)

df = pd.DataFrame({'epoch': list(range(TOTAL_EPOCH)), 'train_accuracy': train_avg_loss},
                  columns=['epoch', 'train_loss'])
df_save_path = './WBCEloss_only_resize_100.csv'
df.to_csv(df_save_path, index=False, encoding='euc-kr')

plt.plot(list(range(TOTAL_EPOCH)), train_avg_loss, 'b', label='Training Loss')
plt.title('Training and validation accuracy')
plt.legend()
save_path = 'LOSS_WBCEloss_only_resize_100.png'
plt.savefig(save_path)
plt.cla()

plt.plot(list(range(TOTAL_EPOCH)), train_avg_acc, 'b', label='Training Accuracy')
plt.title('Training and validation accuracy')
plt.legend()
save_path = 'ACCURACY_WBCEloss_only_resize_100.png'
plt.savefig(save_path)
plt.cla()