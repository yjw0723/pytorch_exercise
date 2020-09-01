from pytorch_ops import *
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt #모형 학습시 accuracy와 loss를 저장하기 위한 라이브러리입니다.
import torch.backends.cudnn as cudnn
import pandas as pd

CSV_PATH = 'E:/data/multi-label_classification_FASHION/labels.csv'
IMG_DIR = 'E:/data/multi-label_classification_FASHION/imgs_resized'
device = torch.device("cuda")

transformed_dataset = readDataset(csv_file_path=CSV_PATH,
                                  root_dir=IMG_DIR,
                                  disciriminator='_',
                                  transform=transforms.Compose([
                                               ToTensor()]))
columns = transformed_dataset.MLB.classes_
DATA_LENGTH = len(transformed_dataset.df['FILENAME'].tolist())
dataloader = DataLoader(transformed_dataset, batch_size=192,
                        shuffle=False, num_workers=0)

model = models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(2048, 512),
                         nn.ReLU(),
                         nn.Dropout(0.2),
                         nn.Linear(512, 6))
model.load_state_dict(torch.load('./fashion_multilabel_classification_with_WBCEloss_only_resize_100.pth'))
model.eval()
model.cuda()
m = nn.Sigmoid()
basket = []
acc = 0.
for i, data in enumerate(tqdm(dataloader)):
    # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
    inputs, labels = data['image'], data['label']
    inputs = torch.where(torch.isnan(inputs), torch.zeros_like(inputs), inputs)
    inputs = inputs.type(torch.cuda.FloatTensor)
    labels = labels.type(torch.cuda.FloatTensor)
    inputs, labels = inputs.cuda(), labels.detach().cpu().numpy().tolist()

    # 순전파 + 역전파 + 최적화를 한 후
    outputs = model(inputs)
    outputs = torch.round(m(outputs))
    outputs = outputs.detach().cpu().numpy().tolist()
    for label, output in zip(labels, outputs):
        if label == output:
            acc += 1.

print(acc/DATA_LENGTH)
