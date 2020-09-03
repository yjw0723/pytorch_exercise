from pytorch_ops import *
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt #모형 학습시 accuracy와 loss를 저장하기 위한 라이브러리입니다.
import torch.backends.cudnn as cudnn

CSV_PATH = 'D:/data/multi-label_classification_FASHION/labels.csv'
IMG_DIR = 'D:/data/multi-label_classification_FASHION/imgs_resized'
SAVE_NAME = 'FASHION_WBCEloss_only_resize_10'
TOTAL_EPOCH = 10
device = torch.device("cuda")

dataset = devideDataset(csv_path = CSV_PATH, train_ratio=0.7)
train_df = dataset.train_df
test_df = dataset.test_df

train_dataset = readDataset(data=train_df,
                            img_dir=IMG_DIR,
                            disciriminator='_',
                            transform=transforms.Compose([
                                           ToTensor()]))
test_dataset = readDataset(data=test_df,
                           img_dir=IMG_DIR,
                           disciriminator='_',
                           transform=transforms.Compose([
                                           ToTensor()]))

TRAIN_DATA_LENGTH = len(train_dataset.df.iloc[:,0])
TEST_DATA_LENGTH = len(test_dataset.df.iloc[:,0])
CLASS_LENGTH = len(train_dataset.MLB.classes_)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

resnet_50 = models.resnet50(pretrained=True)

modules=list(resnet_50.children())[:-2]
resnet_50=nn.Sequential(*modules)
for p in resnet_50.parameters():
    p.requires_grad = False

class Net(nn.Module):
    def __init__(self, pretrained_model, class_length):
        super(Net, self).__init__()
        self.class_length = class_length
        self.feature = pretrained_model
        self.GAP = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc = nn.Linear(2048, class_length)

    def forward(self, x):
        print(x.size())
        x = self.feature(x)
        x = self.GAP(x)
        x = x.view(-1,2048)
        x = self.fc(x)
        x = F.sigmoid(x)
        return x

    def returnCAM(self, x):
        feature = self.feature(x)
        batch, channel, height, width = feature.size()
        print(feature.size())
        feature = self.feature(x).cpu().data.numpy()
        params = np.squeeze(list(self.parameters())[-2].cpu().data.numpy())
        output = []
        # for c1 in range(self.class_length):
        #     cam = params[i].dot(feature.reshape(channel,height*width))
        #     cam = cam.reshape(height, width)
        #     cam = cam - np.min(cam)
        #     cam_img = cam / np.max(cam)
        #     cam_img = np.uint8(255 * cam_img)
        #     output.append(cv2.resize(cam_img, (224,224)))
        # output = torch.from_numpy(np.array(output)).float().cuda()
        # print(output.size())
        # return output

model = Net(resnet_50, CLASS_LENGTH)
model.cuda()
criterion = WBCEloss
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)


train_avg_loss = []
train_avg_acc = []
test_avg_loss = []
test_avg_acc = []

for epoch in range(TOTAL_EPOCH):
    with torch.autograd.detect_anomaly():
        loss_list = []
        accuracy = 0
        for i, data in enumerate(train_dataloader):
            inputs, labels, weight_p, weight_n = data['image'], data['label'], data['weight_p'], data['weight_n']
            inputs = torch.where(torch.isnan(inputs), torch.zeros_like(inputs), inputs)
            inputs = inputs.type(torch.cuda.FloatTensor)
            labels = labels.type(torch.cuda.FloatTensor)
            weight_p = weight_p.type(torch.cuda.FloatTensor)
            weight_n = weight_n.type(torch.cuda.FloatTensor)
            inputs, labels, weight_p, weight_n = inputs.cuda(), labels.cuda(), weight_p.cuda(), weight_n.cuda()
            test = model.returnCAM(inputs)
#
#             outputs = model(inputs)
#             loss = criterion(labels, outputs, weight_p, weight_n)
#             loss.backward()
#             optimizer.step()
#
#             outputs = torch.round(outputs)
#             outputs = outputs.detach().cpu().numpy().tolist()
#             labels = labels.cpu().numpy().tolist()
#             for label, output in zip(labels, outputs):
#                 if label == output:
#                     accuracy += 1.
#
#             # 통계를 출력합니다.
#             loss_list.append(loss.item())
#             print(f'iteration:{i}/{len(train_dataloader)}|loss:{loss.item()}')
#
#         train_avg_loss.append(np.mean(loss_list))
#         train_avg_acc.append(accuracy / TRAIN_DATA_LENGTH)
#         print(f'EPOCH:{epoch}/{TOTAL_EPOCH}|Train Average Loss:{np.mean(loss_list)}|Train Accuracy:{accuracy / TRAIN_DATA_LENGTH}')
#
#         loss_list = []
#         accuracy = 0
#         print('VALIDATION START')
#         for i, data in enumerate(tqdm(test_dataloader), 0):
#             # [inputs, labels]의 목록인 data로부터 입력을 받은 후;
#             inputs, labels, weight_p, weight_n = data['image'], data['label'], data['weight_p'], data['weight_n']
#             inputs = torch.where(torch.isnan(inputs), torch.zeros_like(inputs), inputs)
#             inputs = inputs.type(torch.cuda.FloatTensor)
#             labels = labels.type(torch.cuda.FloatTensor)
#             weight_p = weight_p.type(torch.cuda.FloatTensor)
#             weight_n = weight_n.type(torch.cuda.FloatTensor)
#             inputs, labels, weight_p, weight_n = inputs.cuda(), labels.cuda(), weight_p.cuda(), weight_n.cuda()
#
#             # 순전파 + 역전파 + 최적화를 한 후
#             outputs = model(inputs)
#             loss = criterion(labels, outputs, weight_p, weight_n)
#             outputs = torch.round(outputs)
#             outputs = outputs.detach().cpu().numpy().tolist()
#             labels = labels.cpu().numpy().tolist()
#             for label, output in zip(labels, outputs):
#                 if label == output:
#                     accuracy += 1.
#
#             # 통계를 출력합니다.
#             loss_list.append(loss.item())
#
#         test_avg_loss.append(np.mean(loss_list))
#         test_avg_acc.append(accuracy / TEST_DATA_LENGTH)
#         print(f'EPOCH:{epoch}/{TOTAL_EPOCH}|Test Average Loss:{np.mean(loss_list)}|Test Accuracy:{accuracy / TEST_DATA_LENGTH}')
#
# PATH = f'./fashion_multilabel_classification_with_{SAVE_NAME}.pth'
# torch.save(model.state_dict(), PATH)
#
# df = pd.DataFrame({'epoch': list(range(TOTAL_EPOCH)), 'train_loss': train_avg_loss, 'train_acc': train_avg_acc,
#                    'test_loss': test_avg_loss, 'test_acc': test_avg_acc},
#                   columns=['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc'])
#
# df_save_path = f'./{SAVE_NAME}.csv'
# df.to_csv(df_save_path, index=False, encoding='euc-kr')
#
# plt.plot(list(range(TOTAL_EPOCH)), train_avg_loss, 'b', label='Training Loss')
# plt.plot(list(range(TOTAL_EPOCH)), test_avg_loss, 'r', label='Validation Loss')
# plt.title('Training and validation loss')
# plt.legend()
# save_path = f'LOSS_{SAVE_NAME}.png'
# plt.savefig(save_path)
# plt.cla()
#
# plt.plot(list(range(TOTAL_EPOCH)), train_avg_acc, 'b', label='Training Accuracy')
# plt.plot(list(range(TOTAL_EPOCH)), test_avg_acc, 'r', label='Validation Accuracy')
# plt.title('Training and validation accuracy')
# plt.legend()
# save_path = f'ACCURACY_{SAVE_NAME}.png'
# plt.savefig(save_path)
# plt.cla()