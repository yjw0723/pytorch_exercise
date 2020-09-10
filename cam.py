from pytorch_ops import *
from nets import *
from data_loader import *
from train_validation import *
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim


CSV_PATH = 'D:/data/multi-label_classification_FASHION/labels.csv'
IMG_DIR = 'D:/data/multi-label_classification_FASHION/imgs_resized'
SAVE_NAME = 'FASHION_GlobalNet'
TOTAL_EPOCH = 100
BATCH_SIZE = 128
TRAIN_RATIO = 0.7
LR = 0.0005
DISCRIMINATOR = '_'
device = torch.device("cuda")

dataset = devideDataset(csv_path = CSV_PATH, train_ratio=TRAIN_RATIO)
train_df = dataset.train_df
test_df = dataset.test_df

train_dataset = readDataset(data=train_df,
                            img_dir=IMG_DIR,
                            disciriminator=DISCRIMINATOR,
                            transform=transforms.Compose([
                                           ToTensor()]))
test_dataset = readDataset(data=test_df,
                           img_dir=IMG_DIR,
                           disciriminator=DISCRIMINATOR,
                           transform=transforms.Compose([
                                           ToTensor()]))

TRAIN_DATA_LENGTH = len(train_dataset.df.iloc[:,0])
TEST_DATA_LENGTH = len(test_dataset.df.iloc[:,0])
CLASS_LENGTH = len(train_dataset.MLB.classes_)

resnet_50 = models.resnet50(pretrained=True)

modules=list(resnet_50.children())[:-2]
resnet_50=nn.Sequential(*modules)
for p in resnet_50.parameters():
    p.requires_grad = False

model = GlobalNet(resnet_50, CLASS_LENGTH)
model.cuda()
criterion = WBCEloss
optimizer = optim.Adam(model.parameters(), lr=LR)

train_ = returnLossAndAcc(dataset=train_dataset,
                          batch_size=128)
val_ = returnLossAndAcc(dataset=test_dataset,
                        batch_size=128,
                        shuffle=False)

save = Save(save_name=SAVE_NAME,
            total_epoch=TOTAL_EPOCH,
            train_avg_loss=train_.AVG_LOSS,
            train_avg_acc=train_.AVG_ACC,
            val_avg_loss=val_.AVG_LOSS,
            val_avg_acc=val_.AVG_ACC)

train_and_val = TrainAndValidation(save=save,
                                   train_=train_,
                                   val_=val_,
                                   model=model,
                                   criterion=criterion,
                                   optimizer=optimizer,
                                   total_epoch=TOTAL_EPOCH)
train_and_val.execute()
