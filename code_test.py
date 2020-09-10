from pytorch_ops import *
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt #모형 학습시 accuracy와 loss를 저장하기 위한 라이브러리입니다.
import torch.backends.cudnn as cudnn

CSV_PATH = 'E:/data/multi-label_classification_PLANET/labels.csv'
IMG_DIR = 'E:/data/multi-label_classification_PLANET/imgs_resized'
SAVE_NAME = 'TRADEMARK_WBCEloss_only_resize_100_lr_0_01'
TOTAL_EPOCH = 100
device = torch.device("cuda")

dataset = devideDataset(csv_path = CSV_PATH, train_ratio=0.7, disciriminator=' ')
train_df = dataset.train_df
test_df = dataset.test_df
mlb = dataset.returnMLB()
CLASS_LENGTH = len(mlb.classes_)
print(CLASS_LENGTH)
print(mlb.classes_)