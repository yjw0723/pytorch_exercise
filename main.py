from pytorch_ops import *
from data_loader import *
from train_validation import *
import warnings
warnings.filterwarnings("ignore")

CSV_PATH = 'E:/data/viennacode_img_19600101_20191231_unique_preprocessed/labels_middle.csv'
IMG_DIR = 'E:/data/viennacode_img_19600101_20191231_unique_preprocessed/imgs'
SAVE_NAME = 'TRADEMARK_AG_CNN'
TOTAL_EPOCH = 100
BATCH_SIZE = 128
TRAIN_RATIO = 0.7
LR = 0.0005
DISCRIMINATOR = '|'

device = torch.device("cuda")

dataloader = returnDataLoader(csv_path=CSV_PATH,
                              train_ratio=TRAIN_RATIO,
                              img_dir=IMG_DIR,
                              discriminator=DISCRIMINATOR)

train_ = returnLossAndAcc(dataset=dataloader.TRAIN_DATASET,
                          batch_size=BATCH_SIZE)
val_ = returnLossAndAcc(dataset=dataloader.VAL_DATASET,
                        batch_size=BATCH_SIZE,
                        shuffle=False)

save = Save(save_name=SAVE_NAME,
            total_epoch=TOTAL_EPOCH,
            train_=train_,
            val_=val_)

train_and_val = TrainAndValidation(save=save,
                                   train_=train_,
                                   val_=val_,
                                   class_length=len(dataloader.MLB.classes_),
                                   learning_rate=LR,
                                   total_epoch=TOTAL_EPOCH)
train_and_val.execute()
