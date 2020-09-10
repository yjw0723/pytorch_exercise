from __future__ import print_function, division
import os
import pandas as pd
from skimage import transform
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class devideDataset:
    def __init__(self, csv_path, train_ratio):
        """
        Args:
            :param csv_path: label정보가 있는 csv 파일의 경로(string)
            :param train_ratio: 전체 데이터 셋 중에서 학습 데이터가 차지할 비중(float(ex:0.7))
        """
        self.df = pd.read_csv(csv_path, engine='python')
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.train_ratio = train_ratio
        self.devide()

    def devide(self):
        data_length = len(self.df.iloc[:,0])
        train_length = int(data_length * self.train_ratio)
        self.train_df = self.df.iloc[:train_length,:]
        self.test_df = self.df.iloc[train_length:,:]

class returnMLB():
    def __init__(self, csv_path, discriminator):
        self.df = pd.read_csv(csv_path)
        self.discriminator = discriminator

    def returnMLB(self):
        label_list = self.df.iloc[:, 1].tolist()
        label_list = np.unique(np.array(label_list))
        multilabel_list = [label.split(self.discriminator) for label in label_list]
        multilabel_list = [tuple(label) for label in multilabel_list]
        mlb = MultiLabelBinarizer()
        mlb.fit(multilabel_list)
        return mlb

class readDataset(Dataset):
    def __init__(self, data, img_dir, mlb, discriminator, transform=None):
        """
        Args:
            data (string): pandas dataframe
            img_dir (string): 모든 이미지가 존재하는 디렉토리 경로
            disciriminator: label의 구분자(만약 하나의 라벨이 'black_jeans'라면 discriminator는 '_'를 의미함)
            transform (callable, optional): 샘플에 적용될 Optional transform
        """
        self.df = data
        self.root_dir = img_dir
        self.transform = transform
        self.discriminator = discriminator
        self.MLB = mlb
        self.updateDf()


    def updateDf(self):
        label_list = self.df.iloc[:, 1].tolist()
        label_list = [tuple(label.split(self.discriminator)) for label in label_list]
        self.df.iloc[:,1] = label_list
        onehot_list = []
        P = self.returnP()
        W_P = []
        W_N = []
        for label in label_list:
            onehot = np.ndarray.flatten(self.MLB.transform([label]))
            onehot_list.append(onehot)
            w_p = []
            for i, p in zip(onehot,P):
                if i == 1:
                    w_p.append(np.exp(1-p))
                else:
                    w_p.append(0.)
            W_P.append(np.array(w_p))
            w_n = []
            for i, p in zip(onehot,P):
                if i == 0:
                    w_n.append(np.exp(p))
                else:
                    w_n.append(0.)
            W_N.append(np.array(w_n))
        self.df['onehot'] = onehot_list
        self.df['weight_p'] = W_P
        self.df['weight_n'] = W_N
        self.df.columns = ['FILENAME', 'LABEL', 'onehot', 'weight_p', 'weight_n']

    def returnP(self):
        label_list = self.df.iloc[:, 1].tolist()
        P = np.zeros(len(np.unique(np.array(label_list))))
        for i, label in enumerate(label_list):
            label = [self.df.iloc[i,1]]
            result = np.ndarray.flatten(self.MLB.transform(label))
            P = P+result
        return P / np.sum(P)

    def returnOneHotMultiLabel(self, idx):
        label = [self.df.iloc[idx,1]]
        result = np.ndarray.flatten(self.MLB.transform(label))
        result = np.array(result, dtype='float').flatten()
        return result

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.df.iloc[idx, 0])
        image = cv2.imread(img_name)
        image = np.array(image, dtype='float') / 255.0 #0.~1. 사이의 값으로 Normalizing
        # label = self.returnOneHotMultiLabel(idx)
        label = self.df.iloc[idx, 2]
        weight_p = self.df.iloc[idx,3]
        weight_n = self.df.iloc[idx,4]

        sample = {'image': image, 'label': label, 'weight_p':weight_p, 'weight_n':weight_n}

        if self.transform:
            sample = self.transform(sample)

        return sample

class Rescale(object):
    """주어진 사이즈로 샘플크기를 조정합니다.

    Args:
        output_size(tuple or int) : 원하는 사이즈 값
            tuple인 경우 해당 tuple(output_size)이 결과물(output)의 크기가 되고,
            int라면 비율을 유지하면서, 길이가 작은 쪽이 output_size가 됩니다.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label, weight_p, weight_n = sample['image'], sample['label'], sample['weight_p'], sample['weight_n']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'label': label, 'weight_p':weight_p, 'weight_n':weight_n}

class Resize(object):
    """주어진 사이즈로 샘플크기를 조정합니다.

    Args:
        output_size(tuple or int) : 원하는 사이즈 값
            tuple인 경우 해당 tuple(output_size)이 결과물(output)의 크기가 되고,
            int라면 비율을 유지하면서, 길이가 작은 쪽이 output_size가 됩니다.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label, weight_p, weight_n = sample['image'], sample['label'], sample['weight_p'], sample['weight_n']
        img = transform.resize(image, (self.output_size, self.output_size))

        return {'image': img, 'label': label, 'weight_p':weight_p, 'weight_n':weight_n}


class RandomCrop(object):
    """샘플데이터를 무작위로 자릅니다.

    Args:
        output_size (tuple or int): 줄이고자 하는 크기입니다.
                        int라면, 정사각형으로 나올 것 입니다.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label, weight_p, weight_n = sample['image'], sample['label'], sample['weight_p'], sample['weight_n']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'label': label, 'weight_p':weight_p, 'weight_n':weight_n}


class ToTensor(object):
    """numpy array를 tensor(torch)로 변환 시켜줍니다."""

    def __call__(self, sample):
        image, label, weight_p, weight_n = sample['image'], sample['label'], sample['weight_p'], sample['weight_n']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': image, 'label': label, 'weight_p':weight_p, 'weight_n':weight_n}


class returnDataLoader:
    def __init__(self, csv_path, train_ratio, img_dir, discriminator):
        self.DATASET = devideDataset(csv_path, train_ratio)
        self.MLB = returnMLB(csv_path, discriminator).returnMLB()
        self.TRAIN_DATASET = readDataset(self.DATASET.train_df,
                                         img_dir,
                                         self.MLB,
                                         discriminator,
                                         transforms.Compose([ToTensor()]))
        self.VAL_DATASET = readDataset(self.DATASET.test_df,
                                       img_dir,
                                       self.MLB,
                                       discriminator,
                                       transforms.Compose([ToTensor()]))
        self.TRAIN_DATA_LENGTH = len(self.DATASET.train_df.iloc[:,0])
        self.VAL_DATA_LENGTH = len(self.DATASET.test_df.iloc[:,0])