from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
from tqdm import tqdm
import shutil

# 경고 메시지 무시하기
import warnings
warnings.filterwarnings("ignore")

class readDataset(Dataset):
    """dataset."""

    def __init__(self, csv_file_path, root_dir, disciriminator, transform=None):
        """
        Args:
            csv_file_path (string): csv 파일의 경로
            root_dir (string): 모든 이미지가 존재하는 디렉토리 경로
            disciriminator: label의 구분자(만약 하나의 라벨이 'black_jeans'라면 discriminator는 '_'를 의미함)
            transform (callable, optional): 샘플에 적용될 Optional transform
        """
        self.df = pd.read_csv(csv_file_path)
        self.root_dir = root_dir
        self.transform = transform
        self.discriminator = disciriminator
        self.MLB = self.returnMLB()
        self.updateDf()

    def updateDf(self):
        label_list = self.df.iloc[:, 1].tolist()
        label_list = [tuple(label.split(self.discriminator)) for label in label_list]
        self.df.iloc[:,1] = label_list

    def returnMLB(self):
        label_list = self.df.iloc[:, 1].tolist()
        label_list = np.unique(np.array(label_list))
        multilabel_list = [label.split(self.discriminator) for label in label_list]
        multilabel_list = [tuple(label) for label in multilabel_list]
        mlb = MultiLabelBinarizer()
        mlb.fit(multilabel_list)
        return mlb

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
        image = io.imread(img_name)
        # image = np.array(image, dtype='float') / 255.0 #0.~1. 사이의 값으로 Normalizing
        label = self.returnOneHotMultiLabel(idx)

        sample = {'image': image, 'label': label}

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
        image, label = sample['image'], sample['label']

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

        return {'image': img, 'label': label}


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
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'label': label}


class ToTensor(object):
    """numpy array를 tensor(torch)로 변환 시켜줍니다."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}

