import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import numpy as np
import cv2

def ResNet50():
    resnet_50 = models.resnet50(pretrained=True)
    modules=list(resnet_50.children())[:-2]
    resnet_50=nn.Sequential(*modules)
    for p in resnet_50.parameters():
        p.requires_grad = False
    return resnet_50

class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)

class CAMNet(nn.Module):
    def __init__(self, class_length):
        super(CAMNet, self).__init__()
        self.class_length = class_length
        self.feature = ResNet50()
        self.dropout = nn.Dropout
        self.GAP = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.fc_1 = nn.Linear(2048, 512)
        self.fc_2 = nn.Linear(512, class_length)

    def forward(self, x):
        x = self.feature(x)
        x = self.GAP(x)
        x = x.view(-1,2048)
        x = self.fc_1(x)
        x = F.relu(x)
        x = F.dropout(input=x, p=0.2, training=self.training)
        x = self.fc_2(x)
        x = F.sigmoid(x)
        return x

    def returnCAM(self, x):
        features = self.feature(x)
        params = np.squeeze(list(self.fc_2.parameters())[0].cpu().data.numpy())
        batch, channel, height, width = features.size()
        features = self.feature(x).cpu().data.numpy()
        f_output = []
        for b in range(batch):
            feature = features[b].reshape(channel, height*width)
            output = []
            for c1 in range(self.class_length):
                cam = params[c1].dot(feature)
                cam = cam.reshape(height, width)
                cam = cam - np.min(cam)
                cam_img = cam / np.max(cam)
                cam_img = np.uint8(255 * cam_img)
                if c1 == 0:
                    output = cv2.resize(cam_img, (224, 224))
                else:
                    output = output + cv2.resize(cam_img, (224, 224))
            output = output - np.min(output)
            output = output / np.max(output)
            output = np.expand_dims(output, axis=0)
            f_output.append(output)

        return f_output

    def returnHeatMapImg(self, x):
        cams = self.returnCAM(x)
        imgs = x
        batch, channel, height, width = imgs.size()
        imgs = x.cpu().data.numpy()
        output = []
        for b in range(batch):
            heatmap = cams[b]
            img = imgs[b]
            heatmap_img = heatmap * img
            output.append(heatmap_img)
        return torch.from_numpy(np.array(output)).float().cuda()

class GlobalNet(nn.Module):
    def __init__(self, class_length):
        super(GlobalNet, self).__init__()
        self.class_length = class_length
        self.conv_layer = nn.Sequential(ResNet50(),
                                        nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1))
        self.flatten_layer = nn.Sequential(nn.ReLU(),
                                           nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                           Flatten())
        self.fc_layer = nn.Sequential(nn.Linear(2048,512),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(512,self.class_length),
                                      nn.Sigmoid())

    def forward(self, x):
        f = self.conv_layer(x)
        p = self.flatten_layer(f)
        o = self.fc_layer(p)
        return o, p, f

    def gOutput(self, x):
        outputs, poolings, features = self.forward(x)
        batch, _, height, width = x.size()
        attention = torch.max(features, 1).values.cpu().data.numpy()
        attentions = []
        for b in range(batch):
            output = cv2.resize(attention[b], (height,width))
            output = output - np.min(output)
            attentions.append(output / np.max(output))
        attentions = torch.from_numpy(np.array(attentions)).float().cuda()
        attentions = torch.reshape(attentions, (x.size()[0], 1, x.size()[2], x.size()[3]))
        return outputs, poolings, attentions * x

class LocalNet(nn.Module):
    def __init__(self, class_length):
        super(LocalNet, self).__init__()
        self.class_length = class_length
        self.conv_layer = nn.Sequential(ResNet50(),
                                        nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=1, padding=1))
        self.flatten_layer = nn.Sequential(nn.ReLU(),
                                           nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                                           Flatten())
        self.fc_layer = nn.Sequential(nn.Linear(2048,512),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(512,self.class_length),
                                      nn.Sigmoid())

    def forward(self, x):
        f = self.conv_layer(x)
        p = self.flatten_layer(f)
        o = self.fc_layer(p)
        return o, p

class FusionNet(nn.Module):
    def __init__(self, class_length):
        super(FusionNet, self).__init__()
        self.class_length = class_length
        self.fc_layer = nn.Sequential(nn.Linear(4096,2048),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(2048,512),
                                      nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(512,class_length),
                                      nn.Sigmoid())

    def forward(self, x):
        output = self.fc_layer(x)
        return output