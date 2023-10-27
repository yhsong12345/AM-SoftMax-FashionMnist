import torch
import torch.nn as nn
import torch.nn.functional as F
from common import *



class SphereFace(nn.Module):
    def __init__(self, num_layers=4):
        super(SphereFace, self).__init__()

        if num_layers == 4:
            n = [0, 0, 0, 0]
        elif num_layers == 10:
            n = [0, 1, 2, 1]
        elif num_layers == 20:
            n = [1, 2, 4, 1]
        elif num_layers == 36:
            n = [2, 4, 8, 2]
        elif num_layers == 64:
            n = [3, 8, 16, 3]


        filters = [1, 32, 64, 128, 512]

        self.layer = self._make_layers(filters, n, stride=1)
        self.FC1 = nn.Linear(512, 3)

        #weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                nn.init.constant_(m.bias, 0)


    
    def _make_layers(self, filters, layer, stride):
        layers = []
        for i in range(len(layer)):
            layers.append(conv(filters[i], filters[i+1], stride=stride))
            
        return nn.Sequential(*layers)
        

    def forward(self, x):
        o = self.layer(x)
        o = o.view(o.size(0), -1)
        o = self.FC1(o)
        return o
    


class AM_Softmax(nn.Module):
    def __init__(self, embedding_dim, num_classes=10, scale=30.0, margin=0.4):
        super(AM_Softmax, self).__init__()
        self.scale = scale
        self.margin = margin
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.embedding = nn.Embedding(embedding_dim, num_classes, max_norm=1)
        self.loss = nn.CrossEntropyLoss()

    
    def forward(self, x, labels):

        n, m = x.shape
        assert n == len(labels)
        assert m == self.embedding_dim
        assert torch.min(labels) >= 0
        assert torch.max(labels) < self.num_classes

        x = F.normalize(x, dim=1)
        for wf in self.embedding.parameters():
            wf = F.normalize(wf)
        w = self.embedding.weight
        cos_theta = torch.matmul(x, w)
        phi = cos_theta - self.margin

        onehot = F.one_hot(labels, self.num_classes)
        logits = self.scale * torch.where(onehot == 1, phi, cos_theta)
        err = self.loss(logits, labels)

        return logits, err