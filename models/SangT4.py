# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 08:39:44 2024

@author: tuann
"""

import torch
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self, n_class=10):
        super(MyNet, self).__init__()
        self.initial_conv0 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        
        self.initial_conv = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        #self.initial_conv = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=1)
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        self.maxpool = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(384, 512, kernel_size=5, padding=1)
        self.conv5 = nn.Conv2d(512, 768, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        self.avgpool = nn.AvgPool2d(2, 2)

        #self.flat_size = 128 * 6 * 6

        self.fc_layers = nn.Sequential(
            nn.Linear(768 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, n_class)
        )
    def forward(self, x):
        x = self.relu1(self.initial_conv0(x))
        x = self.relu1(self.initial_conv(x))

        feat1 = self.relu1(self.conv1(x))

        feat2 = self.relu2(self.conv2(feat1))

        feat3 = self.relu3(self.conv3(feat2))

        combined_features = torch.cat([feat1, feat2, feat3], dim=1)

        x = self.maxpool(combined_features)
        x = self.relu4(self.conv4(x))
        x = self.avgpool(x)

        #x = x.view(-1, self.flat_size)
        x = self.avgpool(x)
        x = self.conv5(x)
        #print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)

        return x