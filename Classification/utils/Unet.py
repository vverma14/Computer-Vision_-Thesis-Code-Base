#!/usr/bin/env python
# coding: utf-8
import torch.nn as nn
import torch.nn.functional as F
import torch
# In[ ]:


class UNet_1024(nn.Module):
    def __init__(self):
        super(UNet_1024, self).__init__()
        # POOL
        self.pool = nn.MaxPool2d(2, stride=2)


        # ENCODERS#
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.conv5 = nn.Conv2d(128, 256, 3)
        self.conv6 = nn.Conv2d(256, 256, 3)
        self.conv7 = nn.Conv2d(256, 512, 3)
        self.conv8 = nn.Conv2d(512, 512, 3)
        self.conv9 = nn.Conv2d(512, 1024, 3)
        self.conv10 = nn.Conv2d(1024, 1024, 3)
        self.conv11 = nn.Conv2d(1024, 512, 3)
        self.conv12 = nn.Conv2d(512, 512, 3)
        self.conv13 = nn.Conv2d(512, 256, 3)
        self.conv14 = nn.Conv2d(256, 256, 3)
        self.conv15 = nn.Conv2d(256, 128, 3)
        self.conv16 = nn.Conv2d(128, 128, 3)
        self.conv17 = nn.Conv2d(128, 64, 3)
        self.conv18 = nn.Conv2d(64, 64, 3)
        self.conv19 = nn.Conv2d(64, 2, 1)


        # DECODERS#
        self.deconv1 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 2, 2)

    def forward(self, X):
        # ENCODING
        CONV1 = F.relu(self.conv1(X))
        CONV2 = F.relu(self.conv2(CONV1))
        MAXPOOLEDCONV2 = F.relu(self.pool(CONV2))
        CONV3 = F.relu(self.conv3(MAXPOOLEDCONV2))
        CONV4 = F.relu(self.conv4(CONV3))
        MAXPOOLEDCONV4 = F.relu(self.pool(CONV4))
        CONV5 = F.relu((self.conv5(MAXPOOLEDCONV4)))
        CONV6 = F.relu(self.conv6(CONV5))
        MAXPOOLEDCONV6 = F.relu(self.pool(CONV6))
        CONV7 = F.relu(self.conv7(MAXPOOLEDCONV6))
        CONV8 = F.relu(self.conv8(CONV7))
        MAXPOOLEDCONV8 = F.relu(self.pool(CONV8))
        CONV9 = F.relu(self.conv9(MAXPOOLEDCONV8))
        CONV10 = F.relu(self.conv10(CONV9))

        # DECODING
        DECONV1 = (self.deconv1(CONV10))
        ADDED1 = torch.cat([F.interpolate(CONV8, DECONV1.size()[2], mode='bilinear'), DECONV1],
                           axis=1)
        CONV11 = F.relu(self.conv11(ADDED1))
        CONV12 = F.relu(self.conv12(CONV11))
        DECONV2 = (self.deconv2(CONV12))
        ADDED2 = torch.cat([F.interpolate(CONV6, DECONV2.size()[2], mode='bilinear'), DECONV2],
                           axis=1)
        CONV13 = F.relu(self.conv13(ADDED2))
        CONV14 = F.relu(self.conv14(CONV13))
        DECONV3 = (self.deconv3(CONV14))
        ADDED3 = torch.cat([F.interpolate(CONV4, DECONV3.size()[2], mode='bilinear'), DECONV3],
                           axis=1)
        CONV15 = F.relu(self.conv15(ADDED3))
        CONV16 = F.relu(self.conv16(CONV15))
        DECONV4 = (self.deconv4(CONV16))
        ADDED4 = torch.cat([F.interpolate(CONV2, DECONV4.size()[2], mode='bilinear'), DECONV4],
                           axis=1)
        CONV17 = F.relu(self.conv17(ADDED4))
        CONV18 = F.relu(self.conv18(CONV17))
        CONV19 = (self.conv19(CONV18))

        return F.interpolate(CONV19, X.size()[2], mode='bilinear')

