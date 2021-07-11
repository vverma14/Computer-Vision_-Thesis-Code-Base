# pytorch
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
# classic
import numpy as np

def kernelGauss2D(m=2,sigma=1):
    x = np.arange(-m,m+.1); X, Y = np.meshgrid(x,x)
    g = np.exp(- (X**2+Y**2) / ( 2*sigma**2 ) )
    g /= np.sum(g)
    return g

class SimpleLm(nn.Module):

    def __init__(self):
        super().__init__()
        self.lm0 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=1, bias=True)
        
    def forward(self, X):

        return self.lm0(X)


class SimpleNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.kernel33 = 1/4*torch.tensor([[0., 1., 0.],
                                         [1., 0., 1.],
                                         [0., 1., 0.]])
        self.kernelGauss = torch.tensor( kernelGauss2D(m=3,sigma=3) , dtype = torch.float)
        self.lm0 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=1, bias=True)
        self.lm1 = nn.Conv2d(1,2,1, bias=False)
        self.lm2 = nn.Conv2d(1,2,1, bias=False)
        self.lm3 = nn.Conv2d(1,2,1, bias=False)
        self.lm4 = nn.Conv2d(1,2,1, bias=False)
        self.lm5 = nn.Conv2d(1,2,1, bias=False)
        self.lm6 = nn.Conv2d(1,2,1, bias=False)
        
    def forward(self, X):
        X1 = F.conv2d(X, self.kernel33.view(1,1,3,3), padding=1)
        X2 = F.conv2d(X, self.kernelGauss.view(1,1,7,7), padding=3)
        # or with constant padding
        # X1 = F.conv2d( F.pad(X, pad=(1,1,1,1)), self.kernel33.view(1,1,3,3))
        # X2 = F.conv2d( F.pad(X, pad=(3,3,3,3)), self.kernelGauss2D.view(1,1,7,7))
        X3 = -F.max_pool2d(-X,kernel_size=(3,3),  padding=1,stride=1)
        X4 = -F.max_pool2d(-X,kernel_size=(11,11),padding=5,stride=1)
        X5 =  F.max_pool2d( X,kernel_size=(3,3),  padding=1,stride=1)
        X6 =  F.max_pool2d( X,kernel_size=(11,11),padding=5,stride=1)

        return self.lm0(X) + self.lm1(X1) + self.lm2(X2) + self.lm3(X3) + self.lm4(X4) + self.lm5(X5) + self.lm6(X6)


class SimpleConvNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3, bias=True, padding=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=2, kernel_size=1, bias=True)
        
    def forward(self, X):
        Z = self.conv1(X)
        S = self.conv2(Z)

        return S

    
class SimpleConvReLuNet(nn.Module):

    def __init__(self, nb_channels, kernel_size):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=nb_channels,
                               kernel_size=kernel_size, bias=True)
        self.conv2 = nn.Conv2d(in_channels=nb_channels, out_channels=2,
                               kernel_size=kernel_size, bias=True)
        self.padsize = int( (kernel_size-1) )
        
    def forward(self, X):
        Z = self.conv1(F.pad(X, pad=(self.padsize,self.padsize,self.padsize,self.padsize)))
        S = self.conv2(F.relu(Z))

        return S


class ConvNetPool(nn.Module):

    def __init__(self, nb_channels, kernel_size):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=3, out_channels=nb_channels,
                               kernel_size=kernel_size, bias=True)
        self.pool = nn.MaxPool2d(2)
        # self.convT = nn.Conv2d(in_channels=nb_channels, out_channels=2,
        #                        kernel_size=kernel_size, stride=2)
        # self.convT = nn.ConvTranspose2d(in_channels=nb_channels, out_channels=2,
        #                                 kernel_size=kernel_size, stride=2)
        self.convT = nn.ConvTranspose2d(in_channels=nb_channels, out_channels=2,
                                        kernel_size=kernel_size, stride=1)
        
    def forward(self, X):

        Z1 = F.relu( self.conv(X) )
        Z2 = self.pool(Z1)
        Z3 = F.relu( self.convT(Z2) )
        S  = F.interpolate(Z3, X.size()[2:], mode='bilinear', align_corners=None)

        return S


class ConvNetPool2(nn.Module):

    def __init__(self, nb_channels, kernel_size):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=nb_channels[0],
                               kernel_size=kernel_size[0], bias=True)
        self.conv2 = nn.Conv2d(in_channels=nb_channels[0], out_channels=nb_channels[1],
                               kernel_size=kernel_size[1], bias=True)

        self.pool = nn.MaxPool2d(2)
        self.convT = nn.ConvTranspose2d(in_channels=nb_channels[1], out_channels=2,
                                        kernel_size=1, stride=1)
        
    def forward(self, X):

        Z1 = F.relu( self.conv1(X) )
        Z2 = self.pool(Z1)
        Z3 = F.relu( self.conv2(Z2) )
        Z4 = self.pool(Z3)
        Z5 = F.relu( self.convT(Z4) )
        S  = F.interpolate(Z5, X.size()[2:], mode='bilinear', align_corners=None)

        return S
    
    
class ConvUnet(nn.Module):

    def __init__(self, nb_channels, kernel_size):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=3, out_channels=nb_channels,
                               kernel_size=kernel_size, bias=True)
        self.pool = nn.MaxPool2d(2)
        self.convT = nn.Conv2d(in_channels=nb_channels, out_channels=nb_channels,
                               kernel_size=kernel_size, stride=1)
        self.lm0 = nn.Conv2d(in_channels=2*nb_channels, out_channels=2, kernel_size=1, bias=True)
        
    def forward(self, X):
        
        Z1 = F.relu( self.conv(X) )
        Z1_bis = F.interpolate(Z1, X.size()[2:], mode='bilinear', align_corners=None)
        Z2 = self.pool(Z1)
        Z3 = F.relu( self.convT(Z2) )
        Z3_bis = F.interpolate(Z3, X.size()[2:], mode='bilinear', align_corners=None)
        S = self.lm0( torch.cat([Z1_bis,Z3_bis], dim=1) )
        
        return S
    

class ConvUnetMix(nn.Module):

    def __init__(self, nb_channels, kernel_size):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=3, out_channels=nb_channels,
                               kernel_size=kernel_size, bias=True)
        self.conv_mix = nn.Conv2d(in_channels=3, out_channels=2,
                              kernel_size=kernel_size, bias=True)
        self.pool = nn.MaxPool2d(2)
        self.convT = nn.Conv2d(in_channels=nb_channels, out_channels=2,
                               kernel_size=kernel_size, stride=1)
        self.lm0 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=1, bias=True)
        
    def forward(self, X):
        
        Z1 = F.relu( self.conv(X) )
        Z1_mix = F.relu( self.conv_mix(X) )
        Z1_mix_bis = F.interpolate(Z1_mix, X.size()[2:], mode='bilinear', align_corners=None)
        Z2 = self.pool(Z1)
        Z3 = F.relu( self.convT(Z2) )
        Z3_bis = F.interpolate(Z3, X.size()[2:], mode='bilinear', align_corners=None)
        S = self.lm0( torch.cat([Z1_mix_bis,Z3_bis], dim=1) )
        
        return S
    
    
