import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from ssim_loss import ssim
from ssim_loss import ms_ssim

from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure


class SSIM_L1(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1=nn.L1Loss()
        self.ms_ssim=MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self,pred,actual):
        l1=self.l1(pred,actual)
        _ssim=ms_ssim(pred,actual)

        #return (l1+_ssim)/2
        return _ssim

class SplitLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        #return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

        parts=torch.split(torch.abs(pred-actual),4,3)

        sum=0
        for tensor in parts:
            sum+=torch.mean(tensor)

        return sum
        #return torch.sqrt(self.mse(torch.log(F.relu(pred)+1),torch.log(F.relu(actual)+1)))

class UBDmDN(torch.nn.Module):
    def __init__(self,batch_size=4,img_size=(180,180)):
        super(UBDmDN,self).__init__()

        self.conv=nn.ModuleList()
        self.conv.append(nn.Conv2d(1,64,3,1,padding=1))

        self.conv.extend([nn.Conv2d(64,64,11,1,padding=5) for _ in range(3)])
        self.conv.extend([nn.Conv2d(64,64,3,1,1) for _ in range(3)])

        self.conv.append(nn.Conv2d(64,1,5,1,padding=2))

        self.conv2=nn.ModuleList()
        self.conv2.append(nn.Conv2d(1,64,3,1,padding=1))

        self.conv2.extend([nn.Conv2d(64,64,11,1,padding=5) for _ in range(3)])
        self.conv2.extend([nn.Conv2d(64,64,3,1,1) for _ in range(3)])

        self.conv2.append(nn.Conv2d(64,1,5,1,padding=2))

        self.to_read=0
        self.buf=[[torch.zeros(batch_size,64,img_size[0],img_size[1]).to('cuda') for _ in range(6)] for _ in range(2)]

        for i in range(len(self.conv[:-1])):
            nn.init.kaiming_normal_(self.conv[i].weight.data,nonlinearity='relu')

        for i in range(len(self.conv2[:-1])):
            nn.init.kaiming_normal_(self.conv2[i].weight.data, nonlinearity='relu')

    def forward(self,x):

        h=F.relu(self.conv[0](x))

        for i in range(1,4):
            h_1=F.relu(self.conv[i](h))
            h_2=F.relu(self.conv[i+3](h))
            h=h_1+h_2

        h=x-self.conv[7](h)

        #x=h
        #h=F.relu(self.conv2[0](x))

        #for i in range(1,4):
        #    h_1=F.relu(self.conv2[i](h))
        #    h_2=F.relu(self.conv2[i+3](h))
        #    h=h_1+h_2

        #h=x-self.conv2[7](h)

        return h
