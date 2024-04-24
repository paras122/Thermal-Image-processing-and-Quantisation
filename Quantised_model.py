import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from PIL import Image
import torchvision as tv
from torchvision import transforms
import matplotlib.pyplot as plt

class UBDmDN(torch.nn.Module):
    def __init__(self):
        super(UBDmDN,self).__init__()

        self.conv=nn.ModuleList()
        self.conv.append(nn.Conv2d(1,32,11,1,padding=5))

        self.conv.append(nn.Conv2d(32,64,5,1,2))
        self.conv.append(nn.Conv2d(64,64,3,1,1))
        #self.conv.append(nn.Conv2d(16,8,3,1,1))

        #self.conv.append(nn.Conv2d(8,16,3,1,1))
        self.conv.append(nn.Conv2d(64,64,3,1,1))
        self.conv.append(nn.Conv2d(64,32,5,1,2))

        #self.conv.extend([nn.Conv2d(64,64,3,1,padding=1) for _ in range(6)])

        self.conv.append(nn.Conv2d(32,1,11,1,padding=5))


        ##layer 2 !!!!!!!!!!!!!!
        self.conv.append(nn.Conv2d(1,32,11,1,padding=5))

        self.conv.append(nn.Conv2d(32,64,5,1,2))
        self.conv.append(nn.Conv2d(64,128,3,1,1))
        #self.conv.append(nn.Conv2d(16,8,3,1,1))

        #self.conv.append(nn.Conv2d(8,16,3,1,1))
        self.conv.append(nn.Conv2d(128,64,3,1,1))
        self.conv.append(nn.Conv2d(64,32,5,1,2))

        #self.conv.extend([nn.Conv2d(64,64,3,1,padding=1) for _ in range(6)])

        self.conv.append(nn.Conv2d(32,1,11,1,padding=5))


        self.buf=[torch.zeros([4,64,166,166]).to('mps'),torch.zeros([4,64,166,166]).to('mps'),torch.zeros([4,64,166,166]).to('mps')]

        for i in range(len(self.conv[:-1])):
            nn.init.kaiming_normal_(self.conv[i].weight.data,nonlinearity='relu')

        self.bn = nn.ModuleList()

        self.bn.extend([nn.BatchNorm2d(32, 32) for _ in range(2)])
        # initialize the weights of the Batch normalization layers
        for i in range(2):
            nn.init.constant_(self.bn[i].weight.data, 1.25 * np.sqrt(32))

    def activation_quant(self,x: torch.Tensor):
        """
        Quantize the activation tensor to scaled 8 bit based on the mean of the absolute values described by
        https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf

        This doesn't change the type of the activations, but rather the values themselves.
        Args:
            x (torch.Tensor): activations
        Returns:
            u (torch.Tensor): quantized weights
        """
        scale = 1 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        u = (x * scale).round().clamp_(-1, 1) / scale
        return u

    def convout(self, tensor):
        sum=torch.zeros([1,1,480,640]).to('mps')
        for i in range(tensor.size(1)):
            out_1 = tensor[:, i:i+1, :, :]
            print(out_1.size())
            sum += out_1
        sum=sum/32
        plt.imshow(sum[0].cpu().permute(1, 2, 0) + 1.0 / 2.0, cmap='gray')
        plt.show()



    def forward(self,x):

        h= self.activation_quant(F.relu(self.conv[0](x)))
        out1=h
        self.convout(out1)

        h=self.activation_quant(F.relu(self.conv[1](out1)))
        out2=h
        self.convout(out2)

        h=self.activation_quant(F.relu(self.conv[2](out2)))
        out3=h
        self.convout(out3)

        h=self.activation_quant(F.relu(self.conv[3](out3)))
        out4=out2+h
        self.convout(out4)

        h=self.activation_quant(F.relu((self.conv[4](out4))))
        out5=out1+h
        self.convout(out5)

        h=self.activation_quant(F.relu(self.conv[5](out5)))
        out6=h
        self.convout(out6)

        #second u net
        x_=out6
        h = self.activation_quant(F.relu(self.conv[6](x_)))
        out7 = h
        self.convout(out7)

        h = self.activation_quant(F.relu((self.conv[7](out7))))
        out8 = h
        self.convout(out8)

        h = self.activation_quant(F.relu(self.conv[8](out8)))
        out9 = h
        self.convout(out9)

        h = self.activation_quant(F.relu(self.conv[9](out9)))
        out10 =out8+h
        self.convout(out10)

        h = self.activation_quant(F.relu((self.conv[10](out10))))
        out11 = out7+h
        self.convout(out11)

        h = self.activation_quant(F.relu(self.conv[11](out11)))
        out12 =h
        #out12 = out12[:, :, 0:180, 0:180]
        self.convout(out12)

        output = x - out12




        #for i in range(3):
        #    h=F.relu(self.bn[i*2](self.conv[i*2+1](h)))
        #    h=F.relu(self.bn[i*2+1](self.conv[i*2+2](h)))

        #    if(h.size()!=self.buf[i].size()):
        #        self.buf[i]=torch.zeros(h.size()).to('mps')
        #    else:
        #        h=h+self.buf[i]

        #h=self.conv[7](h)
        #self.buf[i]=h
        return output



class SplitLoss(nn.Module):
        def _init_(self):
            super()._init_()
            self.mse = nn.MSELoss()

        def forward(self, pred, actual):
            # return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

            parts = torch.split(torch.abs(pred - actual), 4, 0)

            sum = 0
            for tensor in parts:
                sum += torch.mean(tensor)
            return sum
