import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from torch.nn import Parameter
from models import MeshConvolution


class MeshConv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=False):
        super(MeshConv, self).__init__()
        self.mc = MeshConvolution(in_ch, out_ch)
        self.bn = bn

    def forward(self, ft, adj):
        x = self.mc(ft, adj)
        x = F.relu(x, inplace=True)
        return x


class CombinationLayer(nn.Module):
    def __init__(self, in_ch, out_ch, bn=False, bias=True):
        super(CombinationLayer, self).__init__()
        self.bn = bn
        self.weight = Parameter(torch.Tensor(in_ch, out_ch))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ch))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        if self.bias is not None:
            output = output + self.bias
        output = F.relu(output, inplace=True)
        return output


class FCLayer(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True):
        super(FCLayer, self).__init__()
        if bn:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.fc(x)
        return x


class MeshNetV0(nn.Module):
    def __init__(self, nfeat, nclass, dropout=0.5):
        super(MeshNetV0, self).__init__()
        self.dropout = dropout
        # 3 -> 64
        self.mc1 = MeshConv(nfeat, 64)
        # 64 -> 64
        self.mc2 = MeshConv(64, 64)
        # 64 -> 64
        self.mc3 = MeshConv(64, 64)
        # 64 -> 128
        self.mc4 = MeshConv(64, 128)
        # 320( 64 + 64 + 64 + 128) -> 128
        self.cb = CombinationLayer(320, 1024)
        # 1024 -> 512
        self.fc1 = FCLayer(1024, 512)
        # 512 -> 256
        self.fc2 = FCLayer(512, 256)
        # 256 -> n
        self.fc3 = nn.Linear(256, nclass)

    def forward(self, x):
        # 3 -> 64
        x = self.mc1(x)
        x1 = x

        # 64 -> 64
        x = self.mc2(x)
        x2 = x

        # 64 -> 64
        x = self.mc3(x)
        x3 = x

        # 64 -> 128
        x = self.mc4(x)
        x4 = x

        # 320( 64 + 64 + 64 + 128) -> 128
        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.cb(x)

        x, _ = torch.max(x, dim=-2, keepdim=True)
        x = x.view(x.size(0), -1)

        # 1024 -> 512
        x = self.fc1(x)

        # 512 -> 256
        x = self.fc2(x)

        # 256 -> n
        x = self.fc3(x)

        return x
