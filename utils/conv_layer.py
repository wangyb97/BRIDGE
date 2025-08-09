import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, if_bias = False, relu=True, same_padding=True, bn=True):
        super(Conv2d, self).__init__()
        p0 = int((kernel_size[0] - 1) / 2) if same_padding else 0
        p1 = int((kernel_size[1] - 1) / 2) if same_padding else 0
        padding = (p0, p1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True if if_bias else False)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        x = F.dropout(x, 0.3, training=self.training)
        return x


class Conv1d(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1,),
                 dilation=(1,), if_bias=False, relu=True, same_padding=True, bn=True):
        super(Conv1d, self).__init__()
        p0 = int((kernel_size[0] - 1) / 2) if same_padding else 0
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=p0,
                              dilation=dilation, bias=True if if_bias else False)
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        # self.relu = nn.SELU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        x = F.dropout(x, 0.3, training=self.training)
        return x


from torch_conv_kan.kan_convs import FastKANConv1DLayer

# class SimpleConvKAN(nn.Module):
#     def __init__(
#             self,
#             input_channels,
#             out_channels,
#             num_classes = 1,
#             spline_order = 3,
#             kernel_size = 3,
#             groups: int = 1,
#             same_padding=True, 
#             bn=True):
#         super(SimpleConvKAN, self).__init__()
#         p0 = int((kernel_size - 1) / 2) if same_padding else 0
#         self.layers = nn.Sequential(
#             FastKANConv1DLayer(input_channels, out_channels, kernel_size=kernel_size, groups=groups, padding=p0, stride=1, dilation=1,grid_size=2),
#             # nn.BatchNorm1d(out_channels) if bn else None,
#             FastKANConv1DLayer(out_channels, out_channels, kernel_size=kernel_size, groups=groups, padding=p0, stride=1, dilation=1,grid_size=4),
#             # nn.BatchNorm1d(out_channels) if bn else None,
#             FastKANConv1DLayer(out_channels, out_channels, kernel_size=kernel_size, groups=groups, padding=p0, stride=1, dilation=1,grid_size=8),
#             # nn.BatchNorm1d(out_channels) if bn else None,
#             # KABNConv1DLayer(out_channels, out_channels, kernel_size=kernel_size, groups=groups, padding=p0, stride=1, dilation=1),
#             # KALNConv1DLayer(out_channels, out_channels, kernel_size=kernel_size, groups=groups, padding=p0, stride=3, dilation=1),
#             # KANConv1DLayer(out_channels, out_channels, spline_order=spline_order, kernel_size=kernel_size, groups=groups, padding=p0, stride=1, dilation=1),
#             nn.AdaptiveAvgPool1d(1)
#         )
#         self.bn = nn.BatchNorm1d(out_channels) if bn else None
        
#         # self.output = nn.Linear(out_channels, num_classes)
#         self.drop = nn.Dropout(p=0.3)
        

#     def forward(self, x):
#         x = self.layers(x)
#         # print(x.shape)
#         # x = torch.flatten(x, 1)
#         # if self.bn is not None:
#         #     x = self.bn(x)
#         x = self.drop(x)
#         # x = self.output(x)
#         return x
    
    
class SimpleConvKAN_1layer(nn.Module):
    def __init__(
            self,
            input_channels,
            out_channels,
            kernel_size = 3,
            groups: int = 1,
            grid_size: int = 8,
            same_padding=True,
            bn=True,
            dropout: float = 0.3):
        super(SimpleConvKAN_1layer, self).__init__()
        p0 = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = FastKANConv1DLayer(input_channels, out_channels, kernel_size=kernel_size, groups=groups, padding=p0, stride=1, dilation=1, grid_size=grid_size, dropout=dropout)
        self.bn = nn.BatchNorm1d(out_channels) if bn else None
        # self.drop = nn.Dropout(p=0.3)
        
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        # x = F.dropout(x, 0.3, training=self.training)
        return x
