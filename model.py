import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

'''
The Double Convolution - because, from the architecture, it is evident that in both the Contractive and Expanded view, 
in each block two 3x3 convultional block is used
'''
class DoubleConv(nn.Module):


    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            
            #Padding =1, which will make it as a the same convolution. 
            # The input height and width are going to be the same after the convolution
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)

            )
    
    def forward(self, x):
        return self.conv(x)
