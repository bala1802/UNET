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
    
class UNET(nn.Module):

    #Outchannels=1, because the problem statement is for the Binary Image segmentation
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        
        self.ups = nn.ModuleList() #This will be storing all the convolutional layers
        self.downs = nn.ModuleList() #This will be storing all the convolutional layers

        #Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #Downpart of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels=in_channels, out_channels=feature))
            in_channels = feature
        
        #Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(in_channels=feature*2, out_channels=feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(in_channels=feature*2, out_channels=feature))
        
        #features[-1] because the end layer 512 in converted to 1024
        self.bottleneck = DoubleConv(in_channels=features[-1], out_channels=features[-1]*2)
        self.final_conv = nn.Conv2d(in_channels=features[-1], out_channels=out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        

