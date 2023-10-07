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
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
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
        #Final Conv is the 1x1 convolutional layer at the end
        self.final_conv = nn.Conv2d(in_channels=features[-1], out_channels=out_channels, kernel_size=1)
    
    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        #Reversing the Skip connections config, when we from down to up, the Convolutions work from 512 to 64
        #[512, 256, 128, 64]
        skip_connections = skip_connections[::-1]

        #By specifying the step=2, this iteration will work only for the Transposed Convolutions
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2] #Getting the Skip Connection for the specified index

            if x.shape != skip_connection.shape:
                 #Taking the height and the width, the batch size and channels are ignored.
                x = TF.resize(x, size=skip_connection.shape[2:])
            
            concat_skip = torch.cat((skip_connection, x), dim=1) #Adding along the channel dimension
            x = self.ups[idx+1](concat_skip) #Now performing the Up sampling (double convolution)

        return self.final_conv(x)


def run_unet_architecture():

    BATCH_SIZE = 3
    CHANNELS = 1
    WIDTH = 160
    HEIGHT = 160

    x = torch.randn((BATCH_SIZE, CHANNELS, WIDTH, HEIGHT))
    model = UNET(in_channels=1, out_channels=1)
    prediction = model(x)

    print("The actual Input shape is : ", x.shape)
    print("The prediction result shape is : ", prediction.shape)

    # assert(prediction.shape == x.shape)

if __name__ == "__main__":
    run_unet_architecture()
        

