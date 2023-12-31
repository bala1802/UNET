# UNET
U-NET architecture implementation from scratch using PyTorch.

## Purpose
In this repository, I've shared the implementation of the U-Net Architecture using PyTorch. This implementation is in reference to the paper `U-Net: Convolutional Networks for Biomedical Image Segmentation`.

## About the Carvana Dataset
This dataset is taken from Kaggle - `Carvana Image Masking Challenge Automatically identify the boundaries of the car in an image`. This dataset contains a large number of car images (as .jpg files). Each car has exactly 16 images, each one taken at different angles. Each car has a unique id and images are named according to id_01.jpg, id_02.jpg … id_16.jpg.

## Problem Statement
To develop a Computer Vision algorithm that can automatically identify the boundaries of cars in images.

## UNET Architecture
![Alt text](UNET_Architecture.png)

### UNET Architecture Explanation
The U-Net architecture is an effective Neural Network architecture, used for Image Segmentation tasks like identifying and deliniating objects within images.

#### Encoder - Decoder Structure
The U-Net architecture can be divided into two main parts: `Encoder` and `Decoder` 

#### Encoder
- The Encoder is the top part of the `U` shape designed to capture features from the input image and gradually reduces the spatial dimension
- It is a series of convolutional layers that work like a funnel, detecting simple features at the top and more complex features as we go deeper in the network. These layers reduces the spatial resolution (`width` and `height`), while increasing the number of channels.
- The Encoder is a **Contraction Path**
    - In the Contraction Path, the model learns `WHAT` features are present in the image. These features can be basic patterns, edges, textures or more complex structures
    - However in this path, we lose some of the spatial information about `WHERE` in the image these features are located, since the spatial dimensions (`width` and `height`) of the feature maps are reduced as we move deeper into the Encoder.
    - Example: When we look at the image of the cat, in the Encoder, the model learns `WHAT` features make up a cat, such as the shapes of ears, the fur texture, and the eyes. But it doesn't precisely know `WHERE` these features are in the original image becuase it has been reduced to lower-resolution feature maps.

##### Code:

- Initialization:
                
                for feature in features:
                    self.downs.append(DoubleConv(in_channels=in_channels, out_channels=feature))
                    in_channels = feature

The `DoubleConv` is just two sequential convolutional layers. As mentioned in the figure, the convolutional layers' features ranges `64, 128, 256, 512`.

- Implementation: 

                for down in self.downs:
                    x = down(x)
                    skip_connections.append(x)
                    x = self.pool(x)

The `Double Convoltion` runs and the results are stored in `skip_connections` which is discussed in `Skip Connections` section

#### Decoder
- The Decoder is the bottom part of the `U` shape and designed to expand the feature maps back to the original image size while preserving the important features learned by the Encoder.
- It consists of a series of upsampling and convolutional layers.
- The upsampling layers increase the spatial dimensions, making the feature maps larger.
- The convolutional layers combine the high-level features from the encoder with the upsampled feature maps to generate the final segmentation mask.
- The Decoder is a **Expansion Path**
    - In the expansion path, the model's goal is to figure out `WHAT` those features from the Contraction Path mean in the context of the whole image. It reconstructs the spatial information.
    - The `WHERE` part is resolved in the Expansion Path. The decoder takes the features learned in the contraction path and gradually upsamples them to match the original image's size. Skip connections, which connect corresponding layers in the encoder and decoder, provide "valuable information" about the location of features.
    - Example: In the Expansion Path, the model uses the information it learned earlier "Cat's ears", and places them in the right location on the image, so it knows `WHERE` those features are.

##### Code

- Initialization

            for feature in reversed(features):
                self.ups.append(nn.ConvTranspose2d(in_channels=feature*2, out_channels=feature, kernel_size=2, stride=2))
                self.ups.append(DoubleConv(in_channels=feature*2, out_channels=feature))

The `Transposed Convolution (ConvTranspose2d)` is used to upsample or increase the spatial dimensions of the input data. Followed by that `DoubleConv` sequentially two convolutional layers are added

- Implementation

            for idx in range(0, len(self.ups), 2):
                x = self.ups[idx](x)
                skip_connection = skip_connections[idx//2]

                if x.shape != skip_connection.shape:
                    x = TF.resize(x, size=skip_connection.shape[2:])
                
                concat_skip = torch.cat((skip_connection, x), dim=1)
                x = self.ups[idx+1](concat_skip)

The `Transposed Convolution` layer is added to the `Skip Connection` which directly comes from the Encoder. Thisis discussed in `Skip Connections` section. Followed by that, two convolutional layers are executed sequentially.

#### Skip Connection

To retain the spatial information from the Encoder and send it to the Decoder, the skip connection is used, which is nothing bu the bridge we can see between the `Encoder` and the `Decoder`. This bridge catties the spatial (width, height) information.

#### Bottle Neck Layer

The Layer which connects between the `Encoder` and the `Decoder`. Bottle Neck layer is applied at the end of the Encoder.

![Alt text](image.png)

The End layer of the `Encoder` with feature size as `512` is converted to `1024` in `Decoder` (this is where the Decoder begins)

#### Output Layer

This is the Final Convolutional Layer, which is responsbile for converting the features from `64` to the output channels = 1. Because, the output will be `background in Black` and the `segmented object in white` color.

                self.final_conv = nn.Conv2d(in_channels=features[0], out_channels=out_channels, kernel_size=1)

### Weighted Loss

Let's understand the `weighted loss` based on the statements mentioned in the paper.

![Alt text](image-1.png)

A technique used to train the U-Net model more effectively when dealing with the problem of segmenting objects that are very close to each other, like cells in a microscopy image.

![Alt text](image-2.png)

Suppose we have to use a model to seperate and identify each cell. From the above example, some of the cells are so close together that their boundaries touch, making it challenging for the model to distinguish between them accurately.

When two cells touch, the boundary between them becomes ambiguous, and it's challenging for the model to predict the exact separation line correctly. This can lead to errors in segmentation.

To address this challenge, the U-Net paper suggests using a `weighted loss` in the training process. In this case, the `weighted loss` function assigns different levels of importance or `weights` to different parts of the image.

![Alt text](image-4.png)

By using this weighted loss, the U-Net model pays special attention to the challenging areas where objects touch (boundary between two different objects), making it more likely to produce a better segmentation result in these complex cases.

## Loss Functions in UNET

Two different loss functions are used in the U-Net Image segmentation architecture

- `Binary Cross-Entropy (BCE) Loss`: In Binary image segmentation, each pixel in the mask can be either a foreground(object) pixel or a background pixel. This loss function measures the dissimilarity between the predicted binary mask and the ground truth binary mask. This enables the model to produce pixel-wise probabilities that are close to the actual binary values (0 or 1).

- `Dice Loss`: This loss function measures the overlap between the predicted and ground truth masks, focusing on the intersection of pixels. It enables the model to generate masks that closely match the ground truth.

                Dice = (2 * Intersection) / (Sum of Pixels in Ground Truth + Sum of Pixels in Predicted)
                Dice = (2 * 2) / (4 + 2) = 4 / 6 = 2/3 ≈ 0.67
                Dice Loss = 1-0.67 = 0.33