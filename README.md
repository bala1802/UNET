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
- It is a series of convolutional layers that work like a funnel, detecting simple features at the top and more complex features as we go deeper in the network. These layers reduces the spatial resolution, while increasing the number of channels.
- The Encoder is a **Contraction Path**
    - TODO
    - TODO



#### Decoder
#### Skip Connections
#### Output Layer
#### What and Where Learning, followed by
#### How U-Net Addresses these issues

## Loss Functions in UNET
### Binary Cross Entropy With Logits
### Dice Loss
### Weighted Loss