# Python packages
import torch
import torchvision
import torch.nn as nn

# Avoid error when downloading for the first time weights from TorchVision. Only run it the first time, you can delete
# the next two lines of code once the weights have been downloaded from the Torchvision servers.
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context


# Create a regression model using the VGG-16 model as architecture
def create_regression_model(number_coordinates: int):
    vgg_weights = torchvision.models.VGG16_Weights
    vgg_model = torchvision.models.vgg16(weights=vgg_weights.IMAGENET1K_V1)
    vgg_model.classifier[6] = nn.Linear(4096, number_coordinates)

    return vgg_model
