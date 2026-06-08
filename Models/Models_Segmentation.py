# Custom functions
from Models.UNet import UNet


# Create a U-Net model that can be used for segmentation or heatmap landmark localization
def create_segmentation_model(number_landmarks: int):
    model = UNet(number_landmarks=number_landmarks)

    return model
