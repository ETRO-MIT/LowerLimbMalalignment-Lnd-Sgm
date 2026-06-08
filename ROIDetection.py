# Python packages
import os
import torch

# Custom functions
from Functions.UtilsROIDetection import (get_model_instance_segmentation, filter_estimations,
                                         transform_bboxes, arrange_bboxes)


# Avoid error when downloading for the first time weights from TorchVision. Only run it the first time, you can delete
# the next two lines of code once the weights have been downloaded from the Torchvision servers.
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Class that takes a Faster R-CNN model from the TorchVision library and adapts it to detect 4 regions on an FLL X-ray:
# hip (femur), diaphysis, knee, and ankle. Once the regions are detected, post-processing is applied to convert the
# coordinates of the bounding boxes from the [1400 x 4200] coordinate system, to the original one.
class DetectionJoints:
    def __init__(self, working_directory):
        super(DetectionJoints, self).__init__()
        # Set parameters
        self.weights = os.path.join(working_directory, 'Weights/ROI_Detection/fll_detection_joints.pth')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_classes = 5  # Background - femur head - diaphysis - knee - ankle

    # Build the Faster R-CNN model
    def build_model(self):
        detection_model = get_model_instance_segmentation(self.num_classes)
        detection_model.load_state_dict(torch.load(self.weights, map_location=self.device))
        detection_model.to(self.device)

        return detection_model

    # Detect the joints on the X-ray
    def detect_joints(self, torch_xray, original_fll_xray, resampled_fll_xray):
        # Build the Faster R-CNN model
        faster_rcnn_model = self.build_model()

        # Pass the torch images to format [Bs, Chn, Width, Height] if needed
        if len(torch_xray.shape) < 3:
            torch_xray = torch_xray.unsqueeze(0)

        # Execute estimations
        faster_rcnn_model.eval()
        with torch.no_grad():
            # Estimate joints
            output = faster_rcnn_model([torch_xray.to(self.device)])

        # Filter -> Transform to original size -> Arrange in right and left
        bboxes = filter_estimations(output)
        original_bboxes = transform_bboxes(resampled_fll_xray, original_fll_xray, bboxes)
        original_bboxes = arrange_bboxes(original_bboxes)

        return original_bboxes
