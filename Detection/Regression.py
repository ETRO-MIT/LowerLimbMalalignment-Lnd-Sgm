# Python packages
import torch
from pathlib import Path

# Custom functions
from Weights.Weights import weights
from Models.Models_Regression import create_regression_model
from Detection.UtilsSGR import execute_femur_transformation, execute_knee_transformation, execute_ankle_transformation


# Build a coordinate regression model
def build_model(number_landmarks: int, path_to_weights: Path):
    # Create VGG-16 model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    regression_model = create_regression_model(number_coordinates=number_landmarks)
    regression_model.load_state_dict(torch.load(path_to_weights, map_location=device))
    regression_model.to(device)

    return regression_model


# Class that builds a Coordinate regression model depending on the type of joint to be analyzed: hip (femur), knee, or
# ankle. After building the model, detection of the landmarks takes place. Finally, post-processing is done to pass the
# estimated landmarks (that are on the 512 x 512 coordinate system) to the original coordinate system (the one of the
# FLL X-ray).
class RegressionDetection:
    def __init__(self, type_joint: str):
        super(RegressionDetection, self).__init__()
        # Set device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # According to the type of joint, set the following parameters:
        # 1) Path to the weights
        # 2) Number of coordinates to estimate
        # Depending on the joint, the number of coordinates is determined by 2 x number of landmarks, since the x-y
        # coordinates have to be computed.
        # Hip (1 landmark): 2 coordinates
        # Knee (5 landmarks): 10 coordinates
        # Ankle (2 landmarks): 4 coordinates
        self.type_joint = type_joint
        # Ankle
        if self.type_joint == 'Ankle':
            self.path_to_weights = weights['Regression']['Ankle']
            self.number_of_coordinates = 4
            self.regression_model = build_model(self.number_of_coordinates, self.path_to_weights)

        # Femur
        elif self.type_joint == 'Femur':
            self.path_to_weights = weights['Regression']['Femur']
            self.number_of_coordinates = 2
            self.regression_model = build_model(self.number_of_coordinates, self.path_to_weights)

        # Knee
        elif self.type_joint == 'Knee':
            self.path_to_weights = weights['Regression']['Knee']
            self.number_of_coordinates = 10
            self.regression_model = build_model(self.number_of_coordinates, self.path_to_weights)

        # Error
        else:
            raise ValueError('Select a correct type of joint: "Ankle", "Femur", or "Knee".')

    # Make estimations
    def estimate_landmarks(self, torch_xrays_joints, sitk_xrays_joints, sitk_fll_xray):
        # Un-pack the data in right/left leg (necessary to correctly transform the landmark physical points)
        sitk_right_xray_joint, sitk_left_xray_joint = sitk_xrays_joints[0], sitk_xrays_joints[1]

        # Estimate the landmarks
        self.regression_model.eval()
        with torch.no_grad():
            # Estimate position of the landmarks
            regression_estimations = self.regression_model(torch_xrays_joints)

        # Split into right/left leg sides
        right_landmarks, left_landmarks = regression_estimations[0], regression_estimations[1]
        right_landmarks = torch.unsqueeze(right_landmarks, dim=0)
        left_landmarks = torch.unsqueeze(left_landmarks, dim=0)

        # Transform the estimations to the original coordinate system. Physical points are needed for calculating
        # the malalignment metrics, the pixel coordinates are required for visualization purposes. Additionally, the
        # estimations are arranged accordingly to the type of joint (necessary to save and display the results)
        if self.type_joint == 'Ankle':
            px_512, px_fll, phys_points_fll = execute_ankle_transformation(right_landmarks,
                                                                           left_landmarks,
                                                                           sitk_right_xray_joint,
                                                                           sitk_left_xray_joint,
                                                                           sitk_fll_xray)
        elif self.type_joint == 'Femur':
            px_512, px_fll, phys_points_fll = execute_femur_transformation(right_landmarks,
                                                                           left_landmarks,
                                                                           sitk_right_xray_joint,
                                                                           sitk_left_xray_joint,
                                                                           sitk_fll_xray)
        elif self.type_joint == 'Knee':
            px_512, px_fll, phys_points_fll = execute_knee_transformation(right_landmarks,
                                                                           left_landmarks,
                                                                          sitk_right_xray_joint,
                                                                          sitk_left_xray_joint,
                                                                          sitk_fll_xray)
        else:
            raise ValueError('Unknown joint type')

        return px_512, px_fll, phys_points_fll
