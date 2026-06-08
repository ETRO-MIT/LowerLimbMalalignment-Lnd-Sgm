# Python packages
import torch

# Custom functions
from Models.SGR import UNetSGR
from Weights.Weights import  weights
from Detection.UtilsSGR import execute_knee_transformation, execute_femur_transformation, execute_ankle_transformation


# Class that builds an SGR model depending on the type of joint to be analyzed: hip (femur), knee, or ankle. After
# building the model, detection of the landmarks takes place. Finally, post-processing is done to pass the estimated
# landmarks (that are on the 512 x 512 coordinate system) to the original coordinate system (the one of the FLL X-ray).
class DetectionSGR:
    def __init__(self, type_joint: str):
        super(DetectionSGR, self).__init__()
        # Set device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # According to the type of joint, set the following parameters:
        # 1) Path to the weights
        # 2) Number of landmark coordinates to estimate
        self.type_joint = type_joint
        # Ankle
        if self.type_joint == 'Ankle':
            self.path_to_weights = weights['SGR']['Ankle']
            self.number_of_coordinates = 4  # 2 x-y landmarks

        # Femur
        elif self.type_joint == 'Femur':
            self.path_to_weights = weights['SGR']['Femur']
            self.number_of_coordinates = 2  # 1 x-y landmark

        # Knee
        elif self.type_joint == 'Knee':
            self.path_to_weights = weights['SGR']['Knee']
            self.number_of_coordinates = 10  # 5 x-y landmarks

        # Error
        else:
            raise ValueError('Select a correct type of joint: "Ankle", "Femur", or "Knee".')

    # Build SGR model
    def build_model(self):
        # Create SGR model
        sgr_model = UNetSGR(number_coordinates=self.number_of_coordinates)
        sgr_model.load_state_dict(torch.load(self.path_to_weights, map_location=self.device))
        sgr_model.to(self.device)

        return sgr_model

    # Make estimations
    def estimate_landmarks(self, torch_xrays_joints, sitk_xrays_joints, sitk_fll_xray):
        # Un-pack the data in right/left leg (necessary to correctly transform the landmark physical points)
        sitk_right_xray_joint, sitk_left_xray_joint = sitk_xrays_joints[0], sitk_xrays_joints[1]

        # Build the model
        sgr_model = self.build_model()

        # Estimate the landmarks
        sgr_model.eval()
        with torch.no_grad():
            # Estimate position of the landmarks
            _, output_landmarks = sgr_model(torch_xrays_joints)

        # Split into right/left leg sides
        right_landmarks, left_landmarks = output_landmarks[0], output_landmarks[1]
        right_landmarks = torch.unsqueeze(right_landmarks, dim=0)
        left_landmarks = torch.unsqueeze(left_landmarks, dim=0)

        # Transform the estimations to the original coordinate system. Physical points are needed for calculating
        # the malalignment metrics, the pixel coordinates are required for visualization purposes. Additionally, the
        # estimations are arranged accordingly to the type of joint (necessary to save and display the results)
        if self.type_joint == 'Ankle':
            px_512, px_fll, phys_points_fll = execute_ankle_transformation(right_landmarks, left_landmarks,
                                                                           sitk_right_xray_joint,
                                                                           sitk_left_xray_joint,
                                                                           sitk_fll_xray)
        elif self.type_joint == 'Femur':
            px_512, px_fll, phys_points_fll = execute_femur_transformation(right_landmarks, left_landmarks,
                                                                           sitk_right_xray_joint,
                                                                           sitk_left_xray_joint,
                                                                           sitk_fll_xray)
        elif self.type_joint == 'Knee':
            px_512, px_fll, phys_points_fll = execute_knee_transformation(right_landmarks, left_landmarks,
                                                                          sitk_right_xray_joint,
                                                                          sitk_left_xray_joint,
                                                                          sitk_fll_xray)
        else:
            raise ValueError('Unknown joint type')

        return px_512, px_fll, phys_points_fll
