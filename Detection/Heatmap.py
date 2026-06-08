# Python packages
import torch

# Custom functions
from Weights.Weights import weights
from Detection.Segmentation import build_model
from Detection.UtilsHmp import compute_batch_centroids
from Detection.UtilsSgm import arrange_knee_estimations, arrange_ankle_estimations
from Detection.SgmPostProcessing import transform_coordinates, get_est_physical_points

# Class that builds a Heatmap model depending on the type of joint to be analyzed: hip (femur), knee, or ankle. After
# building the model, detection of the landmarks takes place. Finally, post-processing is done to pass the estimated
# landmarks (that are on the 512 x 512 coordinate system) to the original coordinate system (the one of the FLL X-ray).
# The heatmap employs a U-Net architecture (similar to the segmentation approach). The main differences, is that the
# heatmap was trained using an MSE loss function with Gaussian kernels as objectives
class HeatmapDetection:
    def __init__(self, type_joint: str):
        super(HeatmapDetection, self).__init__()
        # Set device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # According to the type of joint, set the following parameters:
        # 1) Path to the weights
        # 2) The number of landmarks to estimate.
        # For the Ankle and Knee models, the last layer of the U-Net is modified to yield extra masks (corresponding to
        # the number of landmarks to be estimated).
        self.type_joint = type_joint
        # Ankle
        if self.type_joint == 'Ankle':
            self.path_to_weights = weights['Heatmap']['Ankle']
            self.number_of_landmarks = 2
            self.unet_model = build_model(self.number_of_landmarks, self.path_to_weights)

        # Femur
        elif self.type_joint == 'Femur':
            self.path_to_weights = weights['Heatmap']['Femur']
            self.number_of_landmarks = 1
            self.unet_model = build_model(self.number_of_landmarks, self.path_to_weights)

        # Knee
        elif self.type_joint == 'Knee':
            self.path_to_weights = weights['Heatmap']['Knee']
            self.number_of_landmarks = 5
            self.unet_model = build_model(self.number_of_landmarks, self.path_to_weights)

        # Error
        else:
            raise ValueError('Select a correct type of joint: "Ankle", "Femur", or "Knee".')

    # Make estimations
    def estimate_landmarks(self, torch_xrays_joints, sitk_xrays_joints, sitk_fll_xray):
        # Un-pack the data in right/left leg (necessary to correctly transform the landmark physical points)
        sitk_right_xray_joint, sitk_left_xray_joint = sitk_xrays_joints[0], sitk_xrays_joints[1]

        # Estimate the landmarks
        self.unet_model.eval()
        with torch.no_grad():
            # Estimate position of the landmarks
            heatmap_prob_maps = self.unet_model(torch_xrays_joints)

        # Compute the centroids
        pixel_centroids = compute_batch_centroids(heatmap_prob_maps).cpu().numpy()
        right_pixel_centroid, left_pixel_centroid = pixel_centroids[0], pixel_centroids[1]

        # Transform the estimations to the original coordinate system. Physical points are needed for calculating
        # the malalignment metrics, the pixel coordinates are required for visualization purposes.
        # First, transform to the pixel coordinates of the FLL X-ray. Then, compute the physical points.
        right_points_fll = transform_coordinates(sitk_right_xray_joint, sitk_fll_xray, right_pixel_centroid)
        right_physical_points = get_est_physical_points(sitk_fll_xray, right_points_fll)
        left_points_fll = transform_coordinates(sitk_left_xray_joint, sitk_fll_xray, left_pixel_centroid)
        left_physical_points = get_est_physical_points(sitk_fll_xray, left_points_fll)

        # Arrange estimations accordingly to the type of joint (necessary to save and display the results)
        if self.type_joint == 'Femur':
            centroids = [right_pixel_centroid, left_pixel_centroid]  # Pixel coordinates in the 512x512 domain
            pixel_coordinates = [right_points_fll, left_points_fll]  # Pixel coordinates in the FLL domain
            physical_points = [right_physical_points[0], left_physical_points[0]]  # Physical points in the FLL domain

            return centroids, pixel_coordinates, physical_points
        elif self.type_joint == 'Knee':
            arrange_px_512_r, arrange_px_fll_r, arrange_points_fll_r = arrange_knee_estimations(right_pixel_centroid,
                                                                                                right_points_fll,
                                                                                                right_physical_points)
            arrange_px_512_l, arrange_px_fll_l, arrange_points_fll_l = arrange_knee_estimations(left_pixel_centroid,
                                                                                                left_points_fll,
                                                                                                left_physical_points)
            centroids = [arrange_px_512_r, arrange_px_512_l]
            pixel_coordinates = [arrange_px_fll_r, arrange_px_fll_l]
            physical_points = [arrange_points_fll_r, arrange_points_fll_l]

            return centroids, pixel_coordinates, physical_points
        elif self.type_joint == 'Ankle':
            arrange_px_512_r, arrange_px_fll_r, arrange_points_fll_r = arrange_ankle_estimations(right_pixel_centroid,
                                                                                                 right_points_fll,
                                                                                                 right_physical_points)
            arrange_px_512_l, arrange_px_fll_l, arrange_points_fll_l = arrange_ankle_estimations(left_pixel_centroid,
                                                                                                 left_points_fll,
                                                                                                 left_physical_points)
            centroids = [arrange_px_512_r, arrange_px_512_l]
            pixel_coordinates = [arrange_px_fll_r, arrange_px_fll_l]
            physical_points = [arrange_points_fll_r, arrange_points_fll_l]

            return centroids, pixel_coordinates, physical_points
        else:
            raise ValueError("Unknown joint type")
