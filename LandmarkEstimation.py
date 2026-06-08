# Python packages
import torch
import SimpleITK as sitk

# Custom function
from Detection.SGR import DetectionSGR
from Detection.Heatmap import HeatmapDetection
from Detection.Regression import RegressionDetection
from Detection.Segmentation import  SegmentationDetection


# Main class for landmark coordinate estimation
class LandmarkEstimation:
    def __init__(self, original_fll_xray: sitk.Image, estimation_method: str, femur_tensor: torch.Tensor,
                 femur_sitk: list, knee_torch: torch.Tensor, knee_sitk: list, ankle_torch: torch.Tensor,
                 ankle_sitk: list):
        self.original_fll_xray = original_fll_xray  # FLL X-ray in sitk format
        self.estimation_method = estimation_method  # Estimation method: "Segmentation", "Heatmap", "SGR", or "Regression"
        self.femur_tensor = femur_tensor  # Hip cropped tensors on 512x512 size
        self.femur_sitk = femur_sitk  # List of hip cropped images on sitk format
        self.knee_torch = knee_torch  # Knee cropped tensors on 512x512 size
        self.knee_sitk = knee_sitk  # List of knee cropped images on sitk format
        self.ankle_torch = ankle_torch  # Ankle cropped tensors on 512x512 size
        self.ankle_sitk = ankle_sitk  # List of ankle cropped images on sitk format

    # Execute landmark estimation according to the desired localization method
    def execute(self):
        if self.estimation_method == "Segmentation":
            # Hip landmark
            hip_model = SegmentationDetection(type_joint='Femur')
            hip_results = hip_model.estimate_landmarks(self.femur_tensor, self.femur_sitk, self.original_fll_xray)

            # Knee landmarks
            knee_model = SegmentationDetection(type_joint='Knee')
            knee_results = knee_model.estimate_landmarks(self.knee_torch, self.knee_sitk, self.original_fll_xray)

            # Ankle landmarks
            ankle_model = SegmentationDetection(type_joint='Ankle')
            ankle_results = ankle_model.estimate_landmarks(self.ankle_torch, self.ankle_sitk, self.original_fll_xray)

            return hip_results, knee_results, ankle_results

        elif self.estimation_method == "Heatmap":
            # Hip landmark
            hip_model = HeatmapDetection(type_joint='Femur')
            hip_results = hip_model.estimate_landmarks(self.femur_tensor, self.femur_sitk, self.original_fll_xray)

            # Knee landmarks
            knee_model = HeatmapDetection(type_joint='Knee')
            knee_results = knee_model.estimate_landmarks(self.knee_torch, self.knee_sitk, self.original_fll_xray)

            # Ankle landmarks
            ankle_model = HeatmapDetection(type_joint='Ankle')
            ankle_results = ankle_model.estimate_landmarks(self.ankle_torch, self.ankle_sitk, self.original_fll_xray)

            return hip_results, knee_results, ankle_results

        elif self.estimation_method == "Regression":
            # Hip landmark
            hip_model = RegressionDetection(type_joint='Femur')
            hip_results = hip_model.estimate_landmarks(self.femur_tensor, self.femur_sitk, self.original_fll_xray)

            # Knee landmarks
            knee_model = RegressionDetection(type_joint='Knee')
            knee_results = knee_model.estimate_landmarks(self.knee_torch, self.knee_sitk, self.original_fll_xray)

            # Ankle landmarks
            ankle_model = RegressionDetection(type_joint='Ankle')
            ankle_results = ankle_model.estimate_landmarks(self.ankle_torch, self.ankle_sitk, self.original_fll_xray)

            return hip_results, knee_results, ankle_results
        elif self.estimation_method == "SGR":
            # Hip landmark
            hip_model = DetectionSGR(type_joint='Femur')
            hip_results = hip_model.estimate_landmarks(self.femur_tensor, self.femur_sitk, self.original_fll_xray)

            # Knee landmarks
            knee_model = DetectionSGR(type_joint='Knee')
            knee_results = knee_model.estimate_landmarks(self.knee_torch, self.knee_sitk, self.original_fll_xray)

            # Ankle landmarks
            ankle_model = DetectionSGR(type_joint='Ankle')
            ankle_results = ankle_model.estimate_landmarks(self.ankle_torch, self.ankle_sitk, self.original_fll_xray)

            return hip_results, knee_results, ankle_results
        else:
            raise ValueError("Detection method must be either 'Segmentation', 'Heatmap', 'Regression', or 'SGR'.")



