########################################################################################################################
# Copyright 2026 [Sebastian Amador Sanchez / Vrije Universiteit Brussel (VUB)]
#
# email: sebastian.amador.sanchez@vub.be
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0
########################################################################################################################

########################################################################################################################
# Main script for lower-limb malalignment assessment on full-leg radiographs through different landmark localization
# techniques. The pipeline is composed of three sequential stages (see README for full details):
#
# 1. ROI detection and cropping
#    - Regions of interest (ROIs) are detected using a Faster R-CNN model.
#    - Detected ROIs are cropped and passed to dedicated deep learning models for landmark localization.
#
# 2. Landmark estimation
#    - Hip, knee, and ankle ROIs are analyzed by independent models to estimate the corresponding anatomical landmark
#    positions.
#    - Landmark coordinates can be estimated using one of four possible techniques: landmark segmentation, heatmap
#    regression, coordinate regression, segmentation-guided coordinate regression.
#
# 3. Post-processing and metric computation
#    - Predicted landmarks are initially expressed in local ROI coordinate systems.
#    - Coordinates are transformed back to the original full-leg image reference frame.
#    - Malalignment metrics and CPAK class are computed.
########################################################################################################################

# Python packages
import os
import sys
import time
import torch
from pathlib import Path

# Custom functions
from Detection.UtilsSGR import get_rois
from ROIDetection import DetectionJoints
from LandmarkEstimation import LandmarkEstimation
from Functions.PreProcessing import pre_processing
from Functions.ReadImages import read_dicom, read_image
from Functions.Vis import save_bboxes_figure, save_axes_figure
from Functions.CPAK import compute_jlo, compute_ahka, compute_cpak
from Functions.AnglesCalculation import get_MAD, get_mLDFA, get_mMPTA, get_HKA
from Functions.Utils import make_results_dir, read_directory, make_subject_results_dir
from Functions.SaveResults import save_pixels_512, save_fll_estimations, save_malalignment_results

# Initial time stamp
main_start_time = time.time()

# Main function that execute the complete pipeline
def main(path_to_data: Path,
         detection_method: str,
         is_dicom: bool = True):
    # Inspect detection method
    assert detection_method in ["Segmentation", "Heatmap", "Regression", "SGR"], "Detection method must be either 'Segmentation', 'Heatmap', 'Regression' or 'SGR'."

    # Get current working directory
    main_directory = Path(os.getcwd())

    # Set output directory
    output_directory = Path(os.path.join(main_directory, 'Outputs'))
    make_results_dir(output_directory)

    # Get elements (subjects) in the directory
    subjects = read_directory(path_to_data)

    # Iterate over the subjects on the directory
    for subject_id in subjects:
        try:
            # Subject timer -> Get ID without extension
            subj_start_time = time.time()
            print('Reading subject with ID: {}'.format(subject_id))
            raw_subject_id = subject_id.split('.')[0]

            # Create folder where to store the results
            output_subject_folder = Path(os.path.join(output_directory, raw_subject_id))
            make_subject_results_dir(output_subject_folder)

            # Create a sub-folder according to the type of detection method beinf employed
            output_detection_folder = Path(os.path.join(output_subject_folder, detection_method))
            make_subject_results_dir(output_detection_folder)

            # Read image(s) according to the type of input file; then, pre-process the image(s)
            # DICOM series -> Pre-processing: Generate a torch tensor that can be passed to the deep learning model
            if is_dicom:
                path_to_subject = os.path.join(path_to_data, subject_id)
                original_fll_xray = read_dicom(path_to_subject)
                resampled_fll_xray, torch_xray = pre_processing(original_fll_xray, new_size=(1400, 4200))
            # Read a non-medical format -> Pre-processing
            else:
                path_to_subject = os.path.join(path_to_data, subject_id)
                original_fll_xray = read_image(path_to_subject)
                resampled_fll_xray, torch_xray = pre_processing(original_fll_xray, new_size=(1400, 4200))

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # # # # # # # # # # # # # # # # # # # # Execute ROI detection # # # # # # # # # # # # # # # # # # # # # # #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # Detect the joints on each leg side: hip (head of the femur), diaphysis, knee and ankle.
            # Set model -> Generate estimations
            print('Detecting ROIs ...')
            detect_joints_model = DetectionJoints(main_directory)
            boxes_joints = detect_joints_model.detect_joints(torch_xray, original_fll_xray, resampled_fll_xray)

            # Save image with the resultant detected regions
            save_bboxes_figure(original_fll_xray, boxes_joints, output_detection_folder, raw_subject_id)
            print('Done !')

            # Crop the ROIs
            # Femur
            right_femur_resampled, right_femur_torch = get_rois(boxes_joints[0], original_fll_xray)
            left_femur_resampled, left_femur_torch = get_rois(boxes_joints[1], original_fll_xray)
            torch_femurs = torch.stack([right_femur_torch, left_femur_torch])
            sitk_femurs = [right_femur_resampled, left_femur_resampled]

            # Knee
            right_knee_resampled, right_knee_torch = get_rois(boxes_joints[4], original_fll_xray)
            left_knee_resampled, left_knee_torch = get_rois(boxes_joints[5], original_fll_xray)
            torch_knees = torch.stack([right_knee_torch, left_knee_torch])
            sitk_knees = [right_knee_resampled, left_knee_resampled]

            # Ankle
            right_ankle_resampled, right_ankle_torch = get_rois(boxes_joints[6], original_fll_xray)
            left_ankle_resampled, left_ankle_torch = get_rois(boxes_joints[7], original_fll_xray)
            torch_ankles = torch.stack([right_ankle_torch, left_ankle_torch])
            sitk_ankles = [right_ankle_resampled, left_ankle_resampled]

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # # # # # # # # # # # # # # # # # # # # Execute landmark localization # # # # # # # # # # # # # # # # # # #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            print('Detecting landmarks with method: {} ...'.format(detection_method))
            landmark_estimation = LandmarkEstimation(original_fll_xray=original_fll_xray,
                                                     estimation_method=detection_method,
                                                     femur_tensor=torch_femurs, femur_sitk=sitk_femurs,
                                                     knee_torch=torch_knees, knee_sitk=sitk_knees,
                                                     ankle_torch=torch_ankles, ankle_sitk=sitk_ankles)
            hip_results, knee_results, ankle_results = landmark_estimation.execute()

            # Unpack data
            femur_512_coords, femur_fll_coords, femur_fll_phys_points = hip_results[0], hip_results[1], hip_results[2]
            knee_512_coords, knee_fll_coords, knee_fll_phys_points = knee_results[0], knee_results[1], knee_results[2]
            ankle_512_coords, ankle_fll_coords, ankle_fll_phys_points = ankle_results[0], ankle_results[1], ankle_results[2]

            # Save pixel coordinates, physical points, and visualization of the axes
            save_pixels_512(femur_coords=femur_512_coords, knee_coords=knee_512_coords, ankle_coords=ankle_512_coords,
                            file_name=raw_subject_id, detection_method=detection_method, save_dir=output_detection_folder)
            results_fll_sgr = save_fll_estimations(femur_pix_coords=femur_fll_coords,
                                                   knee_pix_coords=knee_fll_coords,
                                                   ankle_pix_coords=ankle_fll_coords,
                                                   femur_phys_points=femur_fll_phys_points,
                                                   knee_phys_points=knee_fll_phys_points,
                                                   ankle_phys_points=ankle_fll_phys_points,
                                                   file_name=raw_subject_id, detection_method=detection_method,
                                                   save_dir=output_detection_folder)
            save_axes_figure(target_image=original_fll_xray, set_coordinates=results_fll_sgr,
                             file_name=raw_subject_id, detection_method=detection_method,
                             output_path=output_detection_folder)
            print('Done!')

            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # # # # # # # # # # # # # # # # # # # # Malalignment analysis # # # # # # # # # # # # # # # # # # # # # # #
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # Right leg
            print('Executing malalignment analysis')
            mad_right_leg = get_MAD(femur_fll_phys_points[0], ankle_fll_phys_points[0], knee_fll_phys_points[0][2])
            m_ldfa_right_leg = get_mLDFA(femur_fll_phys_points[0], knee_fll_phys_points[0][2],
                                         knee_fll_phys_points[0][0])
            m_mpta_right_leg = get_mMPTA(ankle_fll_phys_points[0], knee_fll_phys_points[0][2],
                                         knee_fll_phys_points[0][1])
            hka_right_leg = get_HKA(femur_fll_phys_points[0], knee_fll_phys_points[0][2], ankle_fll_phys_points[0])
            jlo_right_leg = compute_jlo(mldfa_value=m_ldfa_right_leg, mmpta_value=m_mpta_right_leg)
            a_ahka_right_leg = compute_ahka(mldfa_value=m_ldfa_right_leg, mmpta_value=m_mpta_right_leg)
            cpak_right_leg = compute_cpak(ahka_value=a_ahka_right_leg, jlo_value=jlo_right_leg)
            right_leg_results = [mad_right_leg, m_ldfa_right_leg, m_mpta_right_leg, hka_right_leg,
                                 jlo_right_leg, a_ahka_right_leg, cpak_right_leg]

            # Left leg
            mad_left_leg = get_MAD(femur_fll_phys_points[1], ankle_fll_phys_points[1], knee_fll_phys_points[1][2])
            m_ldfa_left_leg = 180 - get_mLDFA(femur_fll_phys_points[1], knee_fll_phys_points[1][2],
                                              knee_fll_phys_points[1][0])
            m_mpta_left_leg = 180 - get_mMPTA(ankle_fll_phys_points[1], knee_fll_phys_points[1][2],
                                              knee_fll_phys_points[1][1])
            hka_left_leg = get_HKA(femur_fll_phys_points[1], knee_fll_phys_points[1][2], ankle_fll_phys_points[1])
            jlo_left_leg = compute_jlo(mldfa_value=m_ldfa_left_leg, mmpta_value=m_mpta_left_leg)
            a_ahka_left_leg = compute_ahka(mldfa_value=m_ldfa_left_leg, mmpta_value=m_mpta_left_leg)
            cpak_left_leg = compute_cpak(ahka_value=a_ahka_left_leg, jlo_value=jlo_left_leg)
            left_leg_results = [mad_left_leg, m_ldfa_left_leg, m_mpta_left_leg, hka_left_leg,
                                jlo_left_leg, a_ahka_left_leg, cpak_left_leg]

            # Save results
            print('Saving results ...')
            save_malalignment_results(set_right_leg=right_leg_results, set_left_leg=left_leg_results,
                                      file_name=raw_subject_id, detection_method=detection_method,
                                      save_path=output_detection_folder)
            end_time = time.time()  # End the timer
            elapsed_time = end_time - subj_start_time
            print(f"Elapsed time: {elapsed_time:.4f} seconds")
            print('Done!')

        except:
            print('Unexpected error:', sys.exc_info()[0])


# Run main function
if __name__ == '__main__':

    # main_dir_images = Path('/Users/sebastian/Documents/VUB/PhD/Deliverables/LLM-Lnd-Sgm/DemoImages/Demo3')
    # detection_technique = 'Regression'
    # dicom_images = True
    # main(main_dir_images, detection_technique, dicom_images)

    # Parse arguments
    main_dir_images = Path(sys.argv[1])  # Path to the directory containing the images
    landmark_detection = sys.argv[2]  # Detection technique: "Segmentation", "Heatmap", "Regression", or "SGR"
    dicom_images = sys.argv[3]  # Indicate if the images are in DICOM or non-medical format
    main(main_dir_images, landmark_detection, eval(dicom_images))
