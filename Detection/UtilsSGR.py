# Custom functions
from Functions.Utils import transform_coordinates
from Functions.PreProcessing import pre_processing


# Crop the desired region from the FLL image -> Execute pre-processing
def get_rois(bbox_set, original_image, is_diaphysis=False):
    # Crop the image
    cropped_image = original_image[bbox_set[0]:bbox_set[2], bbox_set[1]:bbox_set[3]]

    # Get resampled ITK image and the correspondant torch image
    resampled_crop_image, resampled_crop_torch_image = pre_processing(cropped_image, new_size=(512, 512))
    if is_diaphysis:
        return cropped_image, resampled_crop_image, resampled_crop_torch_image
    else:
        return resampled_crop_image, resampled_crop_torch_image


# Execute post-processing (transformation) of the femur landmarks
def execute_femur_transformation(right_femur_landmark, left_femur_landmark,
                                 right_resampled_image, left_resampled_image,
                                 original_image):
    # Right leg
    # Convert to numpy -> Pass to original image coordinate system -> Convert coordinates to physical points
    femur_right_lnd = right_femur_landmark.cpu().numpy()
    fll_femur_right_lnd = transform_coordinates(right_resampled_image, original_image, femur_right_lnd)
    phy_fll_femur_right_lnd = original_image.TransformIndexToPhysicalPoint(fll_femur_right_lnd[0])

    # Left leg
    # Convert to numpy -> Pass to original image coordinate system -> Convert coordinates to physical points
    femur_left_lnd = left_femur_landmark.cpu().numpy()
    fll_femur_left_lnd = transform_coordinates(left_resampled_image, original_image, femur_left_lnd)
    phy_fll_femur_left_lnd = original_image.TransformIndexToPhysicalPoint(fll_femur_left_lnd[0])

    # Pack the data
    pixels_512 = [femur_right_lnd, femur_left_lnd]
    fll_coordinates = [fll_femur_right_lnd, fll_femur_left_lnd]
    fll_phys_points = [phy_fll_femur_right_lnd, phy_fll_femur_left_lnd]

    return pixels_512, fll_coordinates, fll_phys_points


# Execute post-processing (transformation) of the knee landmarks
def execute_knee_transformation(set_right_knee_landmarks, set_left_knee_landmarks,
                                right_resampled_image, left_resampled_image,
                                original_image):
    # Right leg
    # Pass from torch to numpy
    knee_right_landmarks = set_right_knee_landmarks.cpu().numpy()[0]

    # Put the coordinates in the correct format
    right_condyle_line = [[knee_right_landmarks[0], knee_right_landmarks[1]],
                          [knee_right_landmarks[2], knee_right_landmarks[3]]]
    right_tibial_line = [[knee_right_landmarks[4], knee_right_landmarks[5]],
                         [knee_right_landmarks[6], knee_right_landmarks[7]]]
    center_knee_right = [knee_right_landmarks[8], knee_right_landmarks[9]]

    # Transform coordinates to the original image coordinate system -> convert to physical points
    fll_right_condyle_line = transform_coordinates(right_resampled_image, original_image, right_condyle_line)
    fll_right_condyle_line_r = original_image.TransformIndexToPhysicalPoint(fll_right_condyle_line[0])
    fll_right_condyle_line_l = original_image.TransformIndexToPhysicalPoint(fll_right_condyle_line[1])
    phys_fll_right_femur_condyle_line = [fll_right_condyle_line_r, fll_right_condyle_line_l]

    fll_right_tibial_line = transform_coordinates(right_resampled_image, original_image, right_tibial_line)
    fll_right_tibial_line_r = original_image.TransformIndexToPhysicalPoint(fll_right_tibial_line[0])
    fll_right_tibial_line_l = original_image.TransformIndexToPhysicalPoint(fll_right_tibial_line[1])
    phys_fll_right_femur_tibial_line = [fll_right_tibial_line_r, fll_right_tibial_line_l]

    fll_center_knee_right = transform_coordinates(right_resampled_image, original_image, [center_knee_right])
    phy_fll_center_knee_right = original_image.TransformIndexToPhysicalPoint(fll_center_knee_right[0])

    right_knee_coordinates = [fll_right_condyle_line, fll_right_tibial_line, fll_center_knee_right]
    right_knee_points = [phys_fll_right_femur_condyle_line, phys_fll_right_femur_tibial_line, phy_fll_center_knee_right]

    # Left leg
    # Pass from torch to numpy
    knee_left_landmarks = set_left_knee_landmarks.cpu().numpy()[0]

    # Put the coordinates in the correct format
    left_condyle_line = [[knee_left_landmarks[0], knee_left_landmarks[1]],
                         [knee_left_landmarks[2], knee_left_landmarks[3]]]
    left_tibial_line = [[knee_left_landmarks[4], knee_left_landmarks[5]],
                        [knee_left_landmarks[6], knee_left_landmarks[7]]]
    center_knee_left = [knee_left_landmarks[8], knee_left_landmarks[9]]

    # Transform coordinates to the original image coordinate system -> convert to physical points
    fll_left_condyle_line = transform_coordinates(left_resampled_image, original_image, left_condyle_line)
    fll_left_condyle_line_r = original_image.TransformIndexToPhysicalPoint(fll_left_condyle_line[0])
    fll_left_condyle_line_l = original_image.TransformIndexToPhysicalPoint(fll_left_condyle_line[1])
    phys_fll_left_femur_condyle_line = [fll_left_condyle_line_r, fll_left_condyle_line_l]

    fll_left_tibial_line = transform_coordinates(left_resampled_image, original_image, left_tibial_line)
    fll_left_tibial_line_r = original_image.TransformIndexToPhysicalPoint(fll_left_tibial_line[0])
    fll_left_tibial_line_l = original_image.TransformIndexToPhysicalPoint(fll_left_tibial_line[1])
    phys_fll_left_femur_tibial_line = [fll_left_tibial_line_r, fll_left_tibial_line_l]

    fll_center_knee_left = transform_coordinates(left_resampled_image, original_image, [center_knee_left])
    phy_fll_center_knee_left = original_image.TransformIndexToPhysicalPoint(fll_center_knee_left[0])

    left_knee_coordinates = [fll_left_condyle_line, fll_left_tibial_line, fll_center_knee_left]
    left_knee_points = [phys_fll_left_femur_condyle_line, phys_fll_left_femur_tibial_line, phy_fll_center_knee_left]

    # Pack data
    pixels_512 = [knee_right_landmarks, knee_left_landmarks]
    fll_coordinates = [right_knee_coordinates, left_knee_coordinates]
    fll_phys_points = [right_knee_points, left_knee_points]

    return pixels_512, fll_coordinates, fll_phys_points


# Execute post-processing (transformation) of the ankle landmarks
def execute_ankle_transformation(set_right_ankle_landmarks, set_left_ankle_landmarks,
                                 right_resampled_image, left_resampled_image,
                                 original_image):
    # Right leg
    # Pass from torch to numpy
    ankle_right_landmarks = set_right_ankle_landmarks.cpu().numpy()[0]

    # Set points in the correct format
    right_ankle_line = [[ankle_right_landmarks[0], ankle_right_landmarks[1]],
                        [ankle_right_landmarks[2], ankle_right_landmarks[3]]]

    # Transform to original image coordinate system
    fll_right_ankle_line = transform_coordinates(right_resampled_image, original_image, right_ankle_line)

    # Get mid-point
    fll_mid_point_ankle_right_leg = [(fll_right_ankle_line[0][0] + fll_right_ankle_line[1][0]) / 2,
                                     (fll_right_ankle_line[0][1] + fll_right_ankle_line[1][1]) / 2]

    # Convert to physical point
    fll_mid_point_ankle_right_leg = [int(fll_mid_point_ankle_right_leg[0]), int(fll_mid_point_ankle_right_leg[1])]
    phy_fll_mid_point_ankle_right_leg = original_image.TransformIndexToPhysicalPoint(fll_mid_point_ankle_right_leg)

    # Left leg
    # Pass from torch to numpy
    ankle_left_landmarks = set_left_ankle_landmarks.cpu().numpy()[0]

    # Set points in the correct format
    left_ankle_line = [[ankle_left_landmarks[0], ankle_left_landmarks[1]],
                       [ankle_left_landmarks[2], ankle_left_landmarks[3]]]

    # Transform to the original image coordinate system
    fll_ankle_left_line = transform_coordinates(left_resampled_image, original_image, left_ankle_line)

    # Get mid-point
    fll_mid_point_ankle_left_leg = [(fll_ankle_left_line[0][0] + fll_ankle_left_line[1][0]) / 2,
                                    (fll_ankle_left_line[0][1] + fll_ankle_left_line[1][1]) / 2]

    # Convert to physical point
    fll_mid_point_ankle_left_leg = [int(fll_mid_point_ankle_left_leg[0]), int(fll_mid_point_ankle_left_leg[1])]
    phy_fll_mid_point_ankle_left_leg = original_image.TransformIndexToPhysicalPoint(fll_mid_point_ankle_left_leg)

    # Pack the results
    pixels_512 = [right_ankle_line, left_ankle_line]
    fll_coordinates = [fll_mid_point_ankle_right_leg, fll_mid_point_ankle_left_leg]
    fll_phys_points = [phy_fll_mid_point_ankle_right_leg, phy_fll_mid_point_ankle_left_leg]

    return pixels_512, fll_coordinates, fll_phys_points
