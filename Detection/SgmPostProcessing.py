# Python packages
import math
import torch
import numpy as np
import SimpleITK as sitk


# Convert torch mask estimations to SimpleITK format
def torch_to_sitk(torch_prob_maps: torch.Tensor, metadata: dict):
    # Detach
    torch_image = torch_prob_maps.cpu().numpy().astype('float32')

    # To SITK
    target_spacing = metadata['spacing']
    target_origin = metadata['origin']
    sitk_image = sitk.GetImageFromArray(torch_image)
    sitk_image.SetOrigin([target_origin[0], target_origin[1], 0])
    sitk_image.SetSpacing([target_spacing[0], target_spacing[1], 1])

    return sitk_image


# Inspect if a landmarks was detected by counting the number of labels
def inspect_detection(landmark_prob_map: sitk.Image):
    # Define the filters
    cc_filter = sitk.ConnectedComponentImageFilter()
    lss_filter = sitk.LabelShapeStatisticsImageFilter()

    # Get the total of labels
    labeled_image = cc_filter.Execute(landmark_prob_map)
    lss_filter.Execute(labeled_image)
    total_labels = lss_filter.GetNumberOfLabels()

    return total_labels


# Compute the smart threshold in a landmark probability map
def smart_landmark_threshold(landmark_prob_map: sitk.Image):
    # Convert to array
    prob_map_array = sitk.GetArrayFromImage(landmark_prob_map).astype('float32')

    # Compute the min and max intensities
    max_int = prob_map_array.max()
    min_int = prob_map_array.min()
    threshold_value = (max_int + min_int) / 2.

    # Apply threshold
    thr_prob_map_array = np.where(prob_map_array > threshold_value, prob_map_array, 0)

    # Transform to SimpleITK
    new_landmark_mask = sitk.GetImageFromArray(thr_prob_map_array)
    new_landmark_mask.CopyInformation(landmark_prob_map)

    return new_landmark_mask


# Cast a SITK image into a new pixel type
def cast_filter(mask: sitk.Image):
    # Define the filter
    cast = sitk.CastImageFilter()
    cast.SetOutputPixelType(sitk.sitkUInt8)

    # Execute
    return cast.Execute(mask)


# Get Min-Max intensity
def get_intensities(mask_image: sitk.Image()):
    intensity_filter = sitk.MinimumMaximumImageFilter()
    intensity_filter.Execute(mask_image)
    min_intensity = intensity_filter.GetMinimum()
    max_intensity = intensity_filter.GetMaximum()

    return min_intensity, max_intensity


# Remove small components
def remove_small_components(mask: sitk.Image()):
    # Define the filters
    component_filter = sitk.ConnectedComponentImageFilter()
    component_filter.SetFullyConnected(False)
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()

    # Label the components of the mask image -> Iterate over the components to retrieve the one with the largest
    # amount of pixels
    labeled_image = component_filter.Execute(mask)
    label_shape_filter.Execute(labeled_image)
    total_labels = label_shape_filter.GetNumberOfLabels()
    max_index = 0
    max_pixels = 0
    for idx in range(1, total_labels + 1):  # Start at "1" since "0" is the background
        total_pixels = label_shape_filter.GetNumberOfPixels(idx)
        if total_pixels > max_pixels:
            max_index = idx
            max_pixels = total_pixels
        else:
            continue

    filtered_image = labeled_image == max_index

    return filtered_image


# Resample image, apply transformation to the set of coordinates
def transform_coordinates(original_image, resampled_image, set_coordinates):
    # Convert points of bboxes to physical points
    physical_points = []
    for coordinates in set_coordinates:
        if math.isnan(coordinates[0]):
            physical_points.append(coordinates)
        else:
            physical_point = original_image.TransformIndexToPhysicalPoint((int(coordinates[0]), int(coordinates[1])))
            physical_points.append(physical_point)

    # Convert physical points to the new coordinate system
    new_coordinates = []
    for coordinates in physical_points:
        if math.isnan(coordinates[0]):
            new_coordinates.append(coordinates)
        else:
            new_x_y = resampled_image.TransformPhysicalPointToIndex(coordinates)
            new_coordinates.append(new_x_y)

    return new_coordinates


# Convert image indexes to physical points
def get_est_physical_points(fll_original, set_of_landmarks):
    physical_points = []
    for landmark in set_of_landmarks:
        if math.isnan(landmark[0]):
            physical_points.append(landmark)
        else:
            phys = fll_original.TransformIndexToPhysicalPoint([int(landmark[0]), int(landmark[1])])
            physical_points.append(phys)

    return physical_points


# Compute the center of the ankle
def compute_ankle_center(set_points):
    point_a = set_points[0]
    point_b = set_points[1]
    ankle_center = [(point_a[0] + point_b[0]) / 2, (point_a[1] + point_b[1]) / 2]

    return ankle_center
