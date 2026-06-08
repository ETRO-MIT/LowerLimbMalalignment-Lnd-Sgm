# Python packages
import copy
import torch
import numpy as np
import SimpleITK as sitk

# Custom functions
from Detection.SgmPostProcessing import (inspect_detection, cast_filter, remove_small_components,
                                         smart_landmark_threshold, get_intensities)


# Compute the centroid of a single probability map
def get_prob_map_centroids(prob_map):
    # Ensure the probability map is a torch tensor
    if not isinstance(prob_map, torch.Tensor):
        prob_map = torch.tensor(prob_map, dtype=torch.float32)

    # Get the shape of the probability map
    height, width = prob_map.shape

    # Create coordinate grids
    x_coords = torch.arange(width).float()
    y_coords = torch.arange(height).float()
    x_coords, y_coords = torch.meshgrid(x_coords, y_coords, indexing='xy')

    # Compute the weighted sum of the coordinates
    sum_x = torch.sum(x_coords * prob_map)
    sum_y = torch.sum(y_coords * prob_map)

    # Compute the sum of the weights
    total_weight = torch.sum(prob_map)

    # Compute the centroid coordinates
    c_x = sum_x / total_weight
    c_y = sum_y / total_weight

    return c_x.item(), c_y.item()


# Method thee to compute the centroids. Start by inspecting if a landmarks is present. If a landmark is localized,
# threshold the landmark channel by using the half value between the max and the min intensities. Next, remove all the
# small components and keep the larger one. Compute the centroid on the larger component. If no landmarks where located,
# directly compute the centroid.
def compute_centroids(prob_map: sitk.Image):
    prob_map_pp = copy.deepcopy(prob_map)
    prob_map_pp = prob_map_pp * 255.  # Multiply per 255 in order to get the components
    prob_map_pp = cast_filter(prob_map_pp)  # Cast to UInt8

    # Iterate over the landmarks
    shape = prob_map.GetSize()
    total_landmarks = shape[-1]
    total_centroids = []
    total_labels = []
    labels = 1
    new_prob_map = sitk.Image(shape, sitk.sitkFloat32)
    if total_landmarks == 1:
        # Femur image
        if inspect_detection(prob_map_pp[:, :, 0]) > 0:
            # Landmark was detected --> Apply smart threshold --> Remove small components --> Compute the centroid
            threshold_prob_map = smart_landmark_threshold(prob_map[:, :, 0])
            thr_prob_map_pp = threshold_prob_map * 255.  # Multiply per 255 in order to get the components
            thr_prob_map_pp = cast_filter(thr_prob_map_pp)  # Cast to UInt8
            thr_prob_map_pp = remove_small_components(thr_prob_map_pp)

            # Divide by the max intensity to scale between [0, 1]
            min_intensity, max_intensity = get_intensities(thr_prob_map_pp)
            new_landmark_prob_map = thr_prob_map_pp / max_intensity
            new_landmark_prob_map = cast_filter(new_landmark_prob_map)

            # Multiply the original mask with the binary mask (to keep only the desired probability maps)
            # Change to Np is done since in SimpleITK the images can't be multiply due to different bit-depth and type
            thr_prob_map_arr = sitk.GetArrayFromImage(threshold_prob_map)
            new_prob_map_arr = sitk.GetArrayFromImage(new_landmark_prob_map)
            output_prob_map = np.multiply(thr_prob_map_arr, new_prob_map_arr).astype('float32')

            # Return to SimpleITK in order to use the centroid computation
            output_prob_map_sitk = sitk.GetImageFromArray(output_prob_map)
            output_prob_map_sitk.CopyInformation(prob_map_pp[:, :, 0])

            # Store new mask -> Compute the centroids
            new_prob_map[:, :, 0] = output_prob_map_sitk
            thr_prob_map_torch = torch.from_numpy(sitk.GetArrayFromImage(output_prob_map_sitk).astype('float32'))
            centroids = get_prob_map_centroids(thr_prob_map_torch)

            # Store the values
            total_centroids.append(centroids)
            total_labels.append(labels)

            # Change metadata
            new_prob_map.CopyInformation(prob_map)

            return total_centroids, total_labels
        else:
            # No landmark was detected --> Compute the centroid directly
            new_prob_map[:, :, 0] = prob_map[:, :, 0]
            thr_prob_map_torch = torch.from_numpy(sitk.GetArrayFromImage(prob_map[:, :, 0]).astype('float32'))
            centroids = get_prob_map_centroids(thr_prob_map_torch)

            # Store the values
            total_centroids.append(centroids)
            total_labels.append(labels)

            # Change metadata
            new_prob_map.CopyInformation(prob_map)

            return total_centroids, total_labels
    else:
        # Knee or Ankle image
        for landmark_id in range(total_landmarks):
            # Inspect if there's a landmark
            if inspect_detection(prob_map_pp[:, :, landmark_id]) > 0:
                # Landmark was detected --> Apply smart threshold --> Compute the centroid
                threshold_prob_map = smart_landmark_threshold(prob_map[:, :, landmark_id])
                thr_prob_map_pp = threshold_prob_map * 255.  # Multiply per 255 in order to get the components
                thr_prob_map_pp = cast_filter(thr_prob_map_pp)  # Cast to UInt8
                thr_prob_map_pp = remove_small_components(thr_prob_map_pp)

                # Divide by the max intensity to scale between [0, 1]
                min_intensity, max_intensity = get_intensities(thr_prob_map_pp)
                new_landmark_prob_map = thr_prob_map_pp / max_intensity
                new_landmark_prob_map = cast_filter(new_landmark_prob_map)

                # Multiply the original mask with the binary mask (to keep only the desired probability maps)
                # Change to Np is done since in SITK the images can't be multiply due to different bit-depth and type
                thr_prob_map_arr = sitk.GetArrayFromImage(threshold_prob_map)
                new_prob_map_arr = sitk.GetArrayFromImage(new_landmark_prob_map)
                output_prob_map = np.multiply(thr_prob_map_arr, new_prob_map_arr).astype('float32')

                # Return to SimpleITK in order to use the centroid computation
                output_prob_map_sitk = sitk.GetImageFromArray(output_prob_map)
                output_prob_map_sitk.CopyInformation(prob_map_pp[:, :, landmark_id])

                # Store new mask -> Compute the centroids
                new_prob_map[:, :, landmark_id] = output_prob_map_sitk
                thr_prob_map_torch = torch.from_numpy(sitk.GetArrayFromImage(output_prob_map_sitk).astype('float32'))
                centroids = get_prob_map_centroids(thr_prob_map_torch)

                # Store the values
                total_centroids.append(centroids)
                total_labels.append(labels)
                labels += 1
            else:
                # No landmark was detected --> Compute the centroid directly
                new_prob_map[:, :, landmark_id] = prob_map[:, :, landmark_id]
                thr_prob_map_torch = torch.from_numpy(sitk.GetArrayFromImage(prob_map[:, :, landmark_id]).astype('float32'))
                centroids = get_prob_map_centroids(thr_prob_map_torch)

                # Store the values
                total_centroids.append(centroids)
                total_labels.append(labels)
                labels += 1

        # Change metadata
        new_prob_map.CopyInformation(prob_map)

        return total_centroids, total_labels
