# Python packages
import torch


# Compute the centroids by performing a weighted sum of the coordinates from the estimated torch probability maps
def compute_batch_centroids(input_tensor: torch.Tensor):
    # Prepare coordinate grids
    grid_y, grid_x = torch.meshgrid(torch.arange(input_tensor.size(2)),
                                    torch.arange(input_tensor.size(3)),
                                    indexing='ij')
    grid_y = grid_y.float().to(input_tensor.device)
    grid_x = grid_x.float().to(input_tensor.device)

    # Compute the weighted sum of coordinates
    sum_y = torch.sum(input_tensor * grid_y.unsqueeze(0).unsqueeze(0), dim=[2, 3])
    sum_x = torch.sum(input_tensor * grid_x.unsqueeze(0).unsqueeze(0), dim=[2, 3])
    sum_area = torch.sum(input_tensor, dim=[2, 3])

    # Compute centroid coordinates
    centroid_y = sum_y / (sum_area + 1e-5)  # Adding a small epsilon to avoid division by zero
    centroid_x = sum_x / (sum_area + 1e-5)

    # Combine centroids along channel dimension
    centroid = torch.stack([centroid_x, centroid_y], dim=-1)  # Shape: [batch_size, num_channels, 2]

    return centroid
