# Python packages
import torch
import numpy as np
import SimpleITK as sitk


# Determine the new spacing for an image that will be resample to a new size.
def get_new_spacing(itk_image, new_size=(512, 512)):
    original_size = itk_image.GetSize()
    original_spacing = itk_image.GetSpacing()

    new_spacing = [(original_size[0] * original_spacing[0]) / new_size[0],
                   (original_size[1] * original_spacing[1]) / new_size[1]]

    return new_spacing


# Resample image function
def resample_image(itk_image, out_spacing=(1.0, 1.0), is_label=False):
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)


# Execute pre-processing. First, resample an image to the desired size. Then, convert the image to an array and scale
# the intensity between [0, 1] so that the deep learning models can read the image.
def pre_processing(itk_image, new_size=(512, 512), scale_intensity=True):
    # Resample image
    new_spacing = get_new_spacing(itk_image, new_size=new_size)
    resampled_image = resample_image(itk_image, out_spacing=[new_spacing[0], new_spacing[1]])

    # Convert to array
    image_array = sitk.GetArrayFromImage(resampled_image).astype('float32')
    if len(image_array.shape) < 3:
        image_shape = image_array.shape
        new_array = np.zeros((3, image_shape[0], image_shape[1]))
        new_array[:] = image_array
    else:
        new_array = np.moveaxis(image_array, -1, 0)

    # Scale intensity -> Convert to torch
    if scale_intensity:
        new_array /= 255.
    torch_image = torch.from_numpy(new_array).float()

    return resampled_image, torch_image
