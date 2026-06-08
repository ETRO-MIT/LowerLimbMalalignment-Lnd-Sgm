# Python packages
import os
import SimpleITK as sitk


# Read a DICOM series.
def read_dicom(path_to_dicom):
    # Inspect if the path points to a directory or to a DICOM image
    if os.path.isdir(path_to_dicom):
        # Set DICOM reader -> Get the series present inside the DICOM folder -> Read the image
        dicom_reader = sitk.ImageSeriesReader()
        dicom_series_names = dicom_reader.GetGDCMSeriesFileNames(path_to_dicom)
        dicom_reader.SetFileNames(dicom_series_names)
        image = dicom_reader.Execute()
    elif os.path.isfile(path_to_dicom):
        # Path points to a DICOM image
        image = sitk.ReadImage(path_to_dicom)
    else:
        raise FileNotFoundError

    # Rescale the intensity between [0-255] -> Return the image as a 2D grayscale image
    image_rescaled = rescale_intensity(image)[:, :, 0]

    return image_rescaled


# Read a non-medical image format (.jpg, .png, etc)
def read_image(path_to_image):
    image = sitk.ReadImage(path_to_image)

    # Inspect length -> Return the image as a 2D grayscale image
    if len(image.GetSize()) > 2:
        return image[:,:,0]
    else:
        return image


# Rescale the intensity of a DICOM image between [0-255]
def rescale_intensity(sitk_image):
    # Set SITK filter and the maximum and minimum values
    scale_int_filter = sitk.RescaleIntensityImageFilter()
    scale_int_filter.SetOutputMaximum(255)
    scale_int_filter.SetOutputMinimum(0)

    # Execute
    image_intensity_rsc = scale_int_filter.Execute(sitk_image)

    return image_intensity_rsc
