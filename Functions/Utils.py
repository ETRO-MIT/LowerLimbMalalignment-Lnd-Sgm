# Python functions
import os
import re
import pandas as pd
from pathlib import Path


# Sort string elements in a natural way
def natural_sort(list_to_sort):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(list_to_sort, key=alphanum_key)


# Filter hidden files
def ignore_hidden(list_of_files):
    filtered_files = []
    for file in list_of_files:
        if file.startswith('.'):
            continue
        else:
            filtered_files.append(file)

    # Apply natural sorting
    filtered_files = natural_sort(filtered_files)

    return filtered_files


# Read directory with all possible names
def read_directory(path_to_dir):
    files_in_directory = os.listdir(path_to_dir)
    files_in_directory = ignore_hidden(files_in_directory)

    return files_in_directory


# Flatten a list of lists
def flatten(t):

    return [item for sublist in t for item in sublist]


# Transform a set of coordinates that are in a coordinate-system of a resampled image, to the original coordinate system
def transform_coordinates(original_image, target_image, set_coordinates):
    # Convert pixel points to physical points
    physical_points = []
    for coordinates in set_coordinates:
        physical_point = original_image.TransformIndexToPhysicalPoint([int(coordinates[0]), int(coordinates[1])])
        physical_points.append(physical_point)

    # Convert physical points to the new coordinate system
    new_coordinates = []
    for coordinates in physical_points:
        new_x_y = target_image.TransformPhysicalPointToIndex(coordinates)
        new_coordinates.append(new_x_y)

    return new_coordinates


# Create directory where to save the results
def make_results_dir(path_to_dir):
    if not os.path.exists(path_to_dir):
        os.mkdir(path_to_dir)


# Create a directory where to store the results for a specific case
def make_subject_results_dir(path_to_dir):
    path = Path(path_to_dir)

    if path.is_file() or path.suffix:
        new_path_to_dir = path.parent / path.stem
        make_results_dir(new_path_to_dir)
    else:
        make_results_dir(path)


# Save the malalignment results and the distance measures in a .csv file
def save_results(right_leg_geometrics, left_leg_geometrics, output_path, image_id):
    file_name = str(image_id + '.csv')

    columns = ['Metrics - Right leg', 'Results - Right leg',
               'Metrics - Left leg', 'Results - Left leg']

    metrics_names = ['MAD [mm]', 'mLDFA [°]', 'mMPTA [°]', 'FVA [°]', 'HKA [°]']

    df = pd.DataFrame(columns=columns)
    df['Metrics - Right leg'] = metrics_names
    df['Results - Right leg'] = right_leg_geometrics
    df['Metrics - Left leg'] = metrics_names
    df['Results - Left leg'] = left_leg_geometrics

    df.to_excel(os.path.join(output_path, file_name))
