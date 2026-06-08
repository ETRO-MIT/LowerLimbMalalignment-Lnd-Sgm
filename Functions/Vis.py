# Python packages
import os
import pandas as pd
from pathlib import Path
import SimpleITK as sitk
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Save visualization of the bounding-boxes
def save_bboxes_figure(itk_image, set_bboxes, output_path, file_name):
    # Set file name with extension
    file_name = file_name + '_boxes.png'

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 12))

    # Iterate over bounding boxes -> Plot them
    for box in set_bboxes:
        # Estimated rectangles are in format: x_min, y_min, x_max, y_max | need to change to x, y, w, h
        box_drawn = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
                                      linewidth=2, ls='--', edgecolor='r', facecolor='none')
        ax.add_patch(box_drawn)

    # Show image
    plt.imshow(sitk.GetArrayFromImage(itk_image), cmap='gray')

    # Set plot
    plt.xticks([])
    plt.yticks([])

    # Save figure
    plt.savefig(os.path.join(output_path, file_name), dpi=300, bbox_inches='tight')
    plt.close()


# Save visualization of the main axes
def save_axes_figure(target_image: sitk.Image, set_coordinates: pd.DataFrame,
                     file_name: str, detection_method: str, output_path: Path):
    # Convert image -> array
    fll_array = sitk.GetArrayFromImage(target_image)

    # Get the right leg landmarks
    right_landmarks = set_coordinates.loc[(set_coordinates['Leg side'] == 'Right')]
    left_landmarks = set_coordinates.loc[(set_coordinates['Leg side'] == 'Left')]

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 10))
    ax.imshow(fll_array, cmap='gray')

    # Create the axes
    # Femur mechanical axes
    ax.plot([right_landmarks.loc[(right_landmarks['Landmark'] == 'HF')]['Pix X'].item(),
             right_landmarks.loc[(right_landmarks['Landmark'] == 'CK')]['Pix X'].item()],
            [right_landmarks.loc[(right_landmarks['Landmark'] == 'HF')]['Pix Y'].item(),
            right_landmarks.loc[(right_landmarks['Landmark'] == 'CK')]['Pix Y'].item()], ls='-', c='red'
            )
    ax.plot([left_landmarks.loc[(left_landmarks['Landmark'] == 'HF')]['Pix X'].item(),
             left_landmarks.loc[(left_landmarks['Landmark'] == 'CK')]['Pix X'].item()],
            [left_landmarks.loc[(left_landmarks['Landmark'] == 'HF')]['Pix Y'].item(),
            left_landmarks.loc[(left_landmarks['Landmark'] == 'CK')]['Pix Y'].item()], ls='-', c='red'
            )

    # Tibia mechanical axes
    ax.plot([right_landmarks.loc[(right_landmarks['Landmark'] == 'CK')]['Pix X'].item(),
             right_landmarks.loc[(right_landmarks['Landmark'] == 'AM')]['Pix X'].item()],
            [right_landmarks.loc[(right_landmarks['Landmark'] == 'CK')]['Pix Y'].item(),
             right_landmarks.loc[(right_landmarks['Landmark'] == 'AM')]['Pix Y'].item()], ls='-', c='blue'
            )
    ax.plot([left_landmarks.loc[(left_landmarks['Landmark'] == 'CK')]['Pix X'].item(),
             left_landmarks.loc[(left_landmarks['Landmark'] == 'AM')]['Pix X'].item()],
            [left_landmarks.loc[(left_landmarks['Landmark'] == 'CK')]['Pix Y'].item(),
             left_landmarks.loc[(left_landmarks['Landmark'] == 'AM')]['Pix Y'].item()], ls='-', c='blue'
            )

    # Joint lines
    ax.plot([right_landmarks.loc[(right_landmarks['Landmark'] == 'R-FC')]['Pix X'].item(),
             right_landmarks.loc[(right_landmarks['Landmark'] == 'L-FC')]['Pix X'].item()],
            [right_landmarks.loc[(right_landmarks['Landmark'] == 'R-FC')]['Pix Y'].item(),
             right_landmarks.loc[(right_landmarks['Landmark'] == 'L-FC')]['Pix Y'].item()], ls='-', c='orange'
            )
    ax.plot([left_landmarks.loc[(left_landmarks['Landmark'] == 'R-FC')]['Pix X'].item(),
             left_landmarks.loc[(left_landmarks['Landmark'] == 'L-FC')]['Pix X'].item()],
            [left_landmarks.loc[(left_landmarks['Landmark'] == 'R-FC')]['Pix Y'].item(),
             left_landmarks.loc[(left_landmarks['Landmark'] == 'L-FC')]['Pix Y'].item()], ls='-', c='orange'
            )
    ax.plot([right_landmarks.loc[(right_landmarks['Landmark'] == 'R-TP')]['Pix X'].item(),
             right_landmarks.loc[(right_landmarks['Landmark'] == 'L-TP')]['Pix X'].item()],
            [right_landmarks.loc[(right_landmarks['Landmark'] == 'R-TP')]['Pix Y'].item(),
             right_landmarks.loc[(right_landmarks['Landmark'] == 'L-TP')]['Pix Y'].item()], ls='-', c='purple'
            )
    ax.plot([left_landmarks.loc[(left_landmarks['Landmark'] == 'R-TP')]['Pix X'].item(),
             left_landmarks.loc[(left_landmarks['Landmark'] == 'L-TP')]['Pix X'].item()],
            [left_landmarks.loc[(left_landmarks['Landmark'] == 'R-TP')]['Pix Y'].item(),
             left_landmarks.loc[(left_landmarks['Landmark'] == 'L-TP')]['Pix Y'].item()], ls='-', c='purple'
            )

    # Set plot
    plt.xticks([])
    plt.yticks([])

    # Save figure
    file_name_raw = file_name.split('.')[0]
    fig_file_name = file_name_raw + '_FLL_{}_Axes.png'.format(detection_method)
    plt.savefig(os.path.join(output_path, fig_file_name), dpi=300, bbox_inches='tight')
    plt.close()
