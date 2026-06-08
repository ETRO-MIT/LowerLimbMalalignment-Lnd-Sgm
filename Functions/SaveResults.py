# Python packages
import os
import pandas as pd
from pathlib import Path


# Save the pixel coordinates (in the 512x512 coordinate system) in a csv file
def save_pixels_512(femur_coords: list, knee_coords: list, ankle_coords: list,
                    file_name: str, detection_method: str, save_dir: Path):
    # Parse the results
    # Femur
    r_femur_x = femur_coords[0][0][0]
    r_femur_y = femur_coords[0][0][1]
    l_femur_x = femur_coords[1][0][0]
    l_femur_y = femur_coords[1][0][1]


    # Knee
    r_fem_cond_a_x = knee_coords[0][0]
    r_fem_cond_a_y = knee_coords[0][1]
    r_fem_cond_b_x = knee_coords[0][2]
    r_fem_cond_b_y = knee_coords[0][3]
    r_tib_plat_a_x = knee_coords[0][4]
    r_tib_plat_a_y = knee_coords[0][5]
    r_tib_plat_b_x = knee_coords[0][6]
    r_tib_plat_b_y = knee_coords[0][7]
    r_cent_knee_x = knee_coords[0][8]
    r_cent_knee_y = knee_coords[0][9]
    l_fem_cond_a_x = knee_coords[1][0]
    l_fem_cond_a_y = knee_coords[1][1]
    l_fem_cond_b_x = knee_coords[1][2]
    l_fem_cond_b_y = knee_coords[1][3]
    l_tib_plat_a_x = knee_coords[1][4]
    l_tib_plat_a_y = knee_coords[1][5]
    l_tib_plat_b_x = knee_coords[1][6]
    l_tib_plat_b_y = knee_coords[1][7]
    l_cent_knee_x = knee_coords[1][8]
    l_cent_knee_y = knee_coords[1][9]

    # Ankle
    r_ankle_a_x = ankle_coords[0][0][0]
    r_ankle_a_y = ankle_coords[0][0][1]
    r_ankle_b_x = ankle_coords[0][1][0]
    r_ankle_b_y = ankle_coords[0][1][1]
    l_ankle_a_x = ankle_coords[1][0][0]
    l_ankle_a_y = ankle_coords[1][0][1]
    l_ankle_b_x = ankle_coords[1][1][0]
    l_ankle_b_y = ankle_coords[1][1][1]

    x_coordinates = [r_femur_x, l_femur_x,
                     r_fem_cond_a_x, r_fem_cond_b_x, r_tib_plat_a_x, r_tib_plat_b_x, r_cent_knee_x,
                     l_fem_cond_a_x, l_fem_cond_b_x, l_tib_plat_a_x, l_tib_plat_b_x, l_cent_knee_x,
                     r_ankle_a_x, r_ankle_b_x, l_ankle_a_x, l_ankle_b_x]
    y_coordinates = [r_femur_y, l_femur_y,
                     r_fem_cond_a_y, r_fem_cond_b_y, r_tib_plat_a_y, r_tib_plat_b_y, r_cent_knee_y,
                     l_fem_cond_a_y, l_fem_cond_b_y, l_tib_plat_a_y, l_tib_plat_b_y, l_cent_knee_y,
                     r_ankle_a_y, r_ankle_b_y, l_ankle_a_y, l_ankle_b_y]
    leg_side = ['Right', 'Left',
                'Right', 'Right', 'Right', 'Right', 'Right',
                'Left', 'Left', 'Left', 'Left', 'Left',
                'Right', 'Right', 'Left', 'Left']
    joint = ['Femur', 'Femur',
             'Knee', 'Knee', 'Knee', 'Knee', 'Knee',
             'Knee', 'Knee', 'Knee', 'Knee', 'Knee',
             'Ankle', 'Ankle', 'Ankle', 'Ankle']
    landmark = ['HF', 'HF',
                'R-FC', 'L-FC', 'R-TP', 'L-TP', 'CK',
                'R-FC', 'L-FC', 'R-TP', 'L-TP', 'CK',
                'R-A', 'L-A', 'R-A', 'L-A']

    # Create DF
    columns = ['Joint', 'Landmark', 'Leg side', 'x', 'y']
    results_df = pd.DataFrame(columns=columns)
    results_df['Joint'] = joint
    results_df['Landmark'] = landmark
    results_df['Leg side'] = leg_side
    results_df['x'] = x_coordinates
    results_df['y'] = y_coordinates

    # Save DF
    raw_file_name = file_name.split('.')[0]
    csv_file_name = raw_file_name + '_{}_512px_coordinates.csv'.format(detection_method)
    results_df.to_csv(os.path.join(save_dir, csv_file_name), index=False)

    return results_df


# Save the pixel coordinates and physical points in the FLL domain
def save_fll_estimations(femur_pix_coords: list, knee_pix_coords: list, ankle_pix_coords: list,
                         femur_phys_points: list, knee_phys_points: list, ankle_phys_points: list,
                         file_name: str, detection_method: str, save_dir:Path):
    # Pixel coordinates
    r_femur_x_coords = femur_pix_coords[0][0][0]
    r_femur_y_coords = femur_pix_coords[0][0][1]
    l_femur_x_coords = femur_pix_coords[1][0][0]
    l_femur_y_coords = femur_pix_coords[1][0][1]
    r_fem_cond_a_x_coords = knee_pix_coords[0][0][0][0]
    r_fem_cond_a_y_coords = knee_pix_coords[0][0][0][1]
    r_fem_cond_b_x_coords = knee_pix_coords[0][0][1][0]
    r_fem_cond_b_y_coords = knee_pix_coords[0][0][1][1]
    r_tib_plat_a_x_coords = knee_pix_coords[0][1][0][0]
    r_tib_plat_a_y_coords = knee_pix_coords[0][1][0][1]
    r_tib_plat_b_x_coords = knee_pix_coords[0][1][1][0]
    r_tib_plat_b_y_coords = knee_pix_coords[0][1][1][1]
    r_cent_knee_x_coords = knee_pix_coords[0][2][0][0]
    r_cent_knee_y_coords = knee_pix_coords[0][2][0][1]
    l_fem_cond_a_x_coords = knee_pix_coords[1][0][0][0]
    l_fem_cond_a_y_coords = knee_pix_coords[1][0][0][1]
    l_fem_cond_b_x_coords = knee_pix_coords[1][0][1][0]
    l_fem_cond_b_y_coords = knee_pix_coords[1][0][1][1]
    l_tib_plat_a_x_coords = knee_pix_coords[1][1][0][0]
    l_tib_plat_a_y_coords = knee_pix_coords[1][1][0][1]
    l_tib_plat_b_x_coords = knee_pix_coords[1][1][1][0]
    l_tib_plat_b_y_coords = knee_pix_coords[1][1][1][1]
    l_cent_knee_x_coords = knee_pix_coords[1][2][0][0]
    l_cent_knee_y_coords = knee_pix_coords[1][2][0][1]
    r_ankle_x_coords = ankle_pix_coords[0][0]
    r_ankle_y_coords = ankle_pix_coords[0][1]
    l_ankle_x_coords = ankle_pix_coords[1][0]
    l_ankle_y_coords = ankle_pix_coords[1][1]

    # Physical points coordinates
    r_femur_x_phys_point = femur_phys_points[0][0]
    r_femur_y_phys_point = femur_phys_points[0][1]
    l_femur_x_phys_point = femur_phys_points[1][0]
    l_femur_y_phys_point = femur_phys_points[1][1]
    r_fem_cond_a_x_phys_point = knee_phys_points[0][0][0][0]
    r_fem_cond_a_y_phys_point = knee_phys_points[0][0][0][1]
    r_fem_cond_b_x_phys_point = knee_phys_points[0][0][1][0]
    r_fem_cond_b_y_phys_point = knee_phys_points[0][0][1][1]
    r_tib_plat_a_x_phys_point = knee_phys_points[0][1][0][0]
    r_tib_plat_a_y_phys_point = knee_phys_points[0][1][0][1]
    r_tib_plat_b_x_phys_point = knee_phys_points[0][1][1][0]
    r_tib_plat_b_y_phys_point = knee_phys_points[0][1][1][1]
    r_cent_knee_x_phys_point = knee_phys_points[0][2][0]
    r_cent_knee_y_phys_point = knee_phys_points[0][2][1]
    l_fem_cond_a_x_phys_point = knee_phys_points[1][0][0][0]
    l_fem_cond_a_y_phys_point = knee_phys_points[1][0][0][1]
    l_fem_cond_b_x_phys_point = knee_phys_points[1][0][1][0]
    l_fem_cond_b_y_phys_point = knee_phys_points[1][0][1][1]
    l_tib_plat_a_x_phys_point = knee_phys_points[1][1][0][0]
    l_tib_plat_a_y_phys_point = knee_phys_points[1][1][0][1]
    l_tib_plat_b_x_phys_point = knee_phys_points[1][1][1][0]
    l_tib_plat_b_y_phys_point = knee_phys_points[1][1][1][1]
    l_cent_knee_x_phys_point = knee_phys_points[1][2][0]
    l_cent_knee_y_phys_point = knee_phys_points[1][2][1]
    r_ankle_x_phys_point = ankle_phys_points[0][0]
    r_ankle_y_phys_point = ankle_phys_points[0][1]
    l_ankle_x_phys_point = ankle_phys_points[1][0]
    l_ankle_y_phys_point = ankle_phys_points[1][1]

    # Pack
    x_pix_coordinates = [r_femur_x_coords, l_femur_x_coords,
                         r_fem_cond_a_x_coords, r_fem_cond_b_x_coords, r_tib_plat_a_x_coords, r_tib_plat_b_x_coords,
                         r_cent_knee_x_coords,
                         l_fem_cond_a_x_coords, l_fem_cond_b_x_coords, l_tib_plat_a_x_coords, l_tib_plat_b_x_coords,
                         l_cent_knee_x_coords,
                         r_ankle_x_coords, l_ankle_x_coords]
    y_pix_coordinates = [r_femur_y_coords, l_femur_y_coords,
                         r_fem_cond_a_y_coords, r_fem_cond_b_y_coords, r_tib_plat_a_y_coords, r_tib_plat_b_y_coords,
                         r_cent_knee_y_coords,
                         l_fem_cond_a_y_coords, l_fem_cond_b_y_coords, l_tib_plat_a_y_coords, l_tib_plat_b_y_coords,
                         l_cent_knee_y_coords,
                         r_ankle_y_coords, l_ankle_y_coords]
    x_phys_points = [r_femur_x_phys_point, l_femur_x_phys_point,
                     r_fem_cond_a_x_phys_point, r_fem_cond_b_x_phys_point, r_tib_plat_a_x_phys_point, r_tib_plat_b_x_phys_point,
                     r_cent_knee_x_phys_point,
                     l_fem_cond_a_x_phys_point, l_fem_cond_b_x_phys_point, l_tib_plat_a_x_phys_point, l_tib_plat_b_x_phys_point,
                     l_cent_knee_x_phys_point,
                     r_ankle_x_phys_point, l_ankle_x_phys_point]
    y_phys_points = [r_femur_y_phys_point, l_femur_y_phys_point,
                     r_fem_cond_a_y_phys_point, r_fem_cond_b_y_phys_point, r_tib_plat_a_y_phys_point, r_tib_plat_b_y_phys_point,
                     r_cent_knee_y_phys_point,
                     l_fem_cond_a_y_phys_point, l_fem_cond_b_y_phys_point, l_tib_plat_a_y_phys_point, l_tib_plat_b_y_phys_point,
                     l_cent_knee_y_phys_point,
                     r_ankle_y_phys_point, l_ankle_y_phys_point]

    leg_side = ['Right', 'Left',
                'Right', 'Right', 'Right', 'Right', 'Right',
                'Left', 'Left', 'Left', 'Left', 'Left',
                'Right', 'Left']
    joint = ['Femur', 'Femur',
             'Knee', 'Knee', 'Knee', 'Knee', 'Knee',
             'Knee', 'Knee', 'Knee', 'Knee', 'Knee',
             'Ankle', 'Ankle']
    landmark = ['HF', 'HF',
                'R-FC', 'L-FC', 'R-TP', 'L-TP', 'CK',
                'R-FC', 'L-FC', 'R-TP', 'L-TP', 'CK',
                'AM', 'AM']

    # Create dataframe
    columns = ['Joint', 'Leg side', 'Landmark', 'Pix X', 'Pix Y', 'Phys. points X', 'Phys. points Y']
    results_df = pd.DataFrame(columns=columns)
    results_df['Joint'] = joint
    results_df['Leg side'] = leg_side
    results_df['Landmark'] = landmark
    results_df['Pix X'] = x_pix_coordinates
    results_df['Pix Y'] = y_pix_coordinates
    results_df['Phys. points X'] = x_phys_points
    results_df['Phys. points Y'] = y_phys_points

    # Save results
    raw_id = file_name.split('.')[0]
    csv_id = raw_id + '_FLL_{}_coordinates.csv'.format(detection_method)
    results_df.to_csv(os.path.join(save_dir, csv_id), index=False)

    return results_df


# Save malalignment measurements
def save_malalignment_results(set_right_leg: list, set_left_leg: list,
                              file_name: str, detection_method: str, save_path: Path):
    # Create DF
    metrics = ['MAD [mm]', 'mLDFA [°]', 'mMPTA [°]', 'HKA [°]', 'JLO [°]', 'aHKA [°]', 'CPAK']
    columns = ['Metrics', 'Right leg', 'Left leg']
    results_df = pd.DataFrame(columns=columns)
    results_df['Metrics'] = metrics
    results_df['Right leg'] = set_right_leg
    results_df['Left leg'] = set_left_leg

    # Save results
    raw_id = file_name.split('.')[0]
    csv_id = raw_id + '_{}_Malalignment.csv'.format(detection_method)
    results_df.to_csv(os.path.join(save_path, csv_id), index=False)

    return results_df
