# Arrange the knee estimations ("Segmentation" and "Heatmap") so that the functions to save the coordinates and the
# angles can work
def arrange_knee_estimations(set_coordinates_512: list, set_coordinates_fll: list, set_phys_points: list):
    # Put the coordinates in the correct format: 512x512 coordinates
    results_512 = [set_coordinates_512[0][0], set_coordinates_512[0][1],
                   set_coordinates_512[1][0], set_coordinates_512[1][1],
                   set_coordinates_512[2][0], set_coordinates_512[2][1],
                   set_coordinates_512[3][0], set_coordinates_512[3][1],
                   set_coordinates_512[4][0], set_coordinates_512[4][1]]

    # Put the coordinates in the correct format: FLL pixel coordinates
    condyle_line_fll_coords = [[set_coordinates_fll[0][0], set_coordinates_fll[0][1]],
                        [set_coordinates_fll[1][0], set_coordinates_fll[1][1]]]
    tibial_line_fll_coords = [[set_coordinates_fll[2][0], set_coordinates_fll[2][1]],
                       [set_coordinates_fll[3][0], set_coordinates_fll[3][1]]]
    center_knee_fll_coords = [[set_coordinates_fll[4][0], set_coordinates_fll[4][1]]]
    result_fll_pix_coordinates = [condyle_line_fll_coords, tibial_line_fll_coords, center_knee_fll_coords]

    # Put the coordinates in the correct format: FLL physical points
    condyle_line_fll_phys_points = [[set_phys_points[0][0], set_phys_points[0][1]],
                               [set_phys_points[1][0], set_phys_points[1][1]]]
    tibial_line_fll_phys_points = [[set_phys_points[2][0], set_phys_points[2][1]],
                              [set_phys_points[3][0], set_phys_points[3][1]]]
    center_knee_fll_phys_points = [set_phys_points[4][0], set_phys_points[4][1]]
    results_fll_phys_points = [condyle_line_fll_phys_points, tibial_line_fll_phys_points, center_knee_fll_phys_points]

    return results_512, result_fll_pix_coordinates, results_fll_phys_points


# Arrange the ankle estimations ("Segmentation" and "Heatmap") so that the functions to save the coordinates and the
# angles can work
def arrange_ankle_estimations(set_coordinates_512: list, set_coordinates_fll: list, set_phys_points: list):
    # Set points in the correct format
    ankle_line = [[set_coordinates_512[0][0], set_coordinates_512[0][1]],
                  [set_coordinates_512[1][0], set_coordinates_512[1][1]]]

    # Get mid-point
    fll_mid_point_ankle_pix = [(set_coordinates_fll[0][0] + set_coordinates_fll[1][0]) / 2,
                               (set_coordinates_fll[0][1] + set_coordinates_fll[1][1]) / 2]
    fll_mid_point_ankle_phys_point = [(set_phys_points[0][0] + set_phys_points[1][0]) / 2,
                                      (set_phys_points[0][1] + set_phys_points[1][1]) / 2]

    return ankle_line, fll_mid_point_ankle_pix, fll_mid_point_ankle_phys_point
