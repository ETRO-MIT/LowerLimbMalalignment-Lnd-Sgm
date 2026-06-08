# Custom functions
from Functions.Geometrics import *


# Calculate the Mechanical Axis Deviation (MAD)
def get_MAD(head_femur_point, mid_ankle_point, center_knee_point):
    m_mad, k_mad = get_line_equation(head_femur_point, mid_ankle_point)
    MAD = get_distance_point_to_line(center_knee_point, m_mad, k_mad)

    return MAD


# Calculate the mechanic Lateral Distal Femoral Angle (mLDFA)
def get_mLDFA(head_femur_point, center_knee_point, femur_condyle_line):
    m_mech_axes, k_mech_axes = get_line_equation(head_femur_point, center_knee_point)
    m_femur_cond_line, k_femur_cond = get_line_equation(femur_condyle_line[0],
                                                        femur_condyle_line[1])
    theta = (get_angle_between_lines(m_femur_cond_line, m_mech_axes))
    if theta > 0:
        mLDFA = 180 - theta
    else:
        mLDFA = abs(theta)

    return mLDFA


# Calculate the mechanical Medial Proximal Tibial Angle (mMPTA)
def get_mMPTA(ankle_mid_point, center_knee_point, tibial_plateau_line):
    m_mech_axes, k_mech_axes = get_line_equation(center_knee_point, ankle_mid_point)
    m_tibial_plat_line, k_tibial_plat_line = get_line_equation(tibial_plateau_line[0],
                                                               tibial_plateau_line[1])
    theta = (get_angle_between_lines(m_tibial_plat_line, m_mech_axes))
    if theta > 0:
        mMPTA = 180 - theta
    else:
        mMPTA = abs(theta)

    return mMPTA


# Calculate the Femoral Valgus Angle (FVA)
def get_FVA(center_knee_point, diaphyse_mid_point, head_femur_point):
    m_mech_axes, k_mech_axes = get_line_equation(center_knee_point, head_femur_point)
    m_anat_axes, k_anat_axes = get_line_equation(center_knee_point, diaphyse_mid_point)
    FVA = abs(get_angle_between_lines(m_mech_axes, m_anat_axes))

    return FVA


# Calculate the hip-knee-ankle (HKA) angle
def get_HKA(femur_point, knee_point, ankle_point):
    m_femur_axis, k_femur_axis = get_line_equation(femur_point, knee_point)
    m_tibia_axis, k_tibia_axis = get_line_equation(knee_point, ankle_point)
    HKA = get_angle_between_lines(m_femur_axis, m_tibia_axis)

    return HKA
