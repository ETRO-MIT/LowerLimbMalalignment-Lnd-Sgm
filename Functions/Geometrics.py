# Python packages
import math


# Calculate slope and intercept from two points
def get_line_equation(P, Q):
    y_dif = Q[1] - P[1]
    x_dif = (Q[0] - P[0]) + 1e-9
    m = (y_dif / x_dif)
    k = P[1] - (m * P[0])

    return m, k


# Calculate distance from a point to a line
def get_distance_point_to_line(point, slope, intercept):
    numerator = abs(intercept + (slope * point[0]) - point[1])
    denominator = math.sqrt(1 + slope ** 2)
    distance = numerator / denominator

    return distance


# Calculate angle between two lines:
def get_angle_between_lines(slope_p1, slope_p2):
    numerator = slope_p1 - slope_p2
    denominator = 1 + (slope_p1 * slope_p2)

    angle = math.degrees(math.atan(numerator / denominator))

    return angle

