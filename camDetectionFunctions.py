import math
import cv2
import numpy as np
from typing import NamedTuple


class Point(NamedTuple):
    x: float
    y: float


def get_distance(p1: Point, p2: Point) -> float:
    """
    Calculate the Euclidean distance between two points in a 2D plane.

    Parameters:
    p1 (Point): A point having coordinates (x, y).
    p2 (Point): Another point having coordinates (x, y).

    Returns:
    float: The Euclidean distance between point p1 and point p2.

    Assumes that both p1 and p2 are objects with 'x' and 'y' attributes,
    representing their respective coordinates on the 2D plane.

    Note:
    This function requires the math module to be imported.

    Example:
    >>> p1 = Point(1, 2)
    >>> p2 = Point(4, 6)
    >>> get_distance(p1, p2)
    5.0
    """
    return math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)


def get_face_orientation(face_landmarks, img_w, img_h):
    # Initialize lists to store 2D and 3D points
    face_2d = []
    face_3d = []

    for idx, lm in enumerate(face_landmarks.landmark):
        # Non so perché fa 'sta cosa
        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
            x, y = int(lm.x * img_w), int(lm.y * img_h)

            # Get 2D coordinates
            face_2d.append([x, y])

            # Get 3D coordinates
            face_3d.append([x, y, lm.z])

    # Convert to NumPy array
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    # Camera matrix
    focal_length = 1 * img_w

    cam_matrix = np.array(
        [
            [focal_length, 0, img_h / 2],
            [0, focal_length, img_w / 2],
            [0, 0, 1],
        ]
    )

    # Distortion parameters
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    # Solve PnP
    # This function call is where the magic happens. It uses the camera matrix and the distortion parameters to solve the Perspective-n-Point (PnP) problem. The PnP problem involves finding the position and orientation (rotation and translation) of a 3D object in space, given a set of 3D points on the object (face_3d), their corresponding 2D projections on the image (face_2d), and the camera parameters. In this case, it's likely trying to determine the position and orientation of a face relative to the camera.
    success, rot_vec, trans_vec = cv2.solvePnP(
        face_3d, face_2d, cam_matrix, dist_matrix
    )

    # Get rotational matrix
    rmat, jac = cv2.Rodrigues(rot_vec)

    # Get angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    # Convert angles to degrees
    x = angles[0] * (360)
    y = angles[1] * (360)
    z = angles[2] * (360)

    return x, y, z


def get_eye_aperture_ratio(face_landmarks):
    # Landmarks for eye width calculation
    LElandmarksW = [
        face_landmarks.landmark[33],
        face_landmarks.landmark[133],
    ]  # Corrected indices for left eye horizontal
    RElandmarksW = [
        face_landmarks.landmark[362],
        face_landmarks.landmark[263],
    ]  # Corrected indices for right eye horizontal

    # Landmarks for eye height calculation
    LElandmarksH = [
        face_landmarks.landmark[159],
        face_landmarks.landmark[145],
    ]  # Corrected indices for left eye vertical
    RElandmarksH = [
        face_landmarks.landmark[386],
        face_landmarks.landmark[374],
    ]  # Corrected indices for right eye vertical

    # Calculate distances
    LEw = get_distance(LElandmarksW[0], LElandmarksW[1])
    REw = get_distance(RElandmarksW[0], RElandmarksW[1])
    LEh = get_distance(LElandmarksH[0], LElandmarksH[1])
    REh = get_distance(RElandmarksH[0], RElandmarksH[1])

    # Calculate ratios
    LERatio = LEh / LEw if LEw > 0 else 0
    RERatio = REh / REw if REw > 0 else 0

    return (LERatio, RERatio)


def get_mouth_aperture_ratio(face_landmarks):
    # Landmarks for upper to lower lip distance
    MlandmarksH = [face_landmarks.landmark[13], face_landmarks.landmark[14]]

    # Landmarks for mouth width
    MlandmarksW = [face_landmarks.landmark[78], face_landmarks.landmark[366]]

    # Get distances
    mh = get_distance(MlandmarksH[0], MlandmarksH[1])
    mw = get_distance(MlandmarksW[0], MlandmarksW[1])

    # Calculate ratios
    Mratio = mh / mw if mw > 0 else 0

    return Mratio
