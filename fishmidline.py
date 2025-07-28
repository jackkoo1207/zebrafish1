import cv2
import numpy as np
import scipy.ndimage
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmax
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize

from idtrackerai import Blob

SMOOTH_SIGMA = 5


def get_spline(blob: Blob) -> tuple[np.ndarray, list, int]:
    """Get a spline representation of blob's posture.

    Parameters
    ----------
    blob : Blob
        idtracker.ai Blob instance.

    Returns
    -------
    tuple[np.ndarray, list, int]
        scipy.interpolate.splprep output: A tuple, ``(t,c,k)`` containing the vector
        of knots, the B-spline coefficients, and the degree of the spline.
    """
    binary_image = blob.get_bbox_mask()
    skeleton = skeletonize(binary_image)
    midline = np.asarray(np.where(skeleton))[::-1].T

    end_points = _find_end_points(skeleton)
    head = _find_head(binary_image)
    midline = midline[_midline_order(midline, end_points=end_points, head=head)]

    spline_params, _u = interpolate.splprep(midline.T)
    return spline_params


def _find_head(binary_img: np.ndarray) -> np.ndarray:
    contour = (
        cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0][0]
        .squeeze()
        .astype(np.float32)
    )
    smoother = gaussian_filter1d(contour, SMOOTH_SIGMA, mode="wrap", axis=0)
    curv = abs(_curvature(smoother))

    # We find the head of the fish by looking the second maximum
    # of the curvature ( the first one is the tail)
    max_i = argrelmax(curv, mode="wrap")[0]
    b_s = max_i[np.argsort(curv[max_i])]

    if len(b_s) < 2:
        print("Warning, no nose detected in a frame. The maximum is returned instead")
        return contour[b_s[-1]]

    return contour[b_s[-2]]


def _curvature(contour: np.ndarray) -> np.ndarray:
    dx_1, dy_1 = scipy.ndimage.convolve1d(
        contour, [-0.5, 0.0, 0.5], mode="wrap", axis=0
    ).T

    dx_2, dy_2 = scipy.ndimage.convolve1d(
        contour, [1.0, -2.0, 1.0], mode="wrap", axis=0
    ).T

    return (dx_1 * dy_2 - dy_1 * dx_2) / np.power(dx_1 * dx_1 + dy_1 * dy_1, 3 / 2)


def _midline_order(
    midline: np.ndarray, end_points: np.ndarray, head: np.ndarray
) -> list[int]:
    # take into account that there might be more than two end points.
    # We want to order the points from the closest the head to the
    # farthest ignoring any in between end point.
    sorted_indices: list[int] = []
    free_indices: list[int] = list(range(len(midline)))

    distances_to_head = cdist(head[None, :], end_points)
    first_end_point = end_points[distances_to_head.argmin()]
    last_end_point = end_points[distances_to_head.argmax()]

    distances = cdist(first_end_point[None, :], midline[free_indices])
    sorted_indices.append(free_indices.pop(distances.argmin()))

    while free_indices:
        distances = cdist(
            midline[sorted_indices[-1]][None, :], midline[free_indices]
        ) - cdist(last_end_point[None, :], midline[free_indices])
        sorted_indices.append(free_indices.pop(distances.argmin()))
        if (sorted_indices[-1] == last_end_point).all():
            break
    return sorted_indices


def _find_end_points(skel: np.ndarray) -> np.ndarray:
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], np.uint8)
    filtered = cv2.filter2D(skel.astype(np.uint8), -1, kernel, borderType=0)
    return np.asarray(np.where(filtered == 11))[::-1].T
