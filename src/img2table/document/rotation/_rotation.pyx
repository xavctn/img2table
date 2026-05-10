cimport cython
import numpy as np
cimport numpy as cnp
from libc.math cimport M_PI, atan, round as c_round

cnp.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_angles(double[:, ::1] centroids, double ref_height):
    """
    Compute candidate rotation angles from centroid pairs.
    :param centroids: array of centroid coordinates.
    :param ref_height: reference object height.
    :return: array of candidate angles in degrees.
    """
    cdef Py_ssize_t n, i, j, k, max_pairs
    cdef double xi, yi, xj, yj, angle, slope
    cdef double y_threshold = 10.0 * ref_height
    cdef double rad_to_deg = 180.0 / M_PI

    # Sort centroids by y-coordinate
    cdef cnp.ndarray[cnp.float64_t, ndim=2] centroids_array = np.asarray(centroids, dtype=np.float64)
    cdef cnp.ndarray[cnp.intp_t, ndim=1] idx = np.argsort(centroids_array[:, 1])
    cdef cnp.ndarray[cnp.float64_t, ndim=2] c_sorted_array = np.ascontiguousarray(centroids_array[idx], dtype=np.float64)
    cdef double[:, ::1] c_sorted = c_sorted_array

    # Preallocate angles array
    n = c_sorted.shape[0]
    max_pairs = n * (n - 1) // 2
    cdef cnp.ndarray[cnp.float64_t, ndim=1] angles_array = np.empty(max_pairs, dtype=np.float64)
    cdef double[::1] angles = angles_array
    k = 0

    for i in range(n):
        xi, yi = c_sorted[i, 0], c_sorted[i, 1]
        for j in range(i + 1, n):
            xj, yj = c_sorted[j, 0], c_sorted[j, 1]

            # Continue if both elements are not relevant
            if yj - yi > y_threshold:
                break
            if xi == xj:
                continue

            # Compute angle
            slope = c_round((yi - yj) / (xi - xj) * 1000.0) / 1000.0
            angle = atan(slope) * rad_to_deg
            if not -45.0 <= angle <= 45.0:
                angle = -min(angle + 90.0, 90.0 - angle) * (1.0 if angle > 0 else -1.0)
            angles[k] = angle
            k += 1

    return angles_array[:k]
