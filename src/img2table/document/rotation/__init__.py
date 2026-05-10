import cv2
import numpy as np

from img2table.document.rotation._rotation import compute_angles  # ty:ignore[unresolved-import]


def get_connected_components(img: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Identify connected components in image
    :param img: image array
    :return: list of connected components centroids and thresholded image
    """
    # Thresholding
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Connected components
    _, _, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8, ltype=cv2.CV_32S)

    # Remove connected components with less than 5 pixels
    mask_pixels = stats[:, cv2.CC_STAT_AREA] > 5
    stats = stats[mask_pixels]

    # Compute median width and height
    median_width = np.median(stats[:, cv2.CC_STAT_WIDTH])  # ty: ignore[no-matching-overload]
    median_height = np.median(stats[:, cv2.CC_STAT_HEIGHT])  # ty: ignore[no-matching-overload]

    # Compute bbox area bounds
    upper_bound = 4 * median_width * median_height
    lower_bound = 0.25 * median_width * median_height

    # Filter connected components according to their area
    mask_lower_area = lower_bound < stats[:, cv2.CC_STAT_WIDTH] * stats[:, cv2.CC_STAT_HEIGHT]
    mask_upper_area = upper_bound > stats[:, cv2.CC_STAT_WIDTH] * stats[:, cv2.CC_STAT_HEIGHT]
    mask_area = mask_lower_area & mask_upper_area

    # Filter components based on aspect ratio
    mask_lower_ar = stats[:, cv2.CC_STAT_WIDTH] / stats[:, cv2.CC_STAT_HEIGHT] > 0.5
    mask_upper_ar = stats[:, cv2.CC_STAT_WIDTH] / stats[:, cv2.CC_STAT_HEIGHT] < 2
    mask_ar = mask_lower_ar & mask_upper_ar

    # Create general mask
    mask = mask_area & mask_ar

    # Get centroids from mask
    stats = stats[mask]
    centroids_x = stats[:, cv2.CC_STAT_LEFT] + stats[:, cv2.CC_STAT_WIDTH] / 2
    centroids_y = stats[:, cv2.CC_STAT_TOP] + stats[:, cv2.CC_STAT_HEIGHT] / 2
    filtered_centroids = np.column_stack([centroids_x, centroids_y])

    return filtered_centroids, float(median_height), thresh


def get_relevant_angles(centroids: np.ndarray, ref_height: float, n_max: int = 5) -> list[float]:
    """
    Identify relevant angles from connected components centroids
    :param centroids: array of connected components centroids
    :param ref_height: reference height
    :param n_max: maximum number of returned angles
    :return: list of angle values
    """
    if len(centroids) == 0:
        return [0]

    # Compute angles
    angles = compute_angles(centroids=centroids, ref_height=ref_height)

    if len(angles) == 0:
        return [0]

    # Get n most represented angles
    rounded_angles = np.round(np.asarray(angles, dtype=np.float64), 2)
    unique_angles, counts = np.unique(rounded_angles, return_counts=True)
    idx = np.lexsort((np.abs(unique_angles), -counts))
    idx = idx[:n_max]

    # Keep angles that represent at least 25% of the most represented angle
    max_count = counts[idx[0]]
    idx = idx[counts[idx] >= max_count * 0.25]
    relevant_angles = unique_angles[idx]

    if len(relevant_angles) == 0 or relevant_angles[0] == 0:
        return [0.0]

    return np.sort(relevant_angles).tolist()


def rotate_img(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image by angle
    :param img: image array
    :param angle: rotation angle
    :return: rotated image
    """
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    return cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)


def evaluate_angle(img: np.ndarray, angle: float) -> int:
    """
    Evaluate relevance of angle for image rotation
    :param img: image array
    :param angle: angle
    :return: metric for angle quality
    """
    # Rotate image
    rotated_img = rotate_img(img=img, angle=angle)
    # Apply horizontal projection
    proj = np.sum(rotated_img, 1)
    # Count number of empty rows
    return np.sum((proj[1:] - proj[:-1]) ** 2)


def estimate_skew(angles: list[float], thresh: np.ndarray) -> float:
    """
    Estimate skew from angles
    :param angles: list of angles
    :param thresh: thresholded image
    :return: best angle
    """
    # If there is only one angle, return it
    if len(angles) == 1:
        return angles.pop()

    # Evaluate angles by rotation
    best_angle = None
    best_evaluation = 0
    for angle in angles:
        # Get angle evaluation
        angle_evaluation = evaluate_angle(img=thresh, angle=angle)

        if angle_evaluation > best_evaluation or (
            angle_evaluation == best_evaluation
            and (best_angle is None or abs(angle) < abs(best_angle))
        ):
            best_angle = angle
            best_evaluation = angle_evaluation

    return best_angle if best_angle is not None else 0.0


def rotate_img_with_border(
    img: np.ndarray, angle: float, background_color: tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Rotate an image of the defined angle and add background on border
    :param img: image array
    :param angle: rotation angle
    :param background_color: background color for borders after rotation
    :return: rotated image array
    """
    # Compute image center
    height, width = img.shape[:2]
    image_center = (width // 2, height // 2)

    # Compute rotation matrix
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    # Get rotated image dimension
    bound_w = int(height * abs(rotation_mat[0, 1]) + width * abs(rotation_mat[0, 0]))
    bound_h = int(height * abs(rotation_mat[0, 0]) + width * abs(rotation_mat[0, 1]))

    # Update rotation matrix
    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    # Create rotated image with white background
    return cv2.warpAffine(
        img,
        rotation_mat,
        (bound_w, bound_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background_color,
    )


def fix_rotation_image(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Fix rotation of input image (based on https://www.mdpi.com/2079-9292/9/1/55) by at most 45 degrees
    :param img: image array
    :return: rotated image array and boolean indicating if the image has been rotated
    """
    # Get connected components of the images
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cc_centroids, ref_height, thresh = get_connected_components(img=gray)

    # Check number of centroids
    if len(cc_centroids) < 2:
        return img, False

    # Compute most likely angles from connected components
    angles = get_relevant_angles(centroids=cc_centroids, ref_height=ref_height)
    # Estimate skew
    skew_angle = estimate_skew(angles=angles, thresh=thresh)

    if abs(skew_angle) >= 0.25:
        # Rotate image with borders
        return rotate_img_with_border(img=img, angle=skew_angle), True

    return img, False
