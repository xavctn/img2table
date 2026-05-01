import cv2
import numpy as np


def threshold_dark_areas(img: np.ndarray, char_length: float) -> np.ndarray:
    """
    Threshold image by differentiating areas with light and dark backgrounds
    :param img: image array
    :param char_length: average character length
    :return: threshold image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # If image is mainly black, revert the image
    if np.mean(gray) <= 127:  # ty: ignore[no-matching-overload]
        gray = 255 - gray

    thresh_kernel = int(char_length) // 2 * 2 + 1

    # Threshold original image
    t_sauvola = cv2.ximgproc.niBlackThreshold(
        gray,
        255,
        cv2.THRESH_BINARY_INV,
        thresh_kernel,
        0.2,
        binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA,
    )
    thresh = 255 * (gray <= t_sauvola).astype(np.uint8)

    # Mask on areas with dark background
    blur_size = min(255, int(2 * char_length) // 2 * 2 + 1)
    blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    mask = cv2.inRange(src=blur, lowerb=0, upperb=100)  # ty:ignore[no-matching-overload]

    # Identify dark areas
    _, _, stats, _ = cv2.connectedComponentsWithStats(image=mask, connectivity=8, ltype=cv2.CV_32S)

    for idx, (x, y, w, h, area) in enumerate(stats):
        if idx == 0:
            continue

        # Filter for significant dark regions (likely text blocks on dark backgrounds)
        if area / (w * h) >= 0.5 and min(w, h) >= char_length and max(w, h) >= 5 * char_length:
            # Extract region of interest with margins
            m_left = min(x, thresh_kernel)
            m_right = min(gray.shape[1] - (x + w), thresh_kernel)
            m_top = min(y, thresh_kernel)
            m_bottom = min(gray.shape[0] - (y + h), thresh_kernel)
            roi_inverted = 255 - gray[y - m_top : y + h + m_bottom, x - m_left : x + w + m_right]

            # Apply Sauvola threshold
            bin_t_sauvola = cv2.ximgproc.niBlackThreshold(
                roi_inverted,
                255,
                cv2.THRESH_BINARY_INV,
                thresh_kernel,
                0.2,
                binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA,
            )
            roi_binary = (roi_inverted <= bin_t_sauvola).astype(np.uint8) * 255

            # Replace in threshold image
            thresh[y : y + h, x : x + w] = roi_binary[m_top : m_top + h, m_left : m_left + w]

    return thresh
