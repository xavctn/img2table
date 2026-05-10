import cv2
import numpy as np
from sewar import ssim

from img2table.document.rotation import (
    fix_rotation_image,
    get_connected_components,
    get_relevant_angles,
    rotate_img_with_border,
)


def test_get_connected_components() -> None:
    img = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)
    assert img is not None

    cc, _ref_height, _thresh = get_connected_components(img=img)

    assert len(cc) == 98


def test_get_relevant_angles() -> None:
    centroids = [
        [35.8676, 5473.6768],
        [45.4648, 8734.32],
        [476.386, 98.437],
        [9834.4648, 468.47],
        [746.746, 7348.43],
        [846.462, 8474.48],
        [2983.846, 94483.46],
        [1093.46, 8473.46],
        [3676.77, 84783.64],
    ]

    result = get_relevant_angles(centroids=np.array(centroids), ref_height=1000, n_max=5)

    assert len(result) == 5


def test_fix_rotation_image() -> None:
    def crop_to_orig_img(img: np.ndarray, orig_img: np.ndarray) -> np.ndarray:
        # Get original dimensions
        orig_height, orig_width = orig_img.shape[:2]

        # Get center of img
        center = (img.shape[0] // 2, img.shape[1] // 2)
        # Crop img around centre
        return img[
            center[0] - orig_height // 2 : center[0] + orig_height // 2 + 1,
            center[1] - orig_width // 2 : center[1] + orig_width // 2 + 1,
        ]

    img = cv2.imread("test_data/test.png")
    assert img is not None

    similarities = []
    for angle in range(-30, 30, 3):
        # Create test image by rotating it
        test_img = rotate_img_with_border(img=img.copy(), angle=angle)
        result = crop_to_orig_img(img=fix_rotation_image(img=test_img)[0], orig_img=img)

        # Compute similarity between original image and result
        similarities.append(ssim(GT=img, P=result)[0])

    assert np.mean(similarities) >= 0.85
