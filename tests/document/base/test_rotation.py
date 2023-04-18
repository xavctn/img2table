# coding: utf-8
import cv2
import numpy as np
from sewar import ssim

from img2table.document.base.rotation import rotate_img_with_border, fix_rotation_image, get_connected_components, \
    get_relevant_angles, angle_dixon_q_test


def test_get_connected_components():
    img = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)

    cc, ref_height, thresh = get_connected_components(img=img)

    assert len(cc) == 98


def test_get_relevant_angles():
    centroids = [[35.8676, 5473.6768],
                 [45.4648, 8734.32],
                 [476.386, 98.437],
                 [9834.4648, 468.47],
                 [746.746, 7348.43],
                 [846.462, 8474.48],
                 [2983.846, 94483.46],
                 [1093.46, 8473.46],
                 [3676.77, 84783.64]]

    result = get_relevant_angles(centroids=np.array(centroids), ref_height=1000, n_max=5)

    assert len(result) == 5


def test_angle_dixon_q_test():
    result = angle_dixon_q_test(angles=[12.23, 12.78, 12.79, 12.82], confidence=0.9)

    assert round(result, 3) == 12.797


def test_fix_rotation_image():
    def crop_to_orig_img(img, orig_img):
        # Get original dimensions
        orig_height, orig_width = orig_img.shape

        # Get center of img
        center = (img.shape[0] // 2, img.shape[1] // 2)
        # Crop img around centre
        cropped = img[center[0] - orig_height // 2: center[0] + orig_height // 2 + 1,
                      center[1] - orig_width // 2: center[1] + orig_width // 2 + 1]

        return cropped

    img = cv2.imread("test_data/test.png", cv2.IMREAD_GRAYSCALE)

    similarities = list()
    for angle in range(-30, 30, 3):
        # Create test image by rotating it
        test_img = rotate_img_with_border(img=img.copy(), angle=angle)
        result = crop_to_orig_img(img=fix_rotation_image(img=test_img),
                                  orig_img=img)

        # Compute similarity between original image and result
        similarities.append(ssim(GT=img, P=result)[0])

    assert np.mean(similarities) >= 0.85
