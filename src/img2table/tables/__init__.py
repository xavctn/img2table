from collections import defaultdict
from typing import Any, Callable, Optional

import cv2
import numpy as np


def threshold_dark_areas(img: np.ndarray, char_length: Optional[float]) -> np.ndarray:
    """
    Threshold image by differentiating areas with light and dark backgrounds
    :param img: image array
    :param char_length: average character length
    :return: threshold image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # If image is mainly black, revert the image
    if np.mean(gray) <= 127:
        gray = 255 - gray

    thresh_kernel = int(char_length) // 2 * 2 + 1

    # Threshold original image
    t_sauvola = cv2.ximgproc.niBlackThreshold(gray, 255, cv2.THRESH_BINARY_INV, thresh_kernel, 0.2,
                                              binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA)
    thresh = 255 * (gray <= t_sauvola).astype(np.uint8)
    binary_thresh = None

    # Mask on areas with dark background
    blur_size = min(255, int(2 * char_length) // 2 * 2 + 1)
    blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    mask = cv2.inRange(blur, 0, 100)

    # Identify dark areas
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)

    # For each dark area, use binary threshold instead of regular threshold
    for idx, row in enumerate(stats):
        # Get statistics
        x, y, w, h, area = row

        if idx == 0:
            continue

        if area / (w * h) >= 0.5 and min(w, h) >= char_length and max(w, h) >= 5 * char_length:
            if binary_thresh is None:
                # Threshold binary image
                bin_t_sauvola = cv2.ximgproc.niBlackThreshold(255 - gray, 255, cv2.THRESH_BINARY_INV, thresh_kernel,
                                                              0.2,
                                                              binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA)
                binary_thresh = 255 * (255 - gray <= bin_t_sauvola).astype(np.uint8)
            thresh[y:y+h, x:x+w] = binary_thresh[y:y+h, x:x+w]

    return thresh


def cluster_items(items: list[Any], clustering_func: Callable) -> list[list[Any]]:
    """
    Cluster items based on a function
    :param items: list of items
    :param clustering_func: clustering function
    :return: list of list of items based on clustering function
    """
    # Create clusters based on clustering function between items
    clusters = []
    for i in range(len(items)):
        for j in range(i, len(items)):
            # Check if both items corresponds according to the clustering function
            corresponds = clustering_func(items[i], items[j]) or (items[i] == items[j])

            # If both items correspond, find matching clusters or create a new one
            if corresponds:
                matching_clusters = [idx for idx, cl in enumerate(clusters) if {i, j}.intersection(cl)]
                if matching_clusters:
                    remaining_clusters = [cl for idx, cl in enumerate(clusters) if idx not in matching_clusters]
                    new_cluster = {i, j}.union(*[cl for idx, cl in enumerate(clusters) if idx in matching_clusters])
                    clusters = [*remaining_clusters, new_cluster]
                else:
                    clusters.append({i, j})

    return [[items[idx] for idx in c] for c in clusters]


class Node:
    def __init__(self, key: Any) -> None:
        self.key = key
        self.parent = self
        self.size = 1


class UnionFind(dict):
    def find(self, key: Any) -> Node:
        node = self.get(key, None)
        if node is None:
            node = self[key] = Node(key)
        else:
            while node.parent != node:
                # walk up & perform path compression
                node.parent, node = node.parent.parent, node.parent
        return node

    def union(self, key_a: Any, key_b: Any) -> None:
        node_a = self.find(key_a)
        node_b = self.find(key_b)
        if node_a != node_b:  # disjoint? -> join!
            if node_a.size < node_b.size:
                node_a.parent = node_b
                node_b.size += node_a.size
            else:
                node_b.parent = node_a
                node_a.size += node_b.size


def find_components(edges: list[list[Any]]) -> list[list[Any]]:
    forest = UnionFind()

    for edge in edges:
        forest.union(*(edge if len(edge) > 1 else list(edge) * 2))

    result = defaultdict(list)
    for key in forest:
        root = forest.find(key)
        result[root.key].append(key)

    return list(result.values())
