# coding: utf-8
from collections import defaultdict
from typing import List, Any, Callable, Optional

import cv2
import numpy as np


def threshold_dark_areas(img: np.ndarray, char_length: Optional[float], method: str = "adaptive") -> np.ndarray:
    """
    Threshold image by differentiating areas with light and dark backgrounds
    :param img: image array
    :param char_length: average character length
    :param method: threshold method used
    :return: threshold image
    """
    # If image is mainly black, revert the image
    if np.mean(img) <= 100:
        img = 255 - img

    thresh_kernel = int(char_length) // 2 * 2 + 1

    if method == "adaptive":
        margin = int(char_length)

        # Get threshold on image and binary image
        blur = cv2.GaussianBlur(img, (3, 3), 0)

        # Threshold original image
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, thresh_kernel,
                                       5)
        # Threshold binary image
        binary_thresh = cv2.adaptiveThreshold(255 - blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                              thresh_kernel, 5)
    elif method == "sauvola":
        margin = 0

        # Threshold original image
        t_sauvola = cv2.ximgproc.niBlackThreshold(img, 255, cv2.THRESH_BINARY_INV, thresh_kernel, 0.2,
                                                  binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA)
        thresh = 255 * (img <= t_sauvola).astype(np.uint8)
        # Threshold binary image
        bin_t_sauvola = cv2.ximgproc.niBlackThreshold(255 - img, 255, cv2.THRESH_BINARY_INV, thresh_kernel, 0.2,
                                                      binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA)
        binary_thresh = 255 * (255 - img <= bin_t_sauvola).astype(np.uint8)
    else:
        raise ValueError(f"Unknown method {method}")

    # Mask on areas with dark background
    blur_size = min(255, int(2 * char_length) // 2 * 2 + 1)
    blur = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
    mask = cv2.inRange(blur, 0, 100)

    # Identify dark areas
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)

    # For each dark area, use binary threshold instead of regular threshold
    for idx, row in enumerate(stats):
        # Get statistics
        x, y, w, h, area = row

        if idx == 0:
            # Fix area statistics for the first CC
            area = len(np.where(mask[y:y + h, x:x + w] == 255)[0])

        if area / (w * h) >= 0.6 and min(w, h) >= char_length and max(w, h) >= 4 * char_length:
            thresh[y+margin:y+h-margin, x+margin:x+w-margin] = binary_thresh[y+margin:y+h-margin, x+margin:x+w-margin]

    return thresh


def cluster_items(items: List[Any], clustering_func: Callable) -> List[List[Any]]:
    """
    Cluster items based on a function
    :param items: list of items
    :param clustering_func: clustering function
    :return: list of list of items based on clustering function
    """
    # Create clusters based on clustering function between items
    clusters = list()
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
                    clusters = remaining_clusters + [new_cluster]
                else:
                    clusters.append({i, j})

    return [[items[idx] for idx in c] for c in clusters]


class Node:
    def __init__(self, key):
        self.key = key
        self.parent = self
        self.size = 1


class UnionFind(dict):
    def find(self, key):
        node = self.get(key, None)
        if node is None:
            node = self[key] = Node(key)
        else:
            while node.parent != node:
                # walk up & perform path compression
                node.parent, node = node.parent.parent, node.parent
        return node

    def union(self, key_a, key_b):
        node_a = self.find(key_a)
        node_b = self.find(key_b)
        if node_a != node_b:  # disjoint? -> join!
            if node_a.size < node_b.size:
                node_a.parent = node_b
                node_b.size += node_a.size
            else:
                node_b.parent = node_a
                node_a.size += node_b.size


def find_components(edges):
    forest = UnionFind()

    for edge in edges:
        edge = edge if len(edge) > 1 else list(edge) * 2
        forest.union(*edge)

    result = defaultdict(list)
    for key in forest.keys():
        root = forest.find(key)
        result[root.key].append(key)

    return list(result.values())
