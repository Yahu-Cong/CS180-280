"""Exercise the geometry on a known plane, a wrong normal, and a deformed grid."""
import json

import cv2
import numpy as np

from geometry import SIZE, disagreement, fit_plane, lattice, rectify


def run():
    K = np.array([[245.0, 0, 127.5], [0, 245.0, 127.5], [0, 0, 1]])
    angles = np.deg2rad([25, -12])
    normal = np.array([np.sin(angles[0]), np.sin(angles[1]),
                       np.sqrt(1 - np.square(np.sin(angles)).sum())])
    horizontal = np.array([1.0, 0, 0]) - normal[0] * normal
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.cross(normal, horizontal)
    y, x = np.mgrid[:256, :256]
    rays = np.stack((x, y, np.ones_like(x)), axis=2) @ np.linalg.inv(K).T
    depth = 2 / (rays @ normal)
    points = rays * depth[..., None]
    u, v = points @ horizontal, points @ vertical

    def paint(one, two):
        dark = (np.abs((one / 0.25 + 0.5) % 1 - 0.5) < 0.065)
        dark |= (np.abs((two / 0.25 + 0.5) % 1 - 0.5) < 0.065)
        return cv2.GaussianBlur(np.where(dark, 35, 226).astype(np.uint8), (5, 5), 0.8)

    regular = paint(u, v)
    distorted = paint(u + 0.34 * u**2 + 0.35 * u * v, v + 0.29 * v**2)
    wrong = 2 / (rays @ np.array([-0.30, 0.16, np.sqrt(1 - 0.30**2 - 0.16**2)]))
    patch = (32, 32, 192)
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    scores = {}
    crop = cv2.resize(regular[32:224, 32:224], (SIZE, SIZE))
    scores["regular_rgb"] = disagreement(lattice(detector, crop, np.ones_like(crop))[0])
    for key, image, plane_depth in (("regular_correct_plane", regular, depth),
                                    ("regular_wrong_plane", regular, wrong),
                                    ("deformed_correct_plane", distorted, depth)):
        plane = fit_plane(plane_depth, K, *patch)
        assert plane is not None
        warped, mask, _ = rectify(image, plane_depth, K, patch, plane)
        scores[key] = disagreement(lattice(detector, warped, mask)[0])
    assert scores["regular_correct_plane"] < scores["regular_rgb"]
    assert scores["regular_correct_plane"] < scores["regular_wrong_plane"]
    assert scores["regular_correct_plane"] < scores["deformed_correct_plane"]
    return scores


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
