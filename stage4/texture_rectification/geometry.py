"""Local depth-plane fitting and line-grid scoring for 256-pixel scenes."""
from itertools import product

import cv2
import numpy as np

SIZE = 144
FEATURES = ("spacing", "dispersion", "skew", "disconnect")
PATCHES = tuple((x, y, size) for size, origins in
                ((112, (0, 72, 144)), (160, (0, 96)))
                for x, y in product(origins, repeat=2))


def fit_plane(depth, intrinsics, x, y, size):
    """Back-project axial depths, trim 20% of residuals twice, and fit a plane."""
    y_grid, x_grid = np.mgrid[y:y + size:5, x:x + size:5]
    z = depth[y_grid, x_grid]
    valid = np.isfinite(z) & (z > 0)
    if valid.mean() < 0.9:
        return None
    pixels = np.column_stack((x_grid[valid], y_grid[valid], np.ones(valid.sum())))
    points = (pixels @ np.linalg.inv(intrinsics).T) * z[valid, None]
    active = np.arange(len(points))
    scale = np.median(z[valid])
    for _ in range(3):
        center = points[active].mean(axis=0)
        _, _, axes = np.linalg.svd(points[active] - center, full_matrices=False)
        normal = axes[-1]
        errors = np.abs((points - center) @ normal) / scale
        active = np.argsort(errors)[:int(0.8 * len(errors))]
    if normal @ center < 0:
        normal = -normal
    if np.quantile(errors, 0.85) > 0.025 or normal[2] < 0.35:
        return None
    return normal, float(normal @ center), scale


def rectify(gray, depth, intrinsics, patch, plane):
    """Return an image and in-plane mask after front-facing perspective correction."""
    x, y, size = patch
    normal, distance, scale = plane
    horizontal = np.array([1.0, 0, 0]) - normal[0] * normal
    horizontal /= np.linalg.norm(horizontal)
    rotation = np.stack((horizontal, np.cross(normal, horizontal), normal))
    homography = rotation @ np.linalg.inv(intrinsics)
    corners = np.float32([[[x, y], [x + size - 1, y],
                           [x + size - 1, y + size - 1], [x, y + size - 1]]])
    if np.any(corners[0] @ homography[2, :2] + homography[2, 2] < 0.2):
        return None
    projected = cv2.perspectiveTransform(corners, homography)[0]
    extent = np.ptp(projected, axis=0)
    if extent.min() / extent.max() < 0.3:
        return None
    factor = (SIZE - 5) / extent.max()
    shift = (SIZE - factor * extent) / 2 - factor * projected.min(axis=0)
    normalize = np.array([[factor, 0, shift[0]], [0, factor, shift[1]], [0, 0, 1]])
    warp = normalize @ homography

    # Remove depth outliers before evaluating texture support.
    yy, xx = np.mgrid[y:y + size, x:x + size]
    normal_rays = (homography[2, 0] * xx + homography[2, 1] * yy + homography[2, 2])
    in_plane = np.abs(depth[y:y + size, x:x + size] * normal_rays - distance) <= 0.025 * scale
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[y:y + size, x:x + size] = in_plane.astype(np.uint8)
    warped = cv2.warpPerspective(gray, warp, (SIZE, SIZE))
    valid = cv2.warpPerspective(mask, warp, (SIZE, SIZE), flags=cv2.INTER_NEAREST)
    valid = cv2.erode(valid, np.ones((5, 5), dtype=np.uint8))
    return warped, valid, in_plane


def line_segments(detector, gray, mask):
    detected = detector.detect(gray)[0]
    if detected is None:
        return np.empty((0, 2, 2))
    lines = detected.reshape(-1, 2, 2)
    length = np.linalg.norm(lines[:, 1] - lines[:, 0], axis=1)
    lines = lines[length >= 10]
    fractions = np.linspace(0.08, 0.92, 9)[:, None, None]
    samples = np.rint(lines[None, :, 0] + fractions * (lines[:, 1] - lines[:, 0])).astype(int)
    samples = np.clip(samples, 0, SIZE - 1)
    return lines[mask[samples[..., 1], samples[..., 0]].mean(axis=0) >= 0.88]


def axial_difference(one, two):
    return np.abs((one - two + np.pi / 2) % np.pi - np.pi / 2)


def family_spacing(lines, lengths, theta):
    normal = np.array([-np.sin(theta), np.cos(theta)])
    offsets = lines.mean(axis=1) @ normal
    order = np.argsort(offsets)
    groups = np.split(order, np.flatnonzero(np.diff(offsets[order]) > 3) + 1)
    positions = np.array([np.average(offsets[g], weights=lengths[g]) for g in groups])
    if len(positions) < 3 or np.ptp(positions) < 16:
        return None
    typical_gap = np.median(np.diff(positions))
    low, high = max(5, 0.65 * typical_gap), min(np.ptp(positions) / 2, 1.5 * typical_gap)
    if high < low:
        return None
    periods = np.linspace(low, high, 64)
    phases = positions[:, None] / periods
    circular = np.exp(2j * np.pi * phases).mean(axis=0)
    phase_offset = np.angle(circular) / (2 * np.pi)
    cycles = phases - phase_offset
    residual = np.abs(cycles - np.rint(cycles)).mean(axis=0)
    slots = np.rint(cycles).astype(int)
    occupied = np.sum(np.diff(slots, axis=0) > 0, axis=0) + 1
    gaps = slots[-1] - slots[0] + 1
    scores = residual + 0.15 * (1 - occupied / gaps) + 0.2 * (1 - occupied / len(positions))
    best = np.argmin(scores)
    return float(scores[best]), len(positions)


def connection_error(first, second):
    """Fraction of nearby crossings lacking support from both line segments."""
    p, q = first[:, None, 0], second[None, :, 0]
    u, v = first[:, None, 1] - p, second[None, :, 1] - q

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    determinant = cross(u, v)
    safe = np.abs(determinant) > 1e-5
    t = cross(q - p, v) / np.where(safe, determinant, np.inf)
    s = cross(q - p, u) / np.where(safe, determinant, np.inf)
    extension = np.maximum.reduce((-t, t - 1, np.zeros_like(t))) * np.linalg.norm(u, axis=-1)
    extension = np.maximum(extension, np.maximum.reduce((-s, s - 1, np.zeros_like(s)))
                           * np.linalg.norm(v, axis=-1))
    nearby = safe & (extension <= 14)
    if not nearby.any():
        return 1.0
    return float((extension[nearby] > 4).mean())


def lattice(detector, gray, mask):
    lines = line_segments(detector, gray, mask)
    if len(lines) < 6:
        return None
    vectors = lines[:, 1] - lines[:, 0]
    lengths = np.linalg.norm(vectors, axis=1)
    angles = np.arctan2(vectors[:, 1], vectors[:, 0]) % np.pi
    centers = np.arange(36) * np.pi / 36
    distances = axial_difference(angles[:, None], centers)
    weights = (np.maximum(1 - distances / np.deg2rad(17), 0) * lengths[:, None]).sum(axis=0)
    first = weights.argmax()
    allowed = axial_difference(centers, centers[first]) >= np.deg2rad(42)
    second = np.where(allowed, weights, -1).argmax()
    members = distances[:, [first, second]] < np.deg2rad(20)
    statistics = []
    for j in range(2):
        selected = members[:, j]
        if selected.sum() < 3:
            return None
        theta = np.angle(np.sum(lengths[selected] * np.exp(2j * angles[selected]))) / 2
        dispersion = np.average(axial_difference(angles[selected], theta), weights=lengths[selected])
        spacing = family_spacing(lines[selected], lengths[selected], theta)
        if spacing is None:
            return None
        error, offsets = spacing
        statistics.append((theta, dispersion, error, offsets))
    one, two = statistics
    skew = np.pi / 2 - axial_difference(one[0], two[0])
    scoring = np.array([(one[2] + two[2]) / 2, np.rad2deg((one[1] + two[1]) / 2),
                        np.rad2deg(skew), connection_error(lines[members[:, 0]], lines[members[:, 1]])])
    support = min(one[3], two[3])
    return scoring, support


def extract(gray, depth, intrinsics, detector):
    """Extract matched raw and rectified scores on depth-supported windows."""
    valid_planes, raw_grids, rectified_grids = 0, 0, 0
    matched = []
    raw_only = []
    full_mask = np.ones((SIZE, SIZE), dtype=np.uint8)
    for patch in PATCHES:
        x, y, size = patch
        raw = cv2.resize(gray[y:y + size, x:x + size], (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        raw_score = lattice(detector, raw, full_mask)
        if raw_score is not None:
            raw_grids += 1
            raw_only.append(raw_score)
        plane = fit_plane(depth, intrinsics, *patch)
        if plane is None:
            continue
        rectified = rectify(gray, depth, intrinsics, patch, plane)
        if rectified is None:
            continue
        valid_planes += 1
        warped, warped_mask, in_plane = rectified
        corrected_score = lattice(detector, warped, warped_mask)
        if corrected_score is None:
            continue
        rectified_grids += 1
        raw_mask = cv2.resize(in_plane.astype(np.uint8), (SIZE, SIZE),
                              interpolation=cv2.INTER_NEAREST)
        raw_mask = cv2.erode(raw_mask, np.ones((5, 5), dtype=np.uint8),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0)
        matched_raw = lattice(detector, raw, raw_mask)
        if matched_raw is not None:
            matched.append((matched_raw[0], corrected_score[0],
                            min(matched_raw[1], corrected_score[1])))
    return (valid_planes, raw_grids, rectified_grids, len(matched)), raw_only, matched


def disagreement(scores):
    """Equal weight spacing and angular errors in fixed, interpretable units."""
    return float(np.dot(scores[:3], [1 / 0.15, 1 / 8, 1 / 20]) / 3)
