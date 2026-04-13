import numpy as np


def ellipse_points_from_lines(
    line1_start, line1_end, line2_start, line2_end, n_points: int = 64
):
    s1 = np.array(line1_start, dtype=float)
    e1 = np.array(line1_end, dtype=float)
    s2 = np.array(line2_start, dtype=float)
    e2 = np.array(line2_end, dtype=float)

    center = (s1 + e1) / 2.0

    axis1 = (e1 - s1) / 2.0
    axis2 = (e2 - s2) / 2.0

    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    cos_t = np.cos(t)[:, np.newaxis]
    sin_t = np.sin(t)[:, np.newaxis]

    points = center + cos_t * axis1 + sin_t * axis2

    return points


def ellipse_bbox_from_annotation(annotation):
    center = annotation.center
    a, b = annotation.semi_axes
    cy, cx = center[-2], center[-1]
    return [
        [cy - a, cx - b],
        [cy - a, cx + b],
        [cy + a, cx + b],
        [cy + a, cx - b],
    ]


def lines_to_napari_data(annotation):
    lines = []
    if annotation.line1 is not None:
        lines.append(np.array([annotation.line1.start, annotation.line1.end]))
    if annotation.line2 is not None:
        lines.append(np.array([annotation.line2.start, annotation.line2.end]))
    return lines
