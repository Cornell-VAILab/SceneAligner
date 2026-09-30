"""2D images of the demo (RGB): density map, its alignment with the floorplan, and cameras on the floorplan."""
import cv2
import numpy as np

ORANGE = (230, 100, 0)  # reliable correspondences


def draw_solid_points(img, points, color=ORANGE, radius=10):
    canvas = img.copy()
    for x, y in np.round(points).astype(int):
        cv2.circle(canvas, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    return canvas


def draw_alignment(density_img, plan_img, M, points):
    """Density map warped by M over the floorplan, with points of the density map (e.g., reliable correspondences)."""
    H, W = plan_img.shape[:2]
    canvas = cv2.addWeighted(plan_img, 0.6, cv2.warpAffine(density_img, M, (W, H)), 0.4, 0)
    return draw_solid_points(canvas, cv2.transform(points.reshape(-1, 1, 2).astype(np.float32), M).reshape(-1, 2))


def draw_camera_frustum(img, center, forward_vec, color, length=25, width=15):
    """Camera on the floorplan: filled trapezoid with a black outline."""
    if np.linalg.norm(forward_vec) < 1e-6:  # looking straight up or down
        return img
    forward_vec = forward_vec / np.linalg.norm(forward_vec)
    right_vec = np.array([-forward_vec[1], forward_vec[0]])
    corners = [center + forward_vec * length * a + right_vec * width * b / 2 for a, b in ((0.4, -0.4), (1, -1), (1, 1), (0.4, 0.4))]
    pts = [np.array(corners, dtype=np.int32)]
    cv2.fillPoly(img, pts, color, lineType=cv2.LINE_AA)
    cv2.polylines(img, pts, isClosed=True, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    return img


def draw_scene_cameras(plan_img, cam_to_world, color):
    """Cameras of a floorplan-aligned scene (see get_floorplan_aligned_scene) drawn on the floorplan."""
    for pose in cam_to_world:
        center = (np.array([pose[0, 3], -pose[1, 3]]) + 0.5) * plan_img.shape[0]  # floorplan frame -> pixels
        draw_camera_frustum(plan_img, center, np.array([pose[0, 2], -pose[1, 2]]), color)
    return plan_img
