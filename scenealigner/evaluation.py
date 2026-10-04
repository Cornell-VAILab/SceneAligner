"""Evaluation on C3 (Sec. 4.2): ground truth, correspondence accuracy (PCK, RMSE) and camera pose recall."""
import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image

from .inference import EVAL_IMG_RES, get_density_coordinates
from .preprocess import get_resized_size

ANGLE_THRESHOLDS = (5, 10, 20, 30)          # degrees
POSITION_THRESHOLDS = (0.05, 0.1, 0.2)      # relative to the diagonal of the 512 x 512 floorplan
PCK_THRESHOLDS = (0.01, 0.03, 0.05, 0.10, 0.15, 0.30)  # relative to the floorplan resolution


def transform_correspondences(corrs, orientation, w, h):
    """Pixel coordinates (N, 2) of a stored photo of size (w, h) -> coordinates after its EXIF orientation."""
    x, y = corrs[:, 0], corrs[:, 1]
    nx, ny = {1: (x, y), 2: (w - x, y), 3: (w - x, h - y), 4: (x, h - y),
              5: (y, x), 6: (h - y, x), 7: (h - y, w - x), 8: (y, w - x)}[orientation]
    return np.stack([nx, ny], axis=1)


def load_ground_truth(c3_root, rows, plan_transform_params, photo_size=518, plan_size=EVAL_IMG_RES):
    """Ground truth of the photos of one reconstruction (rows of data/c3/test.csv, in order).

    Returns batch_corrs (per photo: corresponding pixels in the preprocessed photo and floorplan, or None) and
    gt_poses (per photo: floorplan-to-camera rotation R and translation t).
    """
    p_scale, p_off_x, p_off_y = plan_transform_params
    batch_corrs, gt_poses = [], []
    for row in rows.itertuples():
        file_name = os.path.join(str(row.uid // 1000), f"{row.uid:06d}.npy")
        plan_coords, photo_coords = np.load(os.path.join(c3_root, "geometric", "test", "correspondences", file_name), allow_pickle=True)
        R_p2c, t_p, _ = np.load(os.path.join(c3_root, "geometric", "test", "camera_poses", file_name), allow_pickle=True)
        with Image.open(os.path.join(c3_root, "visual", row.scene_name, row.photo_path)) as img:
            orientation = img.getexif().get(0x0112) or 1
            width, height = img.size

        # Correspondences in the photo as loaded by pi3 (EXIF orientation, resizing, padding)
        tensor_coords = transform_correspondences(photo_coords, orientation, width, height).astype(np.float32)
        if orientation >= 5:  # these orientations swap width and height
            width, height = height, width
        new_width, new_height = get_resized_size(width, height, photo_size)
        tensor_coords[:, 0] *= new_width / width
        tensor_coords[:, 1] *= new_height / height
        tensor_coords += ((photo_size - new_width) // 2, (photo_size - new_height) // 2)
        # ... and in the resized floorplan
        plan_coords = plan_coords.astype(np.float32)
        plan_coords *= p_scale
        plan_coords += (p_off_x, p_off_y)

        valid = ((tensor_coords >= 0) & (tensor_coords <= photo_size - 1)).all(axis=1) & \
                ((plan_coords >= 0) & (plan_coords <= plan_size - 1)).all(axis=1)
        batch_corrs.append({"tensor_coords": tensor_coords[valid], "plan_coords": plan_coords[valid]} if valid.any() else None)
        gt_poses.append((np.asarray(R_p2c, dtype=np.float64), np.asarray(t_p, dtype=np.float64).reshape(3)))
    return batch_corrs, gt_poses


def process_gt_correspondences(corr_data, s_idx, points_cam0, proj_params):
    """Maps the annotated pixels of photo s_idx to the density map through their 3D points.

    Returns their density-map pixels and the annotated floorplan pixels.
    """
    tc, pc_gt = corr_data["tensor_coords"], corr_data["plan_coords"]
    S, H, W, _ = points_cam0.shape
    idx_x = np.clip(tc[:, 0].round(), 0, W - 1).astype(int)
    idx_y = np.clip(tc[:, 1].round(), 0, H - 1).astype(int)
    pts_cam0_sparse = points_cam0[s_idx, idx_y, idx_x, :]
    valid = ~np.isnan(pts_cam0_sparse).any(axis=1)
    density_map_px = get_density_coordinates(pts_cam0_sparse[valid] @ proj_params["R_rect"].T, proj_params)
    return density_map_px, pc_gt[valid]


def get_camera_center_on_density_px(extrinsics, proj_params):
    """Camera centers (S, 2) on the density map."""
    return get_density_coordinates(extrinsics[:, :3, 3] @ proj_params["R_rect"].T, proj_params)


def get_gt_camera_center_on_plan_px(R_gt, t_gt, plan_transform_params):
    """Ground-truth camera center on the resized floorplan."""
    scale, off_x, off_z = plan_transform_params
    cc = -R_gt.T @ t_gt  # camera center in the floorplan frame
    return np.array([cc[0] * scale + off_x, cc[2] * scale + off_z])


def compute_pose_error_analytic(M_pred, extrinsic_i, R_gt, proj_params, pred_cam_center_den_px, gt_cam_center_plan_px,
                                img_size=(EVAL_IMG_RES, EVAL_IMG_RES)):
    """Positional error (relative to the floorplan diagonal) and angular error (degrees) of one camera on the floorplan."""
    # Position: camera centers on the floorplan
    pred_cam_center_plan = M_pred @ np.array([pred_cam_center_den_px[0], pred_cam_center_den_px[1], 1.0])
    t_error_norm = float(np.linalg.norm(pred_cam_center_plan - gt_cam_center_plan_px) / np.linalg.norm(img_size))

    # Angle: viewing directions (camera z-axes) projected on the floorplan
    forward_gt_plan = R_gt.T[:, 2]  # in the floorplan frame
    forward_pred_ground = proj_params["R_rect"] @ (extrinsic_i[:3, :3] @ np.array([0.0, 0.0, 1.0]))  # cam i -> cam 0 -> ground
    forward_pred_plan = M_pred[:2, :2] @ np.array([forward_pred_ground[0], -forward_pred_ground[2]])  # map v-axis points down
    forward_gt_plan_xz = np.array([forward_gt_plan[0], forward_gt_plan[2]])
    forward_gt_plan_xz /= np.linalg.norm(forward_gt_plan_xz)
    forward_pred_plan_xz = forward_pred_plan / np.linalg.norm(forward_pred_plan)
    relative_angle = np.arctan2(*forward_pred_plan_xz) - np.arctan2(*forward_gt_plan_xz)
    r_error_deg = float(np.abs(np.degrees(np.arctan2(np.sin(relative_angle), np.cos(relative_angle)))))
    return t_error_norm, r_error_deg


def evaluate_alignment(recon, alignment, batch_corrs, gt_poses, plan_transform_params):
    """Errors of one reconstruction: predicted and ground-truth floorplan coordinates of the annotated pixels
    (normalized by the floorplan resolution), and the positional and angular errors of the cameras."""
    proj_params, M = alignment["proj_params"], alignment["M"]
    pred_coords, gt_coords = [], []
    for s_idx, corr_data in enumerate(batch_corrs):
        if corr_data is None:
            continue
        density_px_gt, plan_px_gt = process_gt_correspondences(corr_data, s_idx, recon["points_cam0"], proj_params)
        plan_px_pred = cv2.transform(density_px_gt.reshape(-1, 1, 2), M).reshape(-1, 2)
        pred_coords.append(plan_px_pred / EVAL_IMG_RES)
        gt_coords.append(plan_px_gt / EVAL_IMG_RES)

    pred_cam_center_den_px = get_camera_center_on_density_px(recon["extrinsics"], proj_params)
    t_errs, r_errs = [], []
    for i, (R_gt, t_gt) in enumerate(gt_poses):
        gt_center = get_gt_camera_center_on_plan_px(R_gt, t_gt, plan_transform_params)
        t_err, r_err = compute_pose_error_analytic(M, recon["extrinsics"][i], R_gt, proj_params, pred_cam_center_den_px[i], gt_center)
        t_errs.append(t_err)
        r_errs.append(r_err)
    return np.concatenate(pred_coords), np.concatenate(gt_coords), t_errs, r_errs


def compute_metrics(scene_results):
    """Tables 1 and 2: metrics of each scene averaged over the scenes (recalls and PCK in %)."""
    metrics = defaultdict(list)
    for r in scene_results.values():
        r_errs, t_errs = np.array(r["r_errs"]), np.array(r["t_errs"])
        for thr in ANGLE_THRESHOLDS:
            metrics[f"Ang@{thr}"].append(np.mean(r_errs < thr))
        for thr in POSITION_THRESHOLDS:
            metrics[f"Pos@{thr}"].append(np.mean(t_errs < thr))
        metrics["Ang&Pos@(30,0.2)"].append(np.mean((r_errs < 30) & (t_errs < 0.2)))
        diff = torch.tensor(np.concatenate(r["pred"]) - np.concatenate(r["gt"]))
        dist = torch.norm(diff, p=2, dim=-1)
        for thr in PCK_THRESHOLDS:
            metrics[f"PCK@{thr:.0%}"].append((dist <= thr).float().mean().item() * 100)
        metrics["RMSE"].append(torch.sqrt(torch.mean(diff ** 2)).item())
    return {k: float(np.mean(v)) * (100 if k.startswith(("Ang", "Pos")) else 1) for k, v in metrics.items()}
