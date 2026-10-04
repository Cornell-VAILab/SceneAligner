"""Floorplan alignment: density map (Sec. 3.2), cross-modal correspondences and 2D similarity transform M
(Sec. 3.3), and the floorplan-aligned 3D scene."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from dinov3.hub.backbones import dinov3_vitb16
from huggingface_hub import hf_hub_download
from peft import PeftModel
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms as T

MODEL_REPO = "jhcho99/SceneAligner"  # our LoRA layers
# DINOv3 ViT-B/16 weights from timm, an ungated copy (https://github.com/facebookresearch/dinov3/issues/145)
DINOV3_TIMM_REPO = "timm/vit_base_patch16_dinov3.lvd1689m"
DINOV3_TIMM_REVISION = "c6a5fb7d12bbd3cf3b0079253141c3332aaed7da"
EVAL_IMG_RES = 512  # resolution of density maps and floorplans
SOFT_ARGMAX_TEMPERATURE = 0.02

norm_transform = T.Compose([T.ToTensor(), T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])


def load_timm_weights(base_model):
    """timm's DINOv3 weights with the parameter names of the DINOv3 code."""
    state_dict = load_file(hf_hub_download(DINOV3_TIMM_REPO, "model.safetensors", revision=DINOV3_TIMM_REVISION))
    state_dict = {k.replace("reg_token", "storage_tokens").replace("gamma_1", "ls1.gamma").replace("gamma_2", "ls2.gamma"): v
                  for k, v in state_dict.items()}
    # Not in timm's file: q/k/v biases and their masks, mask token (all zero), RoPE periods (bfloat16 in Meta's file)
    for name, tensor in base_model.state_dict().items():
        if name.endswith(("qkv.bias", "qkv.bias_mask")) or name == "mask_token":
            state_dict[name] = torch.zeros_like(tensor)
    state_dict["rope_embed.periods"] = base_model.rope_embed.periods.bfloat16()
    return state_dict


def load_model(checkpoint_dir=MODEL_REPO, dinov3_weights=None):
    """DINOv3 ViT-B/16 with our LoRA layers, or the pretrained DINOv3 if checkpoint_dir is "pre-trained".

    dinov3_weights: Meta's weight file (default: timm's copy).
    """
    base_model = dinov3_vitb16(pretrained=False)
    state_dict = load_timm_weights(base_model) if dinov3_weights is None else torch.load(dinov3_weights, map_location="cpu")
    base_model.load_state_dict(state_dict)
    model = base_model if checkpoint_dir == "pre-trained" else PeftModel.from_pretrained(base_model, checkpoint_dir)
    return model.eval().to("cuda")


def reconstruct_projection_geometry(points_cam0, points_conf, est_gravity_cam, conf_threshold_percent=45.0,
                                    xz_slice_val=2.5, density_size=EVAL_IMG_RES, padding=0.1, eps=1e-8):
    """Rotates the points to the gravity-aligned ground frame, filters them and fits the top-down projection.

    Returns the projection parameters and the filtered points (N, 3) in the ground frame.
    """
    # Ground frame (Gram-Schmidt): +Y along gravity, +X close to the x-axis of camera 0
    y_target = est_gravity_cam / (np.linalg.norm(est_gravity_cam) + eps)
    x_ref = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(y_target, x_ref)) > 0.9999:  # gravity parallel to the x-axis
        x_ref = np.array([0.0, 0.0, 1.0])
    z_temp = np.cross(x_ref, y_target)
    z_temp /= np.linalg.norm(z_temp)
    x_target = np.cross(y_target, z_temp)
    x_target /= np.linalg.norm(x_target)
    z_target = np.cross(x_target, y_target)
    z_target /= np.linalg.norm(z_target)
    R_rect = np.stack([x_target, y_target, z_target], axis=1).T  # camera 0 -> ground
    points = (points_cam0 @ R_rect.T).reshape(-1, 3)
    conf = points_conf.reshape(-1)

    # Confidence, height (+Y is down: top 5% and bottom 20% removed) and x/z percentile filters
    points = points[conf > np.percentile(conf, conf_threshold_percent)]
    ceil_cutoff, floor_cutoff = np.percentile(points[:, 1], [5.0, 80.0])
    points = points[(points[:, 1] >= ceil_cutoff) & (points[:, 1] <= floor_cutoff)]
    x_min, x_max = np.percentile(points[:, 0], [xz_slice_val, 100 - xz_slice_val])
    z_min, z_max = np.percentile(points[:, 2], [xz_slice_val, 100 - xz_slice_val])
    points = points[(points[:, 0] >= x_min) & (points[:, 0] <= x_max) & (points[:, 2] >= z_min) & (points[:, 2] <= z_max)]

    # Scale and offsets that center the points in the map, with a margin
    min_x, min_z = points[:, 0].min(), points[:, 2].min()
    width_m, height_m = points[:, 0].max() - min_x, points[:, 2].max() - min_z
    scale = (density_size * (1 - padding * 2)) / max(width_m, height_m)
    proj_params = {"R_rect": R_rect, "min_x": min_x, "min_z": min_z, "scale": scale,
                   "cx": (density_size - width_m * scale) / 2, "cz": (density_size - height_m * scale) / 2,
                   "density_size": density_size}
    return proj_params, points


def get_density_coordinates(points, proj_params):
    """Ground-frame 3D points (N, 3) -> density-map pixel coordinates (N, 2)."""
    u = (points[:, 0] - proj_params["min_x"]) * proj_params["scale"] + proj_params["cx"]
    v = (points[:, 2] - proj_params["min_z"]) * proj_params["scale"] + proj_params["cz"]
    return np.stack([u, proj_params["density_size"] - 1 - v], axis=1).astype(np.float32)


def render_density_map(points, proj_params, gamma_val=0.5):
    """Counts the points per pixel, applies gamma correction and normalizes. Returns an RGB image (dark = dense)."""
    size = proj_params["density_size"]
    coords = get_density_coordinates(points, proj_params).astype(np.int32)
    coords = coords[((coords >= 0) & (coords <= size - 1)).all(axis=1)]
    density_map = np.bincount(coords[:, 1].astype(np.int64) * size + coords[:, 0], minlength=size * size)
    density_map = np.power(density_map.astype(np.float32).reshape(size, size), gamma_val)
    if density_map.max() > 0:
        density_map = (density_map / density_map.max()) * 255.0
    return Image.fromarray(np.clip(255.0 - density_map, 0, 255).astype(np.uint8)).convert("RGB")


def intensity_sampling(density_img, num_sample_points=10000):
    """Samples query points with a probability proportional to the density (darker pixels).

    Returns normalized coordinates in [-1, 1], (1, N, 2), without duplicates.
    """
    img_arr = np.array(density_img.convert("L"))
    H, W = img_arr.shape
    prob_map = 255.0 - img_arr.astype(np.float32)
    prob_map[prob_map < 20] = 0  # background noise
    flat_indices = np.random.choice(H * W, size=num_sample_points, replace=True, p=(prob_map / prob_map.sum()).flatten())
    norm_pts = np.stack([2 * ((flat_indices % W + 0.5) / W) - 1, 2 * ((flat_indices // W + 0.5) / H) - 1], axis=1)
    norm_pts = norm_pts.astype(np.float32)
    _, unique_indices = np.unique(np.round(norm_pts, decimals=3), axis=0, return_index=True)  # ignores float jitter
    return torch.tensor(norm_pts[np.sort(unique_indices)]).unsqueeze(0)


def sample_descriptors(features, grid_points):
    """Bilinear interpolation of features (B, C, H, W) at normalized coordinates (B, N, 2). Returns (B, N, C)."""
    sampled = F.grid_sample(features, grid_points.unsqueeze(1), mode="bilinear", align_corners=False)  # (B, C, 1, N)
    return sampled.squeeze(2).permute(0, 2, 1)


@torch.no_grad()
def estimate_correspondences(model, density_img, plan_img):
    """Reliable correspondences from the density map to the floorplan, in pixels (Sec. 3.3)."""
    device = next(model.parameters()).device
    feature_maps = []
    for img in (density_img, plan_img):
        tokens = model.forward_features(norm_transform(img).unsqueeze(0).to(device))["x_norm_patchtokens"]
        B, N_tokens, C = tokens.shape
        grid_dim = int(np.sqrt(N_tokens))
        feature_maps.append(tokens.permute(0, 2, 1).reshape(B, C, grid_dim, grid_dim))
    map_d, map_p = feature_maps

    # Cosine similarity between query descriptors of the density map and all floorplan patches
    src_pts_norm = intensity_sampling(density_img).to(device)  # (1, N, 2)
    desc_d = F.normalize(sample_descriptors(map_d, src_pts_norm), dim=-1)  # (1, N, C)
    flat_p = F.normalize(map_p.flatten(2).permute(0, 2, 1), dim=-1)  # (1, HW, C)
    sim_matrix = torch.matmul(desc_d, flat_p.transpose(1, 2))  # (1, N, HW)

    # Reliable correspondences: top 50% confidence (peak of the softmax) and mutual nearest neighbors
    prob = F.softmax(sim_matrix / SOFT_ARGMAX_TEMPERATURE, dim=-1)
    conf_vals = prob.max(dim=-1).values.squeeze(0)
    fwd_idx = sim_matrix.argmax(dim=-1).squeeze(0)  # density map -> floorplan
    bwd_idx = sim_matrix.argmax(dim=1).squeeze(0)  # floorplan -> density map
    mask_mnn = bwd_idx[fwd_idx] == torch.arange(len(fwd_idx), device=device)
    mask_conf = conf_vals >= torch.topk(conf_vals, int(len(conf_vals) * 0.5)).values[-1]
    final_mask = mask_conf & mask_mnn
    if final_mask.sum() < 10:  # fallback: confidence only
        final_mask = mask_conf

    # Sub-patch floorplan coordinates: expectation of the patch centers (soft-argmax)
    gy, gx = torch.meshgrid(torch.arange(grid_dim), torch.arange(grid_dim), indexing="ij")
    grid_coords = (torch.stack([gx, gy], dim=-1).float().to(device).reshape(-1, 2) + 0.5) / grid_dim  # [0, 1]
    dst_pts_norm = torch.matmul(prob, grid_coords).squeeze(0)  # (N, 2)
    src_px = (((src_pts_norm.squeeze(0) + 1) / 2)[final_mask].cpu().numpy() * EVAL_IMG_RES).astype(np.float32)
    dst_px = (dst_pts_norm[final_mask].cpu().numpy() * EVAL_IMG_RES).astype(np.float32)
    return src_px, dst_px


def align_to_floorplan(recon, plan_img, model):
    """Density map of a reconstruction, its reliable correspondences and M (density map -> floorplan pixels)."""
    proj_params, points = reconstruct_projection_geometry(recon["points_cam0"], recon["conf"], recon["est_gravity_cam"])
    density_img = render_density_map(points, proj_params)
    src_px, dst_px = estimate_correspondences(model, density_img, plan_img)
    M, _ = cv2.estimateAffinePartial2D(src_px, dst_px, method=cv2.RANSAC, ransacReprojThreshold=0.04 * EVAL_IMG_RES)
    if M is None:
        raise ValueError("RANSAC failed to estimate the similarity transform.")
    return {"proj_params": proj_params, "density_img": density_img, "src_px": src_px, "dst_px": dst_px, "M": M}


def get_floorplan_aligned_scene(recon, alignment, conf_threshold_percent=90.0, y_percentiles=(1.0, 99.0),
                                xz_slice_val=2.5):
    """3D points and cameras in the floorplan frame: horizontal coordinates follow M, heights use the scale of M.

    Frame: Z up, the floorplan on the unit square [-0.5, 0.5]^2 at z = 0 (512 pixels = 1, top edge at y = 0.5).
    The floor, the lower bound of the height filter, is at z = 0. The points are filtered for display by confidence,
    height and x/z percentiles. Cameras are camera-to-world poses (OpenCV convention).
    """
    proj_params, M = alignment["proj_params"], alignment["M"]
    R_rect, res = proj_params["R_rect"], proj_params["density_size"]
    s_M = np.sqrt(np.abs(np.linalg.det(M[:2, :2])))  # scale of M

    # Displayed points: confidence, height and x/z percentile filters in the ground frame (+Y down)
    points = (recon["points_cam0"] @ R_rect.T).reshape(-1, 3)
    conf = recon["conf"].reshape(-1)
    mask = conf > np.percentile(conf, conf_threshold_percent)
    points, colors = points[mask], recon["preprocessed_images"].reshape(-1, 3)[mask]
    y_min, y_floor = np.percentile(points[:, 1], y_percentiles)
    mask = (points[:, 1] >= y_min) & (points[:, 1] <= y_floor)
    points, colors = points[mask], colors[mask]
    x_min, x_max = np.percentile(points[:, 0], [xz_slice_val, 100 - xz_slice_val])
    z_min, z_max = np.percentile(points[:, 2], [xz_slice_val, 100 - xz_slice_val])
    mask = (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    points, colors = points[mask], colors[mask]

    def to_floorplan_frame(p):
        plan_px = cv2.transform(get_density_coordinates(p, proj_params).reshape(-1, 1, 2), M).reshape(-1, 2)
        return np.stack([plan_px[:, 0] / res - 0.5, 0.5 - plan_px[:, 1] / res,
                         (y_floor - p[:, 1]) * proj_params["scale"] * s_M / res], axis=1)

    # Cameras: rotation of M about the vertical axis (the v-axis of the maps points down)
    F_flip = np.diag([1.0, -1.0])
    R2d = F_flip @ (M[:2, :2] / s_M) @ F_flip
    R_yaw = np.array([[R2d[0, 0], 0.0, R2d[0, 1]], [0.0, 1.0, 0.0], [R2d[1, 0], 0.0, R2d[1, 1]]])
    to_z_up = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])  # (x, y down, z) -> (x, z, -y)
    cam_to_world = np.tile(np.eye(4), (len(recon["extrinsics"]), 1, 1))
    cam_to_world[:, :3, :3] = to_z_up @ R_yaw @ R_rect @ recon["extrinsics"][:, :3, :3]
    cam_to_world[:, :3, 3] = to_floorplan_frame(recon["extrinsics"][:, :3, 3] @ R_rect.T)
    return {"points": to_floorplan_frame(points).astype(np.float32), "colors": colors,
            "cam_to_world": cam_to_world.astype(np.float32)}
