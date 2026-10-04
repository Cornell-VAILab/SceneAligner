"""3D scene reconstruction and gravity estimation from photos (Sec. 3.1).

pi3 reconstructs the photos in the frame of the first camera. GeoCalib predicts a gravity direction per photo, and
the spherical medoid of these directions gives the gravity direction of the scene.
"""
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from geocalib import GeoCalib
from huggingface_hub import hf_hub_download
from pi3.models.pi3 import Pi3
from pi3.utils.geometry import homogenize_points, se3_inverse
from PIL import Image
from PIL.ImageOps import exif_transpose
from torchvision import transforms as TF

CHECKPOINT = "ckpts/pi3-FT_weight.pt"  # pi3 fine-tuned by MegaDepth-X
MEGADEPTH_X_REVISION = "1c7dd2819846a4a51c10defae4f8c3f907af2b6e"
MAX_LIMIT = 150  # maximum number of photos per 3D reconstruction
GEOCALIB_SEED = 1  # GeoCalib is randomized: seeded per reconstruction


def load_models(checkpoint=None):
    """pi3 (fine-tuned on Internet photos by MegaDepth-X) and GeoCalib."""
    if checkpoint is None:
        checkpoint = hf_hub_download("y-u-a-n-l-i/MegaDepth-X", CHECKPOINT, repo_type="dataset", revision=MEGADEPTH_X_REVISION)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = Pi3()
    model.load_state_dict(state_dict.get("model", state_dict))
    return model.eval().to("cuda"), GeoCalib(weights="pinhole").to("cuda")


def load_and_resize_plan(plan_path, plan_size=512):
    """Resizes a floorplan to fit plan_size x plan_size and centers it on white.

    Returns the image and (scale, off_x, off_y), which map the original pixels to the resized ones. The EXIF
    orientation is not applied, as in the C3 annotations of floorplans.
    """
    pil_plan = Image.open(plan_path)
    if str(plan_path).lower().endswith(".gif"):
        pil_plan.seek(pil_plan.n_frames // 2)
    pil_plan = pil_plan.convert("RGB")
    scale = plan_size / max(pil_plan.size)
    new_w, new_h = int(round(pil_plan.width * scale)), int(round(pil_plan.height * scale))
    off_x, off_y = (plan_size - new_w) // 2, (plan_size - new_h) // 2
    final_plan_img = Image.new("RGB", (plan_size, plan_size), (255, 255, 255))
    final_plan_img.paste(pil_plan.resize((new_w, new_h), Image.Resampling.BICUBIC), (off_x, off_y))
    return final_plan_img, (scale, off_x, off_y)


def get_resized_size(width, height, target_size=518):
    """Photo size after resizing: the longer side is 518 px, the other one a multiple of 14 (patch size)."""
    if width >= height:
        return target_size, round(height * (target_size / width) / 14) * 14
    return round(width * (target_size / height) / 14) * 14, target_size


def load_and_preprocess_images(image_paths, target_size=518):
    """Loads photos (EXIF orientation applied), resizes them and pads them with white to 518 x 518.

    Returns a (S, 3, 518, 518) tensor in [0, 1] and the (width, height) of the photos inside the padded images.
    """
    images, image_sizes = [], []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            img = exif_transpose(img)
        if img.mode == "RGBA":  # transparent pixels become white
            img = Image.alpha_composite(Image.new("RGBA", img.size, (255, 255, 255, 255)), img)
        img = img.convert("RGB")
        width, height = get_resized_size(*img.size, target_size)
        img = TF.ToTensor()(img.resize((width, height), Image.Resampling.BICUBIC))
        pad_w, pad_h = target_size - width, target_size - height
        images.append(F.pad(img, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=1.0))
        image_sizes.append((width, height))
    return torch.stack(images), np.array(image_sizes)


def split_into_chunks(image_paths, max_limit=MAX_LIMIT):
    """Collections of more than 150 photos are shuffled and split into ceil(N / 150) chunks of nearly equal size."""
    image_paths = list(image_paths)
    num_chunks = math.ceil(len(image_paths) / max_limit)
    if num_chunks > 1:
        random.shuffle(image_paths)  # avoids a spatial bias of the file order
    chunk_size = math.ceil(len(image_paths) / num_chunks)
    return [image_paths[i:i + chunk_size] for i in range(0, len(image_paths), chunk_size)]


def compute_robust_gravity_medoid(vectors):
    """Spherical medoid: the unit vector minimizing the sum of angles to all the others (robust to outliers)."""
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    angles = np.arccos(np.clip(vectors @ vectors.T, -1.0, 1.0))
    return vectors[np.argmin(angles.sum(axis=1))]


@torch.no_grad()
def reconstruct(image_paths, model, geocalib):
    """Reconstructs photos in the frame of the first camera and estimates the gravity direction in this frame.

    Returns points_cam0 (S, H, W, 3), conf (S, H, W), extrinsics (S, 4, 4) (camera-to-camera-0 poses, OpenCV
    convention), est_gravity_cam (3,), preprocessed_images (S, H, W, 3) uint8 and image_sizes (S, 2).
    """
    device = next(model.parameters()).device
    images_tensor, image_sizes = load_and_preprocess_images(image_paths)
    images_tensor = images_tensor.to(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.amp.autocast("cuda", dtype=dtype):
        predictions = model(images_tensor[None])  # one collection: (1, S, 3, H, W)

    # Cameras relative to camera 0: T_ci->c0 = T_w->c0 @ T_ci->w
    raw_poses = predictions["camera_poses"]  # camera-to-world, (1, S, 4, 4)
    relative_poses = se3_inverse(raw_poses[:, 0:1]) @ raw_poses
    points_cam0 = torch.einsum("bnij, bnhwj -> bnhwi", relative_poses, homogenize_points(predictions["local_points"]))
    points_cam0 = points_cam0[0, ..., :3].cpu().numpy()
    extrinsics = relative_poses[0].cpu().numpy()
    conf = torch.sigmoid(predictions["conf"][0, ..., 0]).cpu().numpy()

    # GeoCalib predicts the up direction of each photo: negate it (gravity, OpenCV convention), rotate it to camera 0
    torch.manual_seed(GEOCALIB_SEED)
    gravities_cam0 = []
    for i, image_path in enumerate(image_paths):
        with Image.open(image_path) as img:
            geo_img = TF.ToTensor()(exif_transpose(img).convert("RGB")).to(device)
        g_up_local = geocalib.calibrate(geo_img)["gravity"].vec3d.squeeze().cpu().numpy()
        gravities_cam0.append(extrinsics[i, :3, :3] @ -g_up_local)

    return {
        "points_cam0": points_cam0,
        "conf": conf,
        "extrinsics": extrinsics,
        "est_gravity_cam": compute_robust_gravity_medoid(np.stack(gravities_cam0)),
        "preprocessed_images": (images_tensor * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy(),
        "image_sizes": image_sizes,
    }
