"""Demo: photo collections and their floorplan -> floorplan-aligned 3D scene.

    python demo.py --images examples/S221/interior examples/S221/exterior --floorplan examples/S221/floorplan.jpg
"""
import argparse
import glob
import os
import random

import numpy as np
from PIL import Image

from scenealigner import viewer
from scenealigner.inference import MODEL_REPO, align_to_floorplan, get_floorplan_aligned_scene, load_model
from scenealigner.preprocess import load_and_resize_plan, load_models, reconstruct, split_into_chunks
from scenealigner.visualization import draw_alignment, draw_scene_cameras, draw_solid_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True, help="one directory per photo collection")
    parser.add_argument("--floorplan", required=True)
    parser.add_argument("--output", default="outputs/demo")
    parser.add_argument("--checkpoint_dir", default=MODEL_REPO, help="LoRA layers")
    parser.add_argument("--dinov3_weights", default=None, help="Meta's DINOv3 ViT-B/16 weights")
    parser.add_argument("--recon_checkpoint", default=None, help="local copy of the pi3 weights")
    parser.add_argument("--max_points", type=int, default=2_000_000, help="displayed points per reconstruction")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--render", action="store_true", help="saves first_frame.png instead of starting the viewer")
    parser.add_argument("--share", action="store_true", help="also serves the viewer at a public viser share URL (24 hours)")
    args = parser.parse_args()

    recon_model, geocalib = load_models(args.recon_checkpoint)
    model = load_model(args.checkpoint_dir, args.dinov3_weights)
    plan_img, _ = load_and_resize_plan(args.floorplan)
    os.makedirs(args.output, exist_ok=True)
    plan_img.save(os.path.join(args.output, "floorplan.png"))
    plan = np.array(plan_img)

    scenes, cameras_img = {}, plan.copy()
    for directory, color in zip(args.images, viewer.COLORS * len(args.images)):
        # Each collection is reconstructed and aligned independently (in chunks of at most 150 photos)
        random.seed(42)
        np.random.seed(42)
        chunks = split_into_chunks(sorted(glob.glob(os.path.join(directory, "*"))))
        for k, image_paths in enumerate(chunks):
            name = os.path.basename(os.path.normpath(directory)) + (f"_{k}" if len(chunks) > 1 else "")
            recon = reconstruct(image_paths, recon_model, geocalib)
            alignment = align_to_floorplan(recon, plan_img, model)
            scene = get_floorplan_aligned_scene(recon, alignment)
            if len(scene["points"]) > args.max_points:  # own random generator: the alignments do not depend on it
                keep = np.sort(np.random.default_rng(0).choice(len(scene["points"]), args.max_points, replace=False))
                scene["points"], scene["colors"] = scene["points"][keep], scene["colors"][keep]
            scene["thumbnails"], scene["image_sizes"] = recon["preprocessed_images"][:, ::4, ::4], recon["image_sizes"]
            scene["color"] = np.array(color)  # of the cameras
            np.savez_compressed(os.path.join(args.output, f"{name}.npz"), **scene)
            scenes[name] = scene

            # Density map, aligned density map with the reliable correspondences, cameras on the floorplan
            density = np.array(alignment["density_img"])
            Image.fromarray(draw_solid_points(density, alignment["src_px"])).save(os.path.join(args.output, f"{name}_density.png"))
            Image.fromarray(draw_alignment(density, plan, alignment["M"], alignment["src_px"])).save(os.path.join(args.output, f"{name}_alignment.png"))
            draw_scene_cameras(cameras_img, scene["cam_to_world"], color)
            print(f"{name}: {len(image_paths)} photos aligned")
    Image.fromarray(cameras_img).save(os.path.join(args.output, "cameras.png"))

    server = viewer.show(scenes, plan, args.port)
    if args.render:
        viewer.render(server, os.path.join(args.output, "first_frame.png"))
    else:
        if args.share:
            server.request_share_url()
        server.sleep_forever()


if __name__ == "__main__":
    main()
