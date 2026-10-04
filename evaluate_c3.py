"""Evaluation on the C3 test set: camera pose estimation (Table 1) and correspondence estimation (Table 2).

    python evaluate_c3.py --c3_root <C3>                                  # Ours
    python evaluate_c3.py --c3_root <C3> --checkpoint_dir pre-trained     # pretrained DINOv3
    python evaluate_c3.py --c3_root <C3> --recon_cache <reconstructions>  # Ours, on our cached reconstructions
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from scenealigner.evaluation import compute_metrics, evaluate_alignment, load_ground_truth
from scenealigner.inference import MODEL_REPO, align_to_floorplan, load_model
from scenealigner.preprocess import load_and_resize_plan, load_models, reconstruct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c3_root", required=True, help="C3 dataset, with visual/ and geometric/test/")
    parser.add_argument("--test_csv", default="data/c3/test.csv")
    parser.add_argument("--checkpoint_dir", default=MODEL_REPO, help='LoRA layers, or "pre-trained" for DINOv3')
    parser.add_argument("--dinov3_weights", default=None, help="Meta's DINOv3 ViT-B/16 weights")
    parser.add_argument("--recon_checkpoint", default=None, help="local copy of the pi3 weights")
    parser.add_argument("--recon_cache", default=None, help="our cached reconstructions of the test photos (instead of reconstructing them)")
    parser.add_argument("--output", default="outputs/c3_results.json")
    args = parser.parse_args()

    np.random.seed(42)
    if args.recon_cache is None:
        recon_model, geocalib = load_models(args.recon_checkpoint)
    model = load_model(args.checkpoint_dir, args.dinov3_weights)

    # One reconstruction per chunk of photos, in the order of the CSV
    data = pd.read_csv(args.test_csv)
    scene_results = defaultdict(lambda: {"pred": [], "gt": [], "t_errs": [], "r_errs": []})
    for (scene_id, chunk_idx), rows in tqdm(data.groupby(["scene_id", "chunk_idx"], sort=False)):
        visual_dir = os.path.join(args.c3_root, "visual", rows.scene_name.iloc[0])
        plan_img, plan_transform_params = load_and_resize_plan(os.path.join(visual_dir, rows.plan_path.iloc[0]))
        if args.recon_cache is None:
            recon = reconstruct([os.path.join(visual_dir, p) for p in rows.photo_path], recon_model, geocalib)
        else:
            recon = dict(np.load(os.path.join(args.recon_cache, f"{scene_id}_{chunk_idx:05d}.npz")))
            assert np.array_equal(recon["uid"], rows.uid), "the cache and the CSV list different photos"
        alignment = align_to_floorplan(recon, plan_img, model)
        gt = load_ground_truth(args.c3_root, rows, plan_transform_params)
        pred, target, t_errs, r_errs = evaluate_alignment(recon, alignment, *gt, plan_transform_params)
        res = scene_results[scene_id]
        res["pred"].append(pred)
        res["gt"].append(target)
        res["t_errs"] += t_errs
        res["r_errs"] += r_errs

    metrics = compute_metrics(scene_results)
    for keys in (list(metrics)[:8], list(metrics)[8:]):  # Table 1, Table 2
        print("  ".join(f"{k:>16}" for k in keys))
        print("  ".join(f"{metrics[k]:16.4f}" if k == "RMSE" else f"{metrics[k]:16.2f}" for k in keys))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
