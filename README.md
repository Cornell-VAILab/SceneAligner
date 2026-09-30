# [NeurIPS'26] SceneAligner: 3D-Grounded Floorplan Localization in the Wild

### [Paper](https://arxiv.org/abs/2605.22581) | [Project Page](https://Cornell-VAILab.github.io/SceneAligner)

![teaser](./assets/teaser.png)
> Given a collection of in-the-wild images and a rasterized floorplan, SceneAligner reconstructs a gravity-aligned 3D point cloud from the images and globally aligns this reconstruction to the 2D floorplan map, thereby localizing the images within the floorplan. As illustrated above, our approach successfully aligns images capturing large-scale 3D environments, including exterior scenes (Doddabasappa Temple, left) and interior spaces (Église Saint-Martin d’Agonac, right).

## Installation

```bash
conda create -n scenealigner python=3.11
conda activate scenealigner
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Optional, to render images of the 3D viewer without a browser (`--render`):

```bash
pip install playwright && playwright install chromium
```

## Demo

The example is the Friedrichswerder Church in Berlin from the [C3 dataset](https://c3po-correspondence.github.io/):
62 interior and 69 exterior photos, which share one floorplan.

```bash
python demo.py --images examples/S221/interior examples/S221/exterior --floorplan examples/S221/floorplan.jpg --output outputs/S221
```

Each directory is a photo collection, reconstructed and aligned to the floorplan independently; collections without
visual overlap thus end up in a common frame. Collections of more than 150 photos are split into several
reconstructions. For each reconstruction, the demo saves the floorplan-aligned 3D scene (`.npz`), the density map and
its alignment with the reliable correspondences, and it draws the cameras on the floorplan (`cameras.png`). It then
starts the 3D viewer at http://localhost:8080. With `--share`, the viewer is also served at a public viser share URL,
valid for 24 hours.

```bash
python -m scenealigner.viewer outputs/S221                           # interactive viewer
python -m scenealigner.viewer outputs/S221 --render first_frame.png  # image of the first-frame view
```

## Release
Code and data will be updated later this week.

## Citation
If you find our work useful for your research, please cite our paper:

````BibTeX
@inproceedings{cho2026scenealigner,
  title={SceneAligner: 3D-Grounded Floorplan Localization in the Wild},
  author={Junhyeong Cho and Ruojin Cai and Hadar Averbuch-Elor},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
````