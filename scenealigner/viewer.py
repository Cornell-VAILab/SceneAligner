"""Interactive 3D viewer of floorplan-aligned scenes (viser).

    python -m scenealigner.viewer outputs/S221                           # interactive
    python -m scenealigner.viewer outputs/S221 --render first_frame.png  # image of the first-frame view
"""
import argparse
import glob
import os
import time

import numpy as np
import viser
import viser.transforms as vtf
from PIL import Image

# First-frame view in the floorplan frame (floorplan on the unit square at z = 0, Z up)
POSITION, LOOK_AT, FOV = (-1.3571, -2.2206, 2.4501), (0.0, 0.0, 0.0), np.deg2rad(22.9)
COLORS = [(123, 88, 244), (228, 119, 177), (203, 39, 39), (88, 231, 244)]  # cameras of each photo collection


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def show(scenes, plan_img, port=8080):
    """Starts a viewer of the scenes {name: scene} on the floorplan. Returns the viser server."""
    server = viser.ViserServer(port=port)
    server.scene.set_up_direction("+z")
    server.scene.set_background_image(np.full((8, 8, 3), 255, dtype=np.uint8))
    server.initial_camera.position, server.initial_camera.look_at = POSITION, LOOK_AT
    server.initial_camera.up, server.initial_camera.fov = (0.0, 0.0, 1.0), FOV
    server.scene.add_image("floorplan", plan_img, 1.0, 1.0, wxyz=(0.0, 1.0, 0.0, 0.0))  # image top at y = 0.5

    clouds, frustums = [], []
    for name, scene in scenes.items():
        clouds.append(server.scene.add_point_cloud(f"{name}/points", scene["points"], scene["colors"],
                                                   point_size=0.002, point_shape="circle", precision="float32"))
        for i, pose in enumerate(scene["cam_to_world"]):
            w, h = scene["image_sizes"][i] // 4
            top, left = (np.array(scene["thumbnails"][i].shape[:2]) - (h, w)) // 2
            frustums.append(server.scene.add_camera_frustum(
                f"{name}/cameras/{i}", fov=np.deg2rad(50), aspect=w / h, scale=0.03, thickness=0.002, color=scene["color"],
                image=scene["thumbnails"][i][top:top + h, left:left + w], wxyz=vtf.SO3.from_matrix(pose[:3, :3]).wxyz,
                position=pose[:3, 3]))

    point_size = server.gui.add_slider("Point size", min=0.0005, max=0.01, step=0.0005, initial_value=0.002)
    show_cameras = server.gui.add_checkbox("Cameras", initial_value=True)
    turntable = server.gui.add_button("Turntable")

    @point_size.on_update
    def _(_):
        for cloud in clouds:
            cloud.point_size = point_size.value

    @show_cameras.on_update
    def _(_):
        for frustum in frustums:
            frustum.visible = show_cameras.value

    @turntable.on_click
    def _(event):
        camera = event.client.camera
        position, look_at = np.array(camera.position), np.array(camera.look_at)
        for angle in np.linspace(0, 2 * np.pi, 80):  # 4 seconds at 20 frames per second
            with event.client.atomic():
                camera.position, camera.look_at = rotation_z(angle) @ position, rotation_z(angle) @ look_at
            time.sleep(1 / 20)

    return server


def render(server, path, width=1280, height=720):
    """Saves an image of the first-frame view, rendered by a headless browser (Playwright) if installed, or else by
    the first browser that opens the viewer."""
    url = f"http://localhost:{server.get_port()}"
    try:
        from playwright.sync_api import sync_playwright

        playwright = sync_playwright().start()
        playwright.chromium.launch().new_page(viewport={"width": width, "height": height}).goto(url)
    except ImportError:
        playwright = None
        print(f"Open {url} in a browser to render the first frame.")
    while not server.get_clients():
        time.sleep(0.1)
    client = next(iter(server.get_clients().values()))
    time.sleep(3.0)  # scene transfer to the browser

    forward = np.array(LOOK_AT) - POSITION
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, (0.0, 0.0, 1.0))
    right /= np.linalg.norm(right)
    rotation = np.stack([right, np.cross(forward, right), forward], axis=1)  # camera axes: right, down, forward
    image = client.get_render(height=height, width=width, wxyz=vtf.SO3.from_matrix(rotation).wxyz,
                              position=POSITION, fov=FOV, transport_format="png").astype(np.float32)
    alpha = image[..., 3:] / 255
    Image.fromarray((image[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)).save(path)
    if playwright is not None:
        playwright.stop()


def load(directory):
    """Scenes {name: scene} and the floorplan saved by demo.py."""
    scenes = {os.path.basename(p)[:-4]: dict(np.load(p)) for p in sorted(glob.glob(os.path.join(directory, "*.npz")))}
    return scenes, np.array(Image.open(os.path.join(directory, "floorplan.png")).convert("RGB"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="output directory of demo.py")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--render", default=None, help="saves an image of the first-frame view and exits")
    parser.add_argument("--share", action="store_true", help="also serves the viewer at a public viser share URL (24 hours)")
    args = parser.parse_args()

    server = show(*load(args.directory), port=args.port)
    if args.render:
        render(server, args.render)
    else:
        if args.share:
            server.request_share_url()
        server.sleep_forever()
