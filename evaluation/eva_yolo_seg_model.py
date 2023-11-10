import os
import shutil
import sys
from pathlib import Path

import cv2
import hcaptcha_challenger as solver
from hcaptcha_challenger.onnx.modelhub import ModelHub
from hcaptcha_challenger.onnx.yolo import YOLOv8Seg
from tqdm import tqdm

solver.install(upgrade=True)

# Initialize model index
modelhub = ModelHub.from_github_repo()
modelhub.parse_objects()

db_dir = Path(__file__).parent.parent.joinpath("database2309")
input_dirname = "click_on_the_star_with_a_texture_of_bricks_default"

# Select model
model_name = "star_with_a_texture_of_bricks_2309_yolov8s-seg.onnx"
classes = ["star-bricks"]


def yolov8_segment(images_dir: Path, output_dir: Path):
    session = modelhub.match_net(model_name)
    yoloseg = YOLOv8Seg.from_pluggable_model(session, classes)

    # Initialize progress bar
    desc_in = f'"{images_dir.parent.name}/{images_dir.name}"'
    with tqdm(total=len(os.listdir(images_dir)), desc=f"Labeling | {desc_in}") as progress:
        for image_name in os.listdir(images_dir):
            image_path = images_dir.joinpath(image_name)
            if not image_path.is_file():
                progress.total -= 1
                continue
            # Find all the circles in the picture
            yoloseg(image_path, shape_type="point")

            # Draw a bounding box and mask region for all circles
            img = cv2.imread(str(image_path))
            combined_img = yoloseg.draw_masks(img, mask_alpha=0.5)
            output_path = output_dir.joinpath(image_path.name)
            cv2.imwrite(str(output_path), combined_img)

            progress.update(1)

    if "win32" in sys.platform and "PYTEST_RUN_CONFIG" not in os.environ:
        os.startfile(output_dir)
    print(f">> View at {output_dir}")


def demo():
    images_dir = db_dir.joinpath(input_dirname)

    output_dir = Path(__file__).parent.joinpath(
        "yolo_mocker", "figs-star-bricks-seg-out", images_dir.name
    )
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    yolov8_segment(images_dir, output_dir)


if __name__ == "__main__":
    demo()
