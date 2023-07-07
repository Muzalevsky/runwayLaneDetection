import logging
import shutil

import click
from project_paths import paths
from tqdm import tqdm

from lane_detection_medium.datasets.file_datasets import DatasetMode
from lane_detection_medium.utils.fs import get_date_string, read_txt, save_txt
from lane_detection_medium.utils.load import get_zebra_id

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__file__)


@click.command()
@click.option(
    "-v", "--dataset-version", required=True, type=str, help="The name of the dataset folder"
)
def main(dataset_version: str):
    src_dpath = paths.yolo_dpath / "data" / dataset_version
    if not src_dpath.exists():
        raise ValueError(f"Dataset '{dataset_version}' does not exist.")

    out_dpath = paths.yolo_dpath / "data" / get_date_string()
    out_dpath.mkdir(parents=True, exist_ok=True)

    for mode in DatasetMode:
        src_mode_dpath = src_dpath / mode.name
        out_mode_dpath = out_dpath / mode.name

        # NOTE: copy all images from folder
        shutil.copytree(src_mode_dpath / "images", out_mode_dpath / "images", dirs_exist_ok=True)

        out_mode_labels_dpath = out_mode_dpath / "labels"
        out_mode_labels_dpath.mkdir(parents=True, exist_ok=True)

        zebra_id = str(get_zebra_id())

        mode_txt_fpaths = sorted(list((src_mode_dpath / "labels").glob("*.txt")))
        stream = tqdm(mode_txt_fpaths, desc=f"{mode.name.capitalize()} txt processing")
        for txt_fpath in stream:
            src_labels = read_txt(txt_fpath)

            # NOTE: "zebra" boxes are excluded
            out_labels = [label for label in src_labels if label[0] != zebra_id]
            if not len(out_labels):
                # NOTE: do not save empty txt files
                continue

            save_txt(out_mode_labels_dpath / txt_fpath.name, out_labels)


if __name__ == "__main__":
    main()
