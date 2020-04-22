import argparse

import habitat

from train import Trainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )

    return parser.parse_args()

def make_cfg(path):
    config = habitat.get_config(path)

    # In habitat, we have to provide a path to a dataset, even though it won't get used for our purposes
    config.defrost()
    config.DATASET.DATA_PATH = "data/datasets/pointnav/habitat-test-scenes/v1/train/train.json.gz"
    config.DATASET.SCENES_DIR = "data/scene_datasets/"
    config.freeze()

    return config


if __name__ == "__main__":
    args = get_args()
    config = make_cfg(args.config)
    
    trainer = Trainer(config)
    trainer.train()
    print("done!")