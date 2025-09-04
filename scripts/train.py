import argparse
import os
import random
import numpy as np
import torch
from omegaconf import OmegaConf

from pianobind.utils.logger import Logger
from pianobind.utils.utils import (
    load_conf,
    merge_conf,
    get_root_dir,
    update_conf_with_cli_params,
)

from pianobind.models.audio_text import PianoBind_AudioText
from pianobind.models.midi_text import PianoBind_MIDIText
from pianobind.models.trimodal import PianoBind_Trimodal

from pianobind.trainers.pianobind_trainer import PianoBindTrainer

from pianobind.datasets.piast_at import PIAST_AT_Dataset
from pianobind.datasets.piast_yt import PIAST_YT_Dataset

def set_environment(seed: int, device_num: str):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = device_num

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description="Train a PianoBind model")

    parser.add_argument(
        "--experiment_id",
        type=str,
        help="experiment id under which checkpoint was saved",
        default=None,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["audio_text", "midi_text", "trimodal"],
        default="audio_text",
        help="Type of model to train",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="path to base config file",
        default=os.path.join(get_root_dir(), "configs", "training.yaml"),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["piast_yt", "piast_at", "piast_joint"],
        default="piast_joint",
        help="Dataset to use",
    )
    parser.add_argument(
        "--device_num", 
        type=str, 
        default="1")
    
    parser.add_argument(
        "--seed",
        type=int, 
        default=42)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    params = parse_args()

    if params.experiment_id is None:
        base_conf = load_conf(params.config_path)

        dataset_config_map = {
            "piast_at": PIAST_AT_Dataset.config_path(),
            "piast_yt": PIAST_YT_Dataset.config_path(),
            "piast_joint": "configs/datasets/piast_joint.yaml",
        }
        if params.dataset not in dataset_config_map:
            raise ValueError(f"{params.dataset} dataset not supported")
        dataset_conf_path = os.path.join(base_conf.env.base_dir, dataset_config_map[params.dataset])
        
        
        model_config_map = {
            "audio_text": PianoBind_AudioText.config_path(),
            "midi_text": PianoBind_MIDIText.config_path(),
            "trimodal": PianoBind_Trimodal.config_path(),
        }

        if params.model_type not in model_config_map:
            raise ValueError(f"{params.model_type} model is not supported")
        model_conf_path = os.path.join(base_conf.env.base_dir, model_config_map[params.model_type])


        config = merge_conf(params.config_path, dataset_conf_path, model_conf_path)
        update_conf_with_cli_params(params, config)
    else:
        config = OmegaConf.load(
            "./save/experiments/{}/config.yaml".format(params.experiment_id)
        )

    logger = Logger(config)
    set_environment(params.seed, params.device_num)

    trainer = PianoBindTrainer(config, logger)
    print("# of trainable parameters:", trainer.count_parameters())

    trainer.train()
