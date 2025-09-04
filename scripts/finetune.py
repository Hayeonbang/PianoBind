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
from pianobind.trainers.pianobind_finetuner import PianoBindFinetuner
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
    parser = argparse.ArgumentParser(description="Finetune a PianoBind model")
    parser.add_argument(
        "-id",
        "--pretrained_experiment_id",
        type=str,
        help="experiment id of the pretrained model",
    )
    
    parser.add_argument(
        "--config_path",
        type=str,
        help="path to base config file",
        default=os.path.join(get_root_dir(), "configs", "finetuning.yaml"),
    )
    
    parser.add_argument(
        "--dataset", 
        type=str,   
        help="name of the dataset", 
        default="piast_at"
    )
    parser.add_argument(
        "--device_num", 
        type=str, 
        default="1")
    
    parser.add_argument(
        "--seed",
        type=int, 
        default=42)

    parser.add_argument(
        "-a",
        "--trainable_audio_layers", 
        type=int,
        help="Number of audio encoder layers to train (from the top)")
    
    parser.add_argument(
        "-m",
        "--trainable_midi_layers", 
        type=int,
        help="Number of midi encoder layers to train (from the top)")
    
    parser.add_argument(
        "-t",
        "--trainable_text_layers", 
        type=int,
        help="Number of text encoder layers to train (from the top)")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    params = parse_args()

    base_conf = load_conf(params.config_path)
    if params.pretrained_experiment_id is not None:
        trained_config_path = os.path.join(
            get_root_dir(), "save", "experiments", params.pretrained_experiment_id, "config.yaml"
        )
        trained_config = load_conf(trained_config_path)
    else:
        raise ValueError("Pretrained experiment id is required")
    

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

    if trained_config.model_config.model_name not in model_config_map:
        raise ValueError(f"{trained_config.model_config.model_name} model is not supported")
    model_conf_path = os.path.join(base_conf.env.base_dir, model_config_map[trained_config.model_config.model_name])
    
    config = merge_conf(params.config_path, dataset_conf_path, model_conf_path)
    
    if params.pretrained_experiment_id is not None:
        config.finetuning.pretrained_experiment_id = params.pretrained_experiment_id
    if params.trainable_audio_layers is not None:
        config.finetuning.trainable_audio_layers = params.trainable_audio_layers
    if params.trainable_midi_layers is not None:
        config.finetuning.trainable_midi_layers = params.trainable_midi_layers
    if params.trainable_text_layers is not None:
        config.finetuning.trainable_text_layers = params.trainable_text_layers

    logger = Logger(config)
    set_environment(params.seed, params.device_num)

    finetuner = PianoBindFinetuner(config, logger)
    print("# of trainable parameters:", finetuner.count_parameters())

    finetuner.train()
