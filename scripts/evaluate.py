import os
import argparse
from omegaconf import OmegaConf
from pianobind.utils.utils import get_root_dir

from pianobind.tasks.retrieval import Retrieval
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a PianoBind model")

    parser.add_argument(
        "--model_id",
        type=str,
        default="at_trimodal",
        help="experiment id under which trained model was saved",
    )

    parser.add_argument(
        "--test_set_size",
        type=int,
        help="size of the random testing set",
        default=199,
    )

    parser.add_argument(
        "--device_num",
        type=str,
        default="1",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    params = parse_args()
    model_id = params.model_id

    muscall_config = OmegaConf.load(
        os.path.join(get_root_dir(), "save/experiments/{}/config.yaml".format(model_id))
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = params.device_num
    print("GPU: ", params.device_num)
    
    evaluation = Retrieval(muscall_config, params.test_set_size)

    evaluation.evaluate()
