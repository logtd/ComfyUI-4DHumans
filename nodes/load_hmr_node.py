import os

import torch

from folder_paths import models_dir

from ..utils.process_humans import load_hmr_model


HMR_PATH = os.path.join(models_dir, 'hmr')
os.makedirs(HMR_PATH, exist_ok=True)

SMPL_PATH = os.path.join(models_dir, 'smpl')
os.makedirs(SMPL_PATH, exist_ok=True)


class LoadHMRNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "checkpoint": (os.listdir(HMR_PATH), )}}

    RETURN_TYPES = ("HMR",)
    FUNCTION = "load"

    CATEGORY = "4dhumans"

    def load(self, checkpoint):
        checkpoint_path = os.path.join(HMR_PATH, checkpoint)
        model_info = load_hmr_model(checkpoint_path, SMPL_PATH)
        return (model_info,)