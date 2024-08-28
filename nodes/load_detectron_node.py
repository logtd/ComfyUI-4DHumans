import os

from folder_paths import models_dir

from ..utils.process_humans import load_detectron



DETECTRON_PATH = os.path.join(models_dir, 'detectron')
os.makedirs(DETECTRON_PATH, exist_ok=True)



class LoadDetectronNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "checkpoint": (os.listdir(DETECTRON_PATH), )}}

    RETURN_TYPES = ("DETECTRON",)
    FUNCTION = "load"

    CATEGORY = "4dhumans"

    def load(self, checkpoint):
        checkpoint_path = os.path.join(DETECTRON_PATH, checkpoint)
        model = load_detectron(checkpoint_path)
        return (model,)