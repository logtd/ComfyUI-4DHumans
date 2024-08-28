from folder_paths import models_dir
import comfy.model_management

from ..utils.process_humans import process_humans


class ProcessHumansNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "images": ("IMAGE",),
            "hmr": ("HMR", ),
            "detectron": ("DETECTRON", ),
        }}
    RETURN_TYPES = ("SMPL", "IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("smpl", "images", "masks", "semantic_maps")
    FUNCTION = "process"

    CATEGORY = "4dhumans"

    def process(self, images, hmr, detectron):
        intermediate_device = comfy.model_management.intermediate_device()
        torch_device = comfy.model_management.get_torch_device()

        results = process_humans(images, hmr, detectron, figure_scale=None, device=torch_device)

        return (results['smpl'], results['images'], results['masks'], results['semantic_maps'])
