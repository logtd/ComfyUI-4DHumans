from ..utils.process_humans import get_human


class SelectHumanNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "images": ("IMAGE",),
            "smpl": ("SMPL", ),
            "idx": ("INT", {"default": 0, "min": 0, "max": 999, "step": 1}),
        }}
    RETURN_TYPES = ("SMPL", "IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("smpl", "images", "masks", "semantic_maps")
    FUNCTION = "process"

    CATEGORY = "4dhumans"

    def process(self, images, smpl, idx):
        results = get_human(images, smpl, idx)

        return (results['smpl'], results['images'], results['masks'], results['semantic_maps'])
