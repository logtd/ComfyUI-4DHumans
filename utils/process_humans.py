import os
import torch
import numpy as np
import platform
import pyrender
# from pathlib import Path
from tqdm import tqdm


if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"

import comfy.utils

from ..humans4d.hmr2.models import load_hmr2
from ..humans4d.hmr2.utils import recursive_to
from ..humans4d.hmr2.datasets.vitdet_dataset import ViTDetDataset
from ..humans4d.hmr2.utils.renderer import cam_crop_to_full
# from ..humans4d.hmr2.configs import get_config
# from ..humans4d.hmr2.models import check_smpl_exists
from ..humans4d.hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig

from .visualize_humans import SemanticRenderer
from .. import REPO_PATH

# For Windows, remove PYOPENGL_PLATFORM to enable default rendering backend
sys_name = platform.system()
if sys_name == "Windows":
    os.environ.pop("PYOPENGL_PLATFORM")


LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def predict_smpl(batch, model, model_cfg, figure_scale=None):
    all_verts = []
    all_cam_t = []
    with torch.no_grad():
        out = model(batch)

    pred_cam = out["pred_cam"]
    pred_smpl_parameter = out["pred_smpl_params"]
    if figure_scale is not None:
        pred_smpl_parameter['betas'][0][1] = float(figure_scale)
    smpl_output = model.smpl(
        **{k: v.float() for k, v in pred_smpl_parameter.items()},
        pose2rot=False,
    )
    pred_vertices = smpl_output.vertices
    out["pred_vertices"] = pred_vertices.reshape(
        batch["img"].shape[0], -1, 3
    )

    box_center = batch["box_center"].float()
    box_size = batch["box_size"].float()
    img_size = batch["img_size"].float()

    scaled_focal_length = (
        model_cfg.EXTRA.FOCAL_LENGTH
        / model_cfg.MODEL.IMAGE_SIZE
        * img_size.max()
    )
    pred_cam_t_full = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            ).detach().cpu().numpy()
    # Render the result
    batch_size = batch["img"].shape[0]
    for n in range(batch_size):
        # Add all verts and cams to list
        verts = out["pred_vertices"][n].detach().cpu().numpy()
        cam_t = pred_cam_t_full[n]
        all_verts.append(verts)
        all_cam_t.append(cam_t)

    misc_args = dict(
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        focal_length=scaled_focal_length,
    )
    
    smpl_outs = {
        k: v.detach().cpu().numpy() for k, v in pred_smpl_parameter.items()
    }

    results_dict_for_rendering = {
        "verts": all_verts,
        "cam_t": all_cam_t,
        "render_res": img_size[n].cpu().numpy(),
        "smpls": smpl_outs,
        "scaled_focal_length": scaled_focal_length.cpu().numpy(),
    }
    return results_dict_for_rendering, misc_args


def load_image(model_cfg, image, detector):
    # Detect humans in image
    # image = image.permute(1,2,0)*255
    det_out = detector(image)
    # image = image.numpy()
    det_instances = det_out["instances"]
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    # Run HMR2.0 on all detected humans
    dataset = ViTDetDataset(model_cfg, image, boxes)
    return torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=False, num_workers=0
    )

def load_hmr_model(hmr2_checkpoint_path, smpl_dir_path):
    model, model_cfg = load_hmr2(hmr2_checkpoint_path,smpl_dir_path)
    return {
        'model': model,
        'model_cfg': model_cfg
    }


def load_detectron(detectron_checkpoint_path):
    cfg_path = os.path.join(REPO_PATH, "configs", "cascade_mask_rcnn_vitdet_h_75ep.py")
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = detectron_checkpoint_path
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    return detector


def process_humans(images, hmr, detector, figure_scale, device):
    images = images.permute(0, 3, 1, 2)
    _, _, height, width = images.shape
    model, model_cfg = hmr['model'], hmr['model_cfg']


    model = model.to(device)
    detector.model.to(device)
    # This PyRender is only used for visualizing, we use Blender after to render different conditions.
    renderer = SemanticRenderer(
        model_cfg,
        faces=model.smpl.faces,
        lbs=model.smpl.lbs_weights,
        viewport_size=(720, 720),
    )
    results = []
    rendered_images = []
    semantic_maps = []
    masks = []
    focal_length = None
    mesh_base_color = None
    scene_bg_color = None
    pbar = comfy.utils.ProgressBar(len(images))
    for image in tqdm(images, desc="Processing Human Images:"):
        image = image.permute(1,2,0)*255
        image = image.numpy()
        renderer.renderer.delete()
        renderer.renderer = pyrender.OffscreenRenderer(
            viewport_width=width,
            viewport_height=height,
            point_size=1.0,
        )
        dataloader = load_image(model_cfg, image, detector)
        

        for batch in dataloader:
            batch = recursive_to(batch, device)
            results_dict_for_rendering, misc_args = predict_smpl(batch, model, model_cfg, figure_scale)
            mesh_base_color = misc_args['mesh_base_color']
            scene_bg_color = misc_args['scene_bg_color']
            focal_length = misc_args['focal_length'].to('cpu')
            
            rendering_results = renderer.render_all_multiple(
                results_dict_for_rendering["verts"], 
                cam_t=results_dict_for_rendering["cam_t"], 
                render_res=results_dict_for_rendering["render_res"], **misc_args
            )
            # Overlay image
            valid_mask = rendering_results["Image"][:, :, -1][:, :, np.newaxis]
            cam_view = (
                valid_mask * rendering_results["Image"][:, :, [2, 1, 0]]
                + (1 - valid_mask) * image.astype(np.float32)[:, :, ::-1] / 255
            )
            rendered_images.append(torch.from_numpy(cam_view[:, :, ::-1].copy()))
            semantic_maps.append(torch.from_numpy(rendering_results.get("SemanticMap")))
            masks.append(torch.from_numpy(rendering_results.get("Mask")[:, :, 0]))
            results.append(results_dict_for_rendering) # still in np
        pbar.update(1)

    smpl = {
        'results': results,
        'width': width,
        'height': height,
        'model_cfg': model_cfg,
        'faces': model.smpl.faces,
        'lbs': model.smpl.lbs_weights,
        'viewport_size': (720, 720),
        'misc_args': { 'mesh_base_color': mesh_base_color, 'scene_bg_color': scene_bg_color, 'focal_length': focal_length }
    }
    
    return {
        'smpl': smpl,
        'images': torch.stack(rendered_images),
        'masks': torch.stack(masks).to(torch.float16),
        'semantic_maps': torch.stack(semantic_maps),
    }


def get_human(images, smpl, idx):
    results = []
    rendered_images = []
    semantic_maps = []
    masks = []

    model_cfg, faces, lbs = smpl['model_cfg'], smpl['faces'], smpl['lbs']
    viewport_size = smpl['viewport_size']
    width, height = smpl['width'], smpl['height']
    misc_args = smpl['misc_args']

    renderer = SemanticRenderer(
        model_cfg,
        faces=faces,
        lbs=lbs,
        viewport_size=viewport_size,
    )

    renderer.renderer = pyrender.OffscreenRenderer(
        viewport_width=width,
        viewport_height=height,
        point_size=1.0,
    )

    human_smpl = []
    keys = ['verts']
    smpl_keys = ['global_orient', 'body_pose', 'betas']
    for results_dict_for_rendering in smpl['results']:
        result = { 
            'scaled_focal_length': results_dict_for_rendering['scaled_focal_length'], 
            'smpls': {},
            'cam_t': results_dict_for_rendering['cam_t'],
            'render_res': results_dict_for_rendering['render_res']
        }
        good = True
        for key in keys:
            if idx < len(results_dict_for_rendering[key]):
                result[key] = [results_dict_for_rendering[key][idx]]
            else:
                good = False
        
        for key in smpl_keys:
            if idx < len(results_dict_for_rendering['smpls'][key]):
                result['smpls'][key] = [results_dict_for_rendering['smpls'][key][idx]]
            else:
                good = False
        
        if good:
            human_smpl.append(result)
        else:
            human_smpl.append(None)

    for image_orig, results_dict_for_rendering in zip(images, human_smpl):
        image = image_orig.permute(1,2,0)*255
        image = image.numpy()
        if results_dict_for_rendering is not None:
            rendering_results = renderer.render_all_multiple(
                results_dict_for_rendering["verts"], 
                cam_t=results_dict_for_rendering["cam_t"], 
                render_res=results_dict_for_rendering["render_res"], **misc_args
            )
            # Overlay image
            valid_mask = rendering_results["Image"][:, :, -1][:, :, np.newaxis]
            cam_view = (
                valid_mask * rendering_results["Image"][:, :, [2, 1, 0]]
                + (1 - valid_mask) * image.astype(np.float32)[:, :, ::-1] / 255
            )
            rendered_images.append(torch.from_numpy(cam_view[:, :, ::-1].copy()))
            semantic_maps.append(torch.from_numpy(rendering_results.get("SemanticMap")))
            masks.append(torch.from_numpy(rendering_results.get("Mask")[:, :, 0]))
        else:
            rendered_images.append(image_orig)
            semantic_maps.append(torch.ones_like(image_orig))
            masks.append(torch.zeros_like(image_orig))
        results.append(results_dict_for_rendering) # still in np

    smpl = {
        'results': results,
        'width': width,
        'height': height,
        'model_cfg': model_cfg,
        'faces': faces,
        'lbs': lbs,
        'viewport_size': (720, 720),
        'misc_args': misc_args
    }
    
    return {
        'smpl': smpl,
        'images': torch.stack(rendered_images),
        'masks': torch.stack(masks).to(torch.float16),
        'semantic_maps': torch.stack(semantic_maps),
    }
