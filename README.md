# ComfyUI-4DHumans
ComfyUI nodes to use [4D-Humans](https://github.com/shubham-goel/4D-Humans)

## Installation

Install all pip packages via `python -m pip install -r requirements.txt`.

This has only been tested on Ubuntu and may require some additional packages/libraries to be installed for Windows/Mac.

Ubuntu libraries
```
apt-get install freeglut3-dev
apt-get install libosmesa6-dev
```

### Models

You need 3 new model directories: `smpl`, `hmr`, and `detectron`.

The directories should look like:
```
smpl
- basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
- SMPL_to_J19.pkl
- smpl_mean_params.npz

hmr
- hmr35.ckpt (this was renamed from the 35000 checkpoint from HMR)

detectron
- model_final_f05665.pkl
```

| Model | Link |
|------------|-----------|
| Detectron | [link](https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl) |
| SMPL (all files in smpl dir) | [link](https://smplify.is.tue.mpg.de/) |
| HMR | [link](https://people.eecs.berkeley.edu/~jathushan/projects/4dhumans/hmr2a_model.tar.gz) |

#### Note:
The SMPL model files are behind a registration wall. Once you download the tar.gz unzip the file and find the 3 files. Then place them in `ComfyUI/models/smpl`.

## Examples
You can find an example workflow in the `example_workflows` directory.

The node outputs the smpl model, masks, semantic maps, and preview images.

https://github.com/user-attachments/assets/c505ac94-7880-4497-867c-2e66389f9f48

