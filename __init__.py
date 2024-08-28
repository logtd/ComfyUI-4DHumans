import os

REPO_PATH = os.path.dirname(os.path.abspath(__file__))

from folder_paths import models_dir

SMPL_PATH = os.path.join(models_dir, 'smpl')
os.makedirs(SMPL_PATH, exist_ok=True)


from .nodes.process_humans_node import ProcessHumansNode
from .nodes.load_detectron_node import LoadDetectronNode
from .nodes.load_hmr_node import LoadHMRNode
# from .nodes.select_human_node import SelectHumanNode


NODE_CLASS_MAPPINGS = {
    'ProcessHumans': ProcessHumansNode,
    'LoadDetectron': LoadDetectronNode,
    'LoadHMR': LoadHMRNode,
    # 'SelectHuman': SelectHumanNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'ProcessHumans': 'Process 4D Humans',
    'LoadDetectron': 'Load Detectron Model',
    'LoadHMR': 'Load HMR Model',
    # 'SelectHuman': 'Select 4D Human'
}
