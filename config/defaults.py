from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# TASK-RELATED SETTINGS
# -----------------------------------------------------------------------------
_C.TASK = CN()
_C.TASK.name = None
#specify prompts for each class as dictionary pairs of (key = class, value = class_prompts)
_C.TASK.custom_class_prompts_path = None
_C.TASK.custom_seg_prompts = ['window', 'door', "roof", "fence", "facade"]
_C.TASK.custom_prompt_temp = None

# -----------------------------------------------------------------------------
# DIRECTORY-RELATED SETTINGS
# -----------------------------------------------------------------------------
_C.DIR = CN()
_C.DIR.data_dir = None
_C.DIR.output_dir = None

# -----------------------------------------------------------------------------
# INFERENCE-RELATED SETTINGS
# -----------------------------------------------------------------------------
_C.INFER = CN()
_C.INFER.enable_custom_prompts = False