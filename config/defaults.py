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
_C.TASK.custom_class_prompts = {
    1: ['one story house','bungalow','flat house'], #one-story prompts
    2: ['two story house','two-story duplex','raised ranch'], #two-story prompts
    3: ['three story house','three story house','three-decker']
}
_C.TASK.custom_seg_prompts = ['window', 'door', "roof", "fence", "facade"]
_C.TASK.custom_prompt_temp = None

#DEFAULT PROMPTS (that aligns with paper experiments)
_C.TASK.prompt_template_by_task = {
    "nFloors":" a photo of a {}",
    "roofType": "{} roof shape",
    "yearBuilt": "built in {}"
}

_C.TASK.default_text_prompts_by_task = {
    "nFloors":
    {
        1: ['one story house','bungalow','flat house'], #one-story prompts
        2: ['two story house','two-story duplex','raised ranch'], #two-story prompts
        3: ['three story house','three story house','three-decker']
    },
    "roofType":{
        'gable': ['gable'], 
        'hip': ['hip'],
        'flat': ['flat']
    },
    "yearBuilt":{
        'Pre-1970': 'Pre-1970',
        '1970-1979': '1970-1979',
        '1980-1989': '1980-1989',
        '1990-1999': '1990-1999',
        '2000-2009': '2000-2009',
        'Post-2010': 'Post-2010'
    }
}


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