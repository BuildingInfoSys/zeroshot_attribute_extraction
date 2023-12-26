import glob
import os
import json

def get_valid_images(paths):
    return [p for p in paths if p.split('.')[-1] in ["jpg", "png", "jpeg"]]

def process_img_paths(data_path):
    if(data_path[:-3] not in ["jpg", "png", "jpeg"]):
        img_paths = glob.glob(os.path.join(data_path, "*")) #data folder is providee
    else:
        img_paths = [data_path] #single image(path to target img is provided)
    img_paths = get_valid_images(img_paths)
    return img_paths



PROMPT_TEMPLATE_BY_TASK = {
        "nFloors":" a photo of a {}",
        "roofType": "{} roof shape",
        "yearBuilt": "built in {}"
}
DEFAULT_TEXT_PROMPTS_BY_TASK = {
    "nFloors":[
        'one story house','bungalow','flat house', #one-story prompts
        'two story house','two-story duplex','raised ranch', #two-story prompts
        'three story house','three story house','three-decker'
        #'one story house','one story house', 'one story house', 'two story house','two story house', 'two story house', 'three story house','three story house','three story house'
    ],
    "roofType":['gable', 'hip', 'flat'],
    "yearBuilt":['Pre-1970','1970-1979','1980-1989','1990-1999','2000-2009','Post-2010']
}
#required when providing multi-prompts per class 
GT_LABELS_BY_TASK = {
    "nFloors":[1,2,3], 
    "roofType":['gable', 'hip', 'flat'],
    "yearBuilt":['Pre-1970','1970-1979','1980-1989','1990-1999','2000-2009','Post-2010']
}

'''
Helper function that returns default text prompts and prefix/suffix for target task
'''
def text_prompt_by_task(task):
    return DEFAULT_TEXT_PROMPTS_BY_TASK[task], PROMPT_TEMPLATE_BY_TASK[task], GT_LABELS_BY_TASK[task]

def retrieve_custom_prompts(cfg):
    with open(cfg.TASK.custom_class_prompts_path, 'r') as fp:
        prompt_dict = json.load(fp)
    
    gt_labels = prompt_dict.keys()
    text_prompts = prompt_dict.values()

    if(cfg.TASK.custom_prompt_temp == None):
        prompt_template = PROMPT_TEMPLATE_BY_TASK[cfg.TASK.name]
    else:
        prompt_template = cfg.TASK.custom_prompt_temp
    return text_prompts, prompt_template, gt_labels
    
