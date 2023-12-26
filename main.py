import argparse
import os
import shutil
from config import cfg
import torch

import clip
from clip_predict import predict, pred_idx_to_labels
from grounded_sam_predict import build_seg_models, predict_seg
from utils import get_valid_images, process_img_paths, text_prompt_by_task, retrieve_custom_prompts

def main(cfg):
    if(not os.path.exists(cfg.DIR.data_dir)):
        raise Exception(f'data directory not found for {cfg.DIR.data_dir}')
    
    img_paths = process_img_paths(cfg.DIR.data_dir)
    if(cfg.TASK.name == "segmentation"):
        grounding_dino_model, sam_predictor = build_seg_models()
        if(cfg.TASK.custom_seg_prompts!=None):
            predict_seg(img_paths, cfg.DIR.output_dir, grounding_dino_model, sam_predictor, CLASSES = cfg.TASK.custom_seg_prompts)
        else:
            predict_seg(img_paths, cfg.DIR.output_dir, grounding_dino_model, sam_predictor)

    elif(cfg.TASK.name in ["nFloors", "roofType", "yearBuilt"]):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.to(torch.float32)
        #import ipdb;ipdb.set_trace()
        if(cfg.INFER.enable_custom_prompts):
            TEXT_PROMPT, PROMPT_TEMPLATE, GT_LABELS = retrieve_custom_prompts(cfg)
        else:
            TEXT_PROMPT, PROMPT_TEMPLATE, GT_LABELS = text_prompt_by_task(cfg.TASK.name)
        text_input = torch.cat([clip.tokenize(PROMPT_TEMPLATE.format(c)) for c in TEXT_PROMPT]).to(device)
        prediction_df = predict(model, text_input, img_paths, preprocess, device, agg = "max", num_classes = len(GT_LABELS))
        prediction_df = pred_idx_to_labels(gt_labels = GT_LABELS, prediction_df = prediction_df)
        prediction_df.to_csv(os.path.join(cfg.DIR.output_dir, f'{cfg.TASK.name}_pred.csv'), index=False)
    else:
        raise Exception("task mode not supported, available tasks: segmentation, nFloors, roofType, yearBuilt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot repository for building attribute extraction")
    parser.add_argument(
        "--cfg",
        default="config/config.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    if(not os.path.exists(cfg.DIR.output_dir)):
        os.makedirs(os.path.exists(cfg.DIR.output_dir))
    with open(os.path.join(cfg.DIR.output_dir, 'config.yaml'), 'w') as f:
        f.write("{}".format(cfg))
    if(cfg.INFER.enable_custom_prompts):
        prompt_fname = cfg.TASK.custom_class_prompts_path.split("/")[-1]
        shutil.copyfile(cfg.TASK.custom_class_prompts_path, os.path.join(cfg.DIR.output_dir, prompt_fname))
    

    main(cfg)