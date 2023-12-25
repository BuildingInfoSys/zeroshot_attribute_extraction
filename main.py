import argparse
import clip
import glob
import os
import torch
from clip_predict import text_prompt_by_task, predict, pred_idx_to_labels
from grounded_sam_predict import build_seg_models, predict_seg

def get_valid_images(paths):
    return [p for p in paths if p.split('.')[-1] in ["jpg", "png", "jpeg"]]

def process_img_paths(data_path):
    if(data_path[:-3] not in ["jpg", "png", "jpeg"]):
        img_paths = glob.glob(os.path.join(data_path, "*")) #data folder is providee
    else:
        img_paths = [data_path] #single image(path to target img is provided)
    img_paths = get_valid_images(img_paths)
    return img_paths


def main(args):
    if(not os.path.exists(args.data_dir)):
        raise Exception(f'data directory not found for {args.data_dir}')
    if(not os.path.exists(args.output_path)):
        os.makedirs(os.path.exists(args.output_path))
    
    img_paths = process_img_paths(args.data_dir)
    if(args.task == "segmentation"):
        grounding_dino_model, sam_predictor = build_seg_models()
        predict_seg(img_paths[:10], args.output_path, grounding_dino_model, sam_predictor)

    elif(args.task in ["nFloors", "roofType", "yearBuilt"]):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.to(torch.float32)
        #import ipdb;ipdb.set_trace()

        TEXT_PROMPT, PROMPT_TEMPLATE, GT_LABELS = text_prompt_by_task(args.task)
        text_input = torch.cat([clip.tokenize(PROMPT_TEMPLATE.format(c)) for c in TEXT_PROMPT]).to(device)
        prediction_df = predict(model, text_input, img_paths, preprocess, device, agg = "max", num_classes = len(GT_LABELS))
        prediction_df = pred_idx_to_labels(gt_labels = ['gable', 'hip', 'flat'], prediction_df = prediction_df)
        prediction_df.to_csv(os.path.join(args.output_path, f'{args.task}_pred.csv'), index=False)
    else:
        raise Exception("task mode not supported, available tasks: segmentation, nFloors, roofType, yearBuilt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot repository for building attribute extraction")
    parser.add_argument("--data_dir", default="/nfs/turbo/coe-stellayu/brianwang/testData/nFloors/merged_data/Houston, TX",
        help="path to image folder",type=str)
    #parser.add_argument('--csv_label_path')
    #parser.add_argument('--model', default = "brails")
    parser.add_argument("--task")
    parser.add_argument("--visualize_results", default = False, help = "visualize best and worse cases?")
    parser.add_argument("--output_path", default = None, help = "Optional output path when writing result to a new csv file")
    args = parser.parse_args()
    main(args)