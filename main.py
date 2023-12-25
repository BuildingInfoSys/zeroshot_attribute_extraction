import argparse
import clip
import glob
import os
import torch
from models.clip_predict import text_prompt_by_task, predict

def get_valid_images(paths):
    return [p for p in paths if p.split('.') in ["jpg", "png", "jpeg"]]


def main(args):
    #TODO: write body of some functions
    if(args.task == "segmentation"):
        #model = build_sam()
        model = None
    elif(args.task in ["nFloors", "roofType", "yearBuilt"]):
        
        #model = load_clip()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.to(torch.float32)
        if(not os.path.exists(args.data_dir)):
            raise Exception(f'directory not found for {args.data_dir}')
        img_paths = glob.glob(os.path.join(args.data_dir, "*"))
        img_paths = get_valid_images(img_paths)

        TEXT_PROMPT, PROMPT_TEMPLATE, NUM_CLASSES = text_prompt_by_task(args.task)
        text_input = torch.cat([clip.tokenize(PROMPT_TEMPLATE.format(c)) for c in TEXT_PROMPT]).to(device)
        prediction_df = predict(model, text_input, img_paths, preprocess, device, agg = "max", num_classes = NUM_CLASSES)

    else:
        raise Exception("task mode not supported, available tasks: segmentation, nFloors, roofType, yearBuilt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot repository for building attribute extraction")
    parser.add_argument("--data_dir", default="/nfs/turbo/coe-stellayu/brianwang/testData/nFloors/merged_data/Ann Arbor, MI",
        help="path to image folder",type=str)
    #parser.add_argument('--csv_label_path')
    #parser.add_argument('--model', default = "brails")
    parser.add_argument("--task")
    parser.add_argument("--visualize_results", default = False, help = "visualize best and worse cases?")
    parser.add_argument("--output_path", default = None, help = "Optional output path when writing result to a new csv file")
    args = parser.parse_args()
    main(args)