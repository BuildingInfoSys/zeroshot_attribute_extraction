import cv2
import numpy as np
import pandas as pd
import supervision as sv
import os
import glob
import pickle

import torch
import torchvision
from tqdm import tqdm

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

'''

Set Up segmentation models: GroundedSAM (GroundingDINO and SAM)
'''
def build_seg_models():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'DEVICE FOUND: {DEVICE}')
    # GroundingDINO config and checkpoint
    GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"
    # Segment-Anything checkpoint
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"

    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
    sam.to(device=DEVICE)
    sam_predictor = SamPredictor(sam)
    return grounding_dino_model, sam_predictor

'''
Prompting SAM with detected boxes (provided from GroundingDINO)
Input: 
- image: raw image
- xyxy: bounding box coordinates of detected objects (from GroundingDINO)
'''
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

'''
Performs segmentation on an image
Given that SAM only allows point or box annotations as prompts(text prompts was not enabled), 
GroundedSAM incorporates GroundingDINO with SAM -> (1) GroundingDINO accepts an image and text prompt, detected targets and outputs bounding boxes
(2) GroundedSAM accepts those bounding boxes as input and returns the segmentation maps
Function Result: saves segmented image, stores related data on segmented/detected results(detected boxes, class labels, segmented mask)
'''
def run_on_one_image(img_path, output_dir, grounding_dino_model, sam_predictor, CLASSES, BOX_THRESHOLD, TEXT_THRESHOLD, NMS_THRESHOLD):
    SOURCE_IMAGE_PATH = img_path
    img_name = SOURCE_IMAGE_PATH.split("/")[-1][:-4]

    # load image
    image = cv2.imread(SOURCE_IMAGE_PATH)

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=BOX_THRESHOLD
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator(text_scale = 0.4, text_padding = 5)
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _ 
        in detections]
    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)

    # save the annotated grounding dino image
    #cv2.imwrite(os.path.join(output_dir, f"{img_name}_dino.png"), annotated_frame)

    # NMS post process
    output_str = f"{img_name} NMS: Before = {len(detections.xyxy)} boxes, "
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    output_str += f"After = {len(detections.xyxy)} boxes"

    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator(text_scale = 0.4, text_padding = 4)
    mask_annotator = sv.MaskAnnotator()
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _ 
        in detections]
    
    # save the annotated grounded-sam image
    mask = mask_annotator.annotate(scene=np.zeros(image.shape).astype(image.dtype).astype(np.uint8), detections=detections)
    cv2.imwrite(os.path.join(output_dir, f"{img_name}_mask.png"), mask)
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    cv2.imwrite(os.path.join(output_dir, f"{img_name}_mask_overlap.png"), annotated_image)
    output_dict = {"confidence":detections.confidence[nms_idx], "class":detections.class_id[nms_idx], "mask":detections.mask}
    return output_dict

'''
Main function of processing given images to be segmented
'''
def predict_seg(
    img_sources, output_dir, grounding_dino_model, sam_predictor, CLASSES=["door", "window", "roof", "facade"], 
    BOX_THRESHOLD=0.25, TEXT_THRESHOLD=0.25, NMS_THRESHOLD=0.8):
    sam_results = {}
    for img_path in tqdm(img_sources): 
        img_name = img_path.split("/")[-1]
        sam_labels = run_on_one_image(img_path, output_dir, grounding_dino_model, sam_predictor, CLASSES, BOX_THRESHOLD, TEXT_THRESHOLD, NMS_THRESHOLD)
        sam_results[img_name]=sam_labels
    with open(os.path.join(output_dir, "label_dict"), "wb") as fp:   #Pickling
        pickle.dump(sam_results,fp) 
    

    