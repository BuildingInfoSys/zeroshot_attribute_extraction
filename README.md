# Zero-shot Prediction For Building Attribute Extraction
The emergence of large-scale foundational models(e.g. ChatGPT), which are trained on billion of real-world data, are equipped with sufficient understanding of our world. In this repository, we demonstrate such effectivenes by applying AI foundational models onto structural and civil engineering applications. Our WACV paper, Zero-shot Building Attribute Extraction from Large-Scale Vision and Language Models (to be attached), presents the feasibility of this workflow on the task of building attribute extraction. We evaluate the effectiveness on classification tasks (number of floors, roofType, year built) and segmentation tasks(segmenting roof, window, door, facade). Without any additional training on target domains, these models can already achieve competitive performances across multiple tasks, which demonstrates the robustness and applicability of these models. To the best of our knowledge, we are the first to introduce these large-scale foundational models in structural engineering domains. By organizing and abstracting the technical details into convenient usage, we hope this demo could inspire more structural engineering researchers to unleash the power of these robust models and apply to their domains.

## Approach
### Classification
To perform classification task, we utilize [CLIP](https://github.com/openai/CLIP), an open-source, multi-modal model that has learned the relationship between language semantics and visual context. Given an image to be queried and task-related vocabularies, the model is able to output the vocabulary that most closely aligns with the visual context. 




## Getting Started
In this section, we will setup the required packages to run our experimented models - CLIP and SAM. On a GPU-equipped device, please setup conda (as instructed [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)). After installing conda to provide virtual environment, below are the steps to setup all required python packages: 
**Step1; Setting Up Local Variables** 
```
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True
export CUDA_HOME=/path/to/cuda_device/
```
Note that this step is crucial to successfully setup model-dependent packages without error. 


**Step2: Installing actual packages**
```
conda create --y --name zeroshot_extraction python=3.8.0
conda activate zeroshot_extraction
conda install --y -c pytorch pytorch=1.10.0 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm pandas matplotlib==3.7.3
pip install git+https://github.com/openai/CLIP.git
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
```
Note that python version has to be >=3.8.0, and pytorch(>=1.10.0) has to be compatible with CUDA device to successfully compile.

**Step3: Download Model's Pretrained Weights**
```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```


## Usage
### Conceptual Background
TODO: (1) Introduce Concept of Text prompt so user can customize their own prompts based on domain, (2) Some brief background of CLIP & SAM

## Command
Below is the general command to run this repo:
```
python3 main.py --task TARGET_TASK --output_path YOUR_OUTPUT_PATH --data_dir DATA_PATH_TO_EVALUATED_IMAGES
```

#### Description of Arguments
- data_dir: path to target image(s) (see Evaluation section for details)
- task: task to be evaluated. Current supported modes are segmentation, nFloors, roofType, and yearBuilt. 
    - Task = "segmentation" segments out roof, windows, doors, and facade.
    - Task = "nFloors" predicts the number of stories of a given house. The default text prompts are limited to predicting one-story, two-story, and three-story houses
    - Task = "roofType" predicts the roof type of a given house. The default text prompts are limited to predicting flat/gable/hip roof shape
    - Task = "yearBuit" predicts the year the given house is constructed. The default text prompts predicts "Pre-1970','1970-1979','1980-1989','1990-1999','2000-2009','Post-2010"
-  output_path: the output path that stores the prediction(in csv file) of given images from your data directory.

### Evaluation
**Evaluate on single image**: set data_dir argument to be the absolute path to the target image. 
**Evaluate on multiple image**: Place all images in a directory. Then, set data_dir argument to be the absolute path to that target directory(containing all images). 



## Examples
TODO: (1) For each task, show some successful examples and sample code

