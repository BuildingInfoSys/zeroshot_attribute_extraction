# Zero-shot Prediction For Building Attribute Extraction
The emergence of large-scale foundational models(e.g. ChatGPT), which are trained on billion of real-world data, are equipped with sufficient understanding of our world. In this repository, we demonstrate such effectivenes by applying AI foundational models onto structural and civil engineering applications. Our WACV paper, Zero-shot Building Attribute Extraction from Large-Scale Vision and Language Models (to be attached), presents the feasibility of this workflow on the task of building attribute extraction. We evaluate the effectiveness on classification tasks (number of floors, roofType, year built) and segmentation tasks(segmenting roof, window, door, facade). Without any additional training on target domains, these models can already achieve competitive performances across multiple tasks, which demonstrates the robustness and applicability of these models. To the best of our knowledge, we are the first to introduce these large-scale foundational models in structural engineering domains. By organizing and abstracting the technical details into convenient usage, we hope this demo could inspire more structural engineering researchers to unleash the power of these robust models and apply to their domains.

## Approach
### Classification
To perform classification task, we utilize [CLIP](https://github.com/openai/CLIP), an open-source, multi-modal model that has learned the relationship between language semantics and visual context. Given an image to be queried and task-related vocabularies, the model is able to output the vocabulary that most closely aligns with the visual context. (needs rephrasing, I am not sure whether I should make it more concise for WACV researchers, or make it understandable for SimCenter researchers)




## Getting Started
TODO: (1) install conda, python, (2) conda commands to build environment
QUESTION: need to provide BRAIL as baseline model?(python needs to be <3.10)
In this section, we will setup the required packages to run our experimented models - CLIP and SAM. Please setup conda (as instructed [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)). After installing conda to provide virtual environment, below are the commands to setup all required python packages: 
```
conda create --y --name zeroshot_extraction python=3.8.0
conda activate zeroshot_extraction
conda install --y -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

Note that python version has to be >=3.7.1 to successfully compile.


## Usage
### Conceptual Background
TODO: (1) Introduce Concept of Text prompt so user can customize their own prompts based on domain, (2) Some brief background of CLIP & SAM


#### Arguments
- data_dir: path to target image(s) (see Evaluation section for details)
- task: task to be evaluated. Current supported modes are segmentation, nFloors, roofType, and yearBuilt. 
    - Task = "segmentation" segments out roof, windows, doors, and facade.
    - Task = "nFloors" predicts the number of stories of a given house. The default text prompts are limited to predicting one-story, two-story, and three-story houses
    - Task = "roofType" predicts the roof type of a given house. The default text prompts are limited to predicting flat/gable/hip roof shape
    - Task = "yearBuit" predicts the year the given house is constructed. The default text prompts predicts "Pre-1970','1970-1979','1980-1989','1990-1999','2000-2009','Post-2010"
-  

### Evaluation
**Evaluate on single image**: set data_dir argument to be the absolute path to the target image. 

**Evaluate on multiple image**: Place all images in a directory. Then, set data_dir argument to be the absolute path to that target directory(containing all images). 



## Examples
TODO: (1) For each task, show some successful examples and sample code

