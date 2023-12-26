TASK: 
    name: "nFloors"
    custom_class_prompts: {
        1: ['one story house','bungalow','flat house'], #one-story prompts
        2: ['two story house','two-story duplex','raised ranch'], #two-story prompts
        3: ['three story house','three story house','three-decker']
    }
   custom_prompt_temp: None


DIR:
    data_dir: "/nfs/turbo/coe-stellayu/brianwang/testData/nFloors/merged_data/Houston, TX"
    output_dir: "/nfs/turbo/coe-stellayu/brianwang/results/wacv_test"

INFER:
    enable_custom_prompts: False #enable to true if you want to use your own prompts(update prompts in TASK as well)