# Dataset Cartography

## SQUAD Support

Original paper for Dataset Cartography [Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics](https://aclanthology.org/2020.emnlp-main.746) at EMNLP 2020.  

Original repo: [Dataset Cartography](https://github.com/allenai/cartography).

This expansion to the Dataset Cartography framework adds support for computing confidence and
variability on the SQUAD dataset.

### Requirements

This project is based on Hugging Face Transformers.

Install requirements:
```shell
pip install -r requirements.txt
```

### Training a model

A trainer is provided to train a model in SQUAD and produce Dataset dynamics data and
and a json version of training data that transforms ids to integers, for better
torch tensor compatibility.  

To train a model, use:
```shell
python -m cartography.classification.run_squad \
    --do_train \
    --output_dir ./trained_model
```

By default, `google/electra-small-discriminator` model is trained on this task.  

To train another model: use:
```shell
python -m cartography.classification.run_squad \
    --do_train \
    --model $(MODEL)
    --output_dir ./trained_model
```

Other arguments are available. Check Hugging Face for more information and tutorials.  


### Plotting Data Maps

This example plots data maps on the SQUAD. The `trained_model` directory should contain
a directory named `training_dymanics`, which contains logits and gold label data for each
training example, for each epoch. This information is used to calculate confidence and variability 
data.

```shell
python -m cartography.selection.train_dy_filtering \
    --plot \
    --task_name SQUAD \
    --model_dir ./trained_model \
    --model model-name \
    --plots_dir ./plots
```

### Filtering Training Data

To filter hard examples, use:
```shell
python -m cartography.selection.train_dy_filtering \
    --filter \
    --task_name SQUAD \
    --model_dir ./trained_model \
    --metric confidence \
    --data_dir .trained_model/glue_data \
    --filtering_output_dir ./filtered_train_data
```

To filter ambiguous examples, use:
```shell
python -m cartography.selection.train_dy_filtering \
    --filter \
    --task_name SQUAD \
    --model_dir ./trained_model \
    --metric variability \
    --data_dir .trained_model/glue_data \
    --filtering_output_dir ./filtered_train_data
```

