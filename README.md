# Loop Extraction and Generation Offical

This repository is the implementation of "Music Loop Extraction and Generation for Multi-track Symbolic Music".


## Getting Started

### Environments

* Python 3.10.12
* Ubuntu 20.04.5 LTS
* Read [requirements.txt](/requirements.txt) for other Python libraries

### Data Download

* [Lakh MIDI Dataset (LMD-full)](https://colinraffel.com/projects/lmd/)
* [MetaMIDI Dataset (MMD)](https://zenodo.org/record/5142664#.YQN3c5NKgWo)

### Data Preprocess

You can refer to [BERT-stranger](https://github.com/sjhan91/BERT-stranger) Github page.

### Model Training
You should modify [config.json](/config/config.json) for model configuration change ("**GPT-3 Small**", "**GPT-3 Medium**"). By setting "strategy" (ex. ddp) in [train.py](/train.py) and "gpus" in [config.json](/config/config.json) (ex. [0, 1, 2]), you can train the models with distributed GPU settings of pytorch-lightining. Here is an example of **GPT-Medium** configurations.

```json
{
    "version": "GPT-medium",
    "random_seed": 0,
    "batch_size": 16,
    "num_workers": 16,
    "dim_model": 1024,
    "num_layers": 24,
    "num_heads": 16,
    "multiplier": 2.667,
    "lr": 8e-4,
    "dropout": 0.1,
    "random_drop": 0,
    "weight_decay": 0.1,
    "max_length": 1024,
    "warm_up": 2000,
    "epochs": 5,
    "gpus": [4, 5, 6, 7]
}
```


For training the loop generation model, the command is as below;
```
python train.py
```

### Model Inference
You can generate loop samples through [generate.ipynb](/generate.ipynb). Also, you can obtain the metrics of controllability and density & coverage from [evaluate.ipynb](/evaluate.ipynb). The pre-trained models will be released later.

## Samples
You can listen our generated samples on [Google Drive](https://drive.google.com/drive/folders/1cPHXERTKxlUSEFsziMGbnF0Y-3uaYuwn?usp=sharing). It consists of sets of (true_loop_\*.mid, gen_cond_full_\*.mid [corresponding to the full conditions], gen_cond_inst_\*.mid [corresponding to the partial conditions]).


## Appreciation
I have learned a lot and reused available codes from [dvruette FIGARO](https://github.com/dvruette/figaro) for the data preprocess, [lucidrains vit-pytorch](https://github.com/lucidrains/vit-pytorch) for the basic structure of Transformer, and [xformers](https://github.com/facebookresearch/xformers) for advanced techniques.


## References
Sangjun Han, Hyeongrae Ihm, Woohyung Lim (LG AI Research), "Music Loop Extraction and Generation for Multi-track Symbolic Music"