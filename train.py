import os
import glob
import json
import random

from GPT import *
from utils.data import *
from utils.utils import *

from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from loop_extraction.src.utils.utils import dataset_split
from loop_extraction.src.utils.utils import folder_to_file
from loop_extraction.src.utils.bpe_encode import MusicTokenizer


#### load config
with open("./config/config.json", "r") as f:
    config = json.load(f)

#### set seed
seed_everything(config["random_seed"])
torch.set_float32_matmul_precision("medium")

#### initialize model with GPU
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

#### load tokenizer
bpe_path = "./loop_extraction/tokenizer/tokenizer_meta.json"
tokenizer = MusicTokenizer(bpe_path)

pad_idx = tokenizer.encode(["<pad>"])[0]
vocab_size = tokenizer.bpe_vocab.get_vocab_size() + 1

#### load datasets
folder_path = "./data/"
datasets = [
    "lmd_full_loop_" + str(config["max_length"]),
    "meta_midi_loop_" + str(config["max_length"]),
]

folder_list = []
for dataset in datasets:
    folder_list += glob.glob(os.path.join(folder_path, dataset, "*"))

random.shuffle(folder_list)

#### split song into train, val, test
train_folder, val_folder, _ = dataset_split(folder_list, train_ratio=0.98, val_ratio=0.01)

#### get file_path of each dataset
train_files = folder_to_file(train_folder)
val_files = folder_to_file(val_folder)

# print(f"train_files : {len(train_files)}, val_files : {len(val_files)}, test_files : {len(test_files)}")

#### load dataloader
module = DataModule(
    [train_files, val_files, [""]],
    tokenizer,
    batch_size=config["batch_size"],
    num_workers=config["num_workers"],
    random_drop=config["random_drop"],
)

train_module, val_module, test_module = module.return_dataloader()

#### callback functions
keys = ["version", "random_drop", "max_length"]
model_name = [key + "_" + str(value) for key, value in config.items() if key in keys]
model_name = "-".join(param for param in model_name)
model_name = model_name + "-{epoch}-{val_loss:.4f}"

lr_monitor = LearningRateMonitor(logging_interval="step")
checkpoint = ModelCheckpoint(
    filename=model_name,
    dirpath="./model/",
    monitor="val_loss",
    mode="min",
)

#### load model
model = GPT(
    vocab_size,
    pad_idx,
    dim_model=config["dim_model"],
    num_layers=config["num_layers"],
    num_heads=config["num_heads"],
    multiplier=config["multiplier"],
    lr=config["lr"],
    dropout=config["dropout"],
    weight_decay=config["weight_decay"],
    max_length=config["max_length"],
    warm_up=config["warm_up"],
).to(device)

total_params = sum(p.numel() for p in model.parameters())
# print(f"The number of parameters : {sizeof_number(total_params)}")

#### trainer
trainer = Trainer(
    num_nodes=1,
    accelerator="gpu",
    precision="16-mixed",
    devices=config["gpus"],
    max_epochs=config["epochs"],
    callbacks=[lr_monitor, checkpoint],
    strategy=DDPStrategy(find_unused_parameters=True),
)

trainer.fit(model, train_module, val_module)
