from __future__ import print_function, division

from datasets import CUBDataset, FGVCAircraft, FOODDataset
from engine import train
from model import get_model
from utils import plot_results, EarlyStopper

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import ConcatDataset
from torchvision.transforms import v2
import torchvision
from torch.utils.data import default_collate
torchvision.disable_beta_transforms_warning()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

from utils import EarlyStopper

import yaml
import json
import time
import os
import wandb

config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)


run = wandb.init(entity='metalab', project='cv703_assignment1', config=config)


LEARNING_RATE = float(config["LEARNING_RATE"])
LEARNING_SCHEDULER = config["LEARNING_SCHEDULER"]
BATCH_SIZE = int(config["BATCH_SIZE"])
NUM_EPOCHS = int(config["NUM_EPOCHS"])
LINEAR_PROBING = config["LINEAR_PROBING"]
PROBING_EPOCHS = int(config["PROBING_EPOCHS"])
PATIENCE = int(config["PATIENCE"])

LOSS = config["LOSS"]

IMAGE_SIZE = int(config["IMAGE_SIZE"])
MODEL = config["MODEL"]
PRETRAINED = config["PRETRAINED"]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")


def START_seed():
    seed = 9
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    START_seed()

    #run id is date and time of the run
    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")

    #create folder for this run in runs folder
    os.mkdir("/home/vladislav/Documents/Studies/CV703/Assignment 1/runs/" + run_id)

    save_dir = "/home/vladislav/Documents/Studies/CV703/Assignment 1/runs/" + run_id
      
    data_root = "/home/vladislav/Documents/Studies/CV703/Assignment 1/datasets/CUB/CUB_200_2011"

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # write data transform here as per the requirement
    data_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.7, 1.0), antialias=True),

        v2.RandAugment(num_ops=2, magnitude=10),
        v2.RandomErasing(p=0.1),

        v2.ToDtype(torch.float, scale=True),
        v2.Normalize(mean=mean,std=std),
    ])

    val_transform = v2.Compose([
            v2.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            v2.ToTensor(),
            v2.Normalize(mean=mean, std=std)
        ])

    train_dataset_cub = CUBDataset(image_root_path=f"{data_root}", transform=data_transform, split="train")
    test_dataset_cub = CUBDataset(image_root_path=f"{data_root}", transform=val_transform, split="test")


    data_root = "/home/vladislav/Documents/Studies/CV703/Assignment 1/datasets/fgvc-aircraft-2013b"

    num_classes = len(train_dataset_cub.classes)

    cutmix = v2.CutMix(num_classes=num_classes, alpha=1.0)
    mixup = v2.MixUp(num_classes=num_classes, alpha=0.2)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    def collate_fn(batch):
        return cutmix_or_mixup(*default_collate(batch))

    # load in into the torch dataloader to get variable batch size, shuffle 
    train_loader_cub = torch.utils.data.DataLoader(train_dataset_cub, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, num_workers=8, collate_fn=collate_fn)
    test_loader_cub = torch.utils.data.DataLoader(test_dataset_cub, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=8)


    train_dataset_aircraft = FGVCAircraft(root=f"{data_root}", transform=data_transform, train=True)
    test_dataset_aircraft = FGVCAircraft(root=f"{data_root}", transform=val_transform, train=False)

    # load in into the torch dataloader to get variable batch size, shuffle 
    train_loader_aircraft = torch.utils.data.DataLoader(train_dataset_aircraft, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
    test_loader_aircraft = torch.utils.data.DataLoader(test_dataset_aircraft, batch_size=BATCH_SIZE, drop_last=False, shuffle=False)


    data_dir = "/home/vladislav/Documents/Studies/CV703/Assignment 1/datasets/FoodX/food_dataset"

    split = 'train'
    train_df = pd.read_csv(f'{data_dir}/annot/{split}_info.csv', names= ['image_name','label'])
    train_df['path'] = train_df['image_name'].map(lambda x: os.path.join(f'{data_dir}/{split}_set/', x))

    split = 'val'
    val_df = pd.read_csv(f'{data_dir}/annot/{split}_info.csv', names= ['image_name','label'])
    val_df['path'] = val_df['image_name'].map(lambda x: os.path.join(f'{data_dir}/{split}_set/', x))

    train_dataset = FOODDataset(train_df)
    val_dataset = FOODDataset(val_df)

    # load in into the torch dataloader to get variable batch size, shuffle 
    train_loader_food = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
    val_loader_food = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=True)


    ##################### Concatenate CUB Birds and FGVC Aircraft Datasets

    concat_dataset_train = ConcatDataset([train_dataset_cub, train_dataset_aircraft])
    concat_dataset_test = ConcatDataset([test_dataset_cub, test_dataset_aircraft])

    concat_loader_train = torch.utils.data.DataLoader(
                concat_dataset_train,
                batch_size=BATCH_SIZE, shuffle=True,
                num_workers=1, pin_memory=True
                )
    concat_loader_test = torch.utils.data.DataLoader(
                concat_dataset_test,
                batch_size=BATCH_SIZE, shuffle=False,
                num_workers=1, pin_memory=True
                )

    #load model
    model = get_model(MODEL, num_classes, PRETRAINED)


    model.to(DEVICE)
    torch.compile(model)
    
    #load optimizer
    if LOSS == "MSE":
        loss = torch.nn.MSELoss()
    elif LOSS == "L1Loss":
        loss = torch.nn.L1Loss()
    elif LOSS == "SmoothL1Loss":
        loss = torch.nn.SmoothL1Loss()
    elif LOSS == "CrossEntropyLoss":
        loss = torch.nn.CrossEntropyLoss()
        
    else:
        raise Exception("Loss not implemented")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if LEARNING_SCHEDULER == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(NUM_EPOCHS//5))
    else:
        lr_scheduler = None

    early_stopper = EarlyStopper(patience=PATIENCE, min_delta=0.001)

    if LINEAR_PROBING:
        linear_probing_epochs = PROBING_EPOCHS
    else:
        linear_probing_epochs = None
     
    #train model
    results = train(
        model=model,
        train_loader=train_loader_cub,
        val_loader=test_loader_cub,
        criterion=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=DEVICE,
        epochs=NUM_EPOCHS,
        save_dir=save_dir,
        early_stopper=early_stopper,
        linear_probing_epochs=linear_probing_epochs
    )


    train_summary = {
        "config": config,
        "results": results,
    }

    with open(save_dir + "/train_summary.json", "w") as f:
        json.dump(train_summary, f, indent=4)

    plot_results(results["train_loss"], results["val_loss"], "Loss", save_dir)

if __name__ == "__main__":
    main()

