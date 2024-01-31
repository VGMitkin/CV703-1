from __future__ import print_function, division

from datasets import CUBDataset, FGVCAircraft, FOODDataset
from engine import train
from model import get_model
from utils import plot_results, get_concat_set, EarlyStopper, WarmupLR

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sam.sam import SAM
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

datasets = {0: CUBDataset, 1: FGVCAircraft, 2: FOODDataset}
optimizers = {0: optim.AdamW, 1: SAM}

LEARNING_RATE = float(config["LEARNING_RATE"])
LEARNING_SCHEDULER = config["LEARNING_SCHEDULER"]
DECAY_STEP = int(config["DECAY_STEP"])
OPTIMIZER = int(config["OPTIMIZER"])
BATCH_SIZE = int(config["BATCH_SIZE"])
NUM_EPOCHS = int(config["NUM_EPOCHS"])

FINETUNE = config["FINETUNE"]
FINETUNE_EPOCHS = int(config["FINETUNE_EPOCHS"])
FINETUNE_LR = float(config["FINETUNE_LR"])
WARMUP_EPOCHS = int(config["WARMUP_EPOCHS"])
WARMUP_LR = float(config["WARMUP_LR"])
EARLY_STOPPING = config["EARLY_STOPPING"]
PATIENCE = int(config["PATIENCE"])

LOSS = config["LOSS"]
LABEL_SMOOTHING = float(config["LABEL_SMOOTHING"])

IMAGE_SIZE = int(config["IMAGE_SIZE"])
MODEL = config["MODEL"]
PRETRAINED = config["PRETRAINED"]
FREEZE = config["FREEZE"]
UNFREEZE_LAYERS = config["UNFREEZE_LAYERS"]

DATASET = config["DATASET"]
CONCAT_DATASET = config["CONCAT_DATASET"]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")


def START_seed():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    START_seed()

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

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

    if CONCAT_DATASET:
        train_dataset, test_dataset, class_names = get_concat_set(data_transform, val_transform)
    else:
        dataset = datasets[DATASET]
        data_root = f"/home/vladislav/Documents/Studies/CV703/Assignment 1/datasets/{datasets[DATASET].__name__}"

        train_dataset = dataset(image_root_path=f"{data_root}", transform=data_transform, split="train")
        test_dataset = dataset(image_root_path=f"{data_root}", transform=val_transform, split="test")

        class_names = train_dataset.classes
        
    num_classes = len(class_names)

    cutmix = v2.CutMix(num_classes=num_classes, alpha=0.2)
    mixup = v2.MixUp(num_classes=num_classes, alpha=0.2)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    def collate_fn(batch):
        return cutmix_or_mixup(*default_collate(batch)) if MODEL == "TransConvNeXtV2Base" else None

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, num_workers=8, collate_fn=collate_fn)
    test_loader= DataLoader(test_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False, num_workers=8)

    #run id is date and time of the run
    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")

    #create folder for this run in runs folder
    os.mkdir("/home/vladislav/Documents/Studies/CV703/Assignment 1/runs/" + run_id)
    save_dir = "/home/vladislav/Documents/Studies/CV703/Assignment 1/runs/" + run_id

    #load model
    model = get_model(MODEL, num_classes, PRETRAINED, FREEZE)
    unfreeze_order = [3, 2, 1] if UNFREEZE_LAYERS else None


    model.to(DEVICE)
    torch.compile(model)
    
    
    if  LOSS == "CrossEntropyLoss":
        loss = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    else :
        raise Exception("Loss not implemented")
    
    #load optimizer
    if OPTIMIZER == 1:
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(model.parameters(), base_optimizer, rho=2.0, adaptive=True, lr=LEARNING_RATE, weight_decay=0.0005)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=WARMUP_LR)

    lr_scheduler = WarmupLR(optimizer, WARMUP_EPOCHS, WARMUP_LR, LEARNING_RATE)
    lr_scheduler.step()

    results = {
            "train_loss": [],
            "val_loss": [],
            "accuracy": []
        }

    train_summary = {
            "config": config,
            "results": results,
        }
    
    if WARMUP_EPOCHS > 1:
        results = train(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            criterion=loss,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=DEVICE,
            epochs=WARMUP_EPOCHS,
            save_dir=save_dir,
        )
        train_summary["results"]["train_loss"] += results["train_loss"]
        train_summary["results"]["val_loss"] += results["val_loss"]
        train_summary["results"]["accuracy"] += results["accuracy"]

    for param_group in optimizer.param_groups:
        param_group["lr"] = LEARNING_RATE

    if LEARNING_SCHEDULER == "CosineAnnealingLR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(NUM_EPOCHS//DECAY_STEP), eta_min=FINETUNE_LR)
    else:
        lr_scheduler = None

    if EARLY_STOPPING:
        early_stopper = EarlyStopper(patience=PATIENCE, min_delta=0.001)
    else:
        early_stopper = None
    
    #train model
    results = train(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=DEVICE,
        epochs=NUM_EPOCHS,
        save_dir=save_dir,
        early_stopper=early_stopper,
        unfreeze=unfreeze_order
    )

    lr_scheduler.step()

    train_summary["results"]["train_loss"] += results["train_loss"]
    train_summary["results"]["val_loss"] += results["val_loss"]
    train_summary["results"]["accuracy"] += results["accuracy"]

    if FINETUNE:
        for param in model.parameters():
            param.requires_grad = True

        for param_group in optimizer.param_groups:
            param_group["lr"] = FINETUNE_LR
        
        lr_scheduler = None

        results = train(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            criterion=loss,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=DEVICE,
            epochs=FINETUNE_EPOCHS,
            save_dir=save_dir,
        )

        train_summary["results"]["train_loss"] += results["train_loss"]
        train_summary["results"]["val_loss"] += results["val_loss"]
        train_summary["results"]["accuracy"] += results["accuracy"]

    with open(save_dir + "/train_summary.json", "w") as f:
        json.dump(train_summary, f, indent=4)

    plot_results(train_summary["results"]["train_loss"], train_summary["results"]["val_loss"], "Loss", save_dir)
    plot_results(train_summary["results"]["accuracy"], None, "Accuracy", save_dir)

if __name__ == "__main__":
    main()
