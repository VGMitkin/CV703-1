import torch
from sam.sam import SAM
import numpy as np
from tqdm import tqdm
import gc as gc
from utils import save_model
import wandb

def train_epoch(model, 
                train_dl,
                criterion,
                optimizer, 
                device
                ):
    model.train()
    loss_history = 0
    for x_batch, y_batch in tqdm(train_dl, leave=False):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        if isinstance(optimizer, SAM):
            loss.backward()
            optimizer.first_step(zero_grad=True)
            criterion(model(x_batch), y_batch).backward()
            optimizer.second_step(zero_grad=True)
        else:
            loss.backward()
            optimizer.step()
        loss_history += loss.item()
    return loss_history / len(train_dl)


@torch.no_grad()
def val_epoch(model, 
              val_dl, 
              criterion,
              device
              ):
    model.eval()
    loss_history = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in tqdm(val_dl, leave=False):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch)
            _, predicted = torch.max(y_pred.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            loss = criterion(y_pred, y_batch)
            loss_history += loss.item()
    return loss_history / len(val_dl), correct / total


def get_accuracy(model, val_dl, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in tqdm(val_dl, leave=False):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch)
            _, predicted = torch.max(y_pred.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    return correct / total


def train(model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        lr_scheduler,
        device,
        epochs,
        save_dir,
        early_stopper=None,
        unfreeze=None
          ):
    
    results = {
        "train_loss": [],
        "val_loss": [],
        "accuracy": []
    }
    best_val_loss = np.inf

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}:")
        train_loss = train_epoch(model, train_loader, criterion, optimizer,  device)
        print(f"Train Loss: {train_loss:.4f}")

        if lr_scheduler:
            lr_scheduler.step()

        results["train_loss"].append(train_loss)
        val_loss, accuracy = val_epoch(model, val_loader, criterion, device)
        results["accuracy"].append(accuracy)

        print(f"Val Loss: {val_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print()

        results["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, save_dir + "/best_model.pth")

        wandb.log({"train_loss": train_loss, "val_loss": val_loss,"accuracy": accuracy})

        save_model(model, save_dir + "/last_model.pth")

        optimizer.zero_grad()
        gc.collect()
        torch.cuda.empty_cache()

        if early_stopper:
            if early_stopper.early_stop(val_loss):
                print("Early stopping")
                break

        if unfreeze and epoch % 10 == 0:
            stage = unfreeze[epoch // 10 - 1]

            for param in model.features[stage].parameters():
                param.requires_grad = True

    return results