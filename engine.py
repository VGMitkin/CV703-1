import torch
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
    with torch.no_grad():
        for x_batch, y_batch in tqdm(val_dl, leave=False):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss_history += loss.item()
    return loss_history / len(val_dl)


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
        linear_probing_epochs=None
          ):
    
    results = {
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
    }
    best_val_loss = np.inf

    for epoch in range(1, epochs + 1):
        if linear_probing_epochs is not None:
            if epoch == linear_probing_epochs:
                for param in model.parameters():
                    param.requires_grad = True

        print(f"Epoch {epoch}:")
        train_loss = train_epoch(model, train_loader, criterion, optimizer,  device)
        print(f"Train Loss: {train_loss:.4f}")

        
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.5f}")
        results["learning_rate"].append(optimizer.param_groups[0]["lr"])

        if lr_scheduler is not None:
            lr_scheduler.step()

        results["train_loss"].append(train_loss)

        val_loss = val_epoch(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}")
        print()

        results["val_loss"].append(val_loss)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "learning_rate": optimizer.param_groups[0]["lr"]})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, save_dir + "/best_model.pth")

        save_model(model, save_dir + "/last_model.pth")

        optimizer.zero_grad()
        gc.collect()
        torch.cuda.empty_cache()

        if early_stopper is not None:
            if early_stopper.early_stop(val_loss):
                print("Early stopping")
                break

    return results