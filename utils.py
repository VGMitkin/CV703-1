from datasets import CUBDataset, FGVCAircraft

import torch
from torch.utils.data import ConcatDataset
from matplotlib import pyplot as plt


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


def plot_results(train_data, val_data=None, label=None, save_dir=None):

    plt.figure(figsize=(6, 6))
    plt.title(label)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.plot(train_data, label=f'Train {label}')
    if val_data:
        plt.plot(val_data, label=f'Validation {label}')
    plt.legend()
    plt.savefig(save_dir + f'/{label}.png')


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class Permute(torch.nn.Module):
    """This module returns a view of the tensor input with its dimensions permuted.

    Args:
        dims (List[int]): The desired ordering of dimensions
    """

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)


class WarmupLR:
    def __init__(self, optimizer, warmup_epochs, start_lr, target_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.current_epoch = 0

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = target_lr / warmup_epochs

    def step(self):
        if self.current_epoch <= self.warmup_epochs:
            lr = self.start_lr + ((self.target_lr - self.start_lr) / self.warmup_epochs) * self.current_epoch
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

        self.current_epoch += 1


def get_concat_set(data_transform, val_transform, batch_size, collate_fn=None):
    train_dataset_cub = CUBDataset(image_root_path="/home/vladislav/Documents/Studies/CV703/Assignment 1/datasets/CUBDataset", transform=data_transform, split="train")
    test_dataset_cub = CUBDataset(image_root_path="/home/vladislav/Documents/Studies/CV703/Assignment 1/datasets/CUBDataset", transform=val_transform, split="test")

    train_dataset_aircraft = FGVCAircraft(root="/home/vladislav/Documents/Studies/CV703/Assignment 1/datasets/FGVCAircraft", transform=data_transform, train=True)
    test_dataset_aircraft = FGVCAircraft(root="/home/vladislav/Documents/Studies/CV703/Assignment 1/datasets/FGVCAircraft", transform=val_transform, train=False)

    concat_dataset_train = ConcatDataset([train_dataset_cub, train_dataset_aircraft])
    concat_dataset_test = ConcatDataset([test_dataset_cub, test_dataset_aircraft])
    
    classes_1 = concat_dataset_train.datasets[0].classes
    classes_2 = concat_dataset_train.datasets[1].classes

    class_names = [*classes_1, *classes_2]
    
    return concat_dataset_train, concat_dataset_test, class_names