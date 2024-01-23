import torch
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import quad


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model


def plot_results(train_data, val_data, label, save_dir):

    plt.figure(figsize=(6, 6))
    plt.title(label)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.plot(train_data, label=f'Train {label}')
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
 