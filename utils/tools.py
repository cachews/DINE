import torch
import numpy as np
import math

class EarlyStopping:
    def __init__(self, patience=3, verbose=True, delta=1e-4):
        self.patience = patience
        self.verbose = verbose
        # self.delta = delta
        self.delta = 0

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.vali_loss_min = np.Inf

    def __call__(self, vali_loss, model, path):
        score = -vali_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(vali_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}. (Min: {self.vali_loss_min:.6f}, current: {vali_loss:.6f})")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(vali_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, vali_loss, model, path):
        if self.verbose:
            print(f"Validation loss decreased ({self.vali_loss_min:.6f} --> {vali_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.vali_loss_min = vali_loss

def adjust_learning_rate(optimizer, epoch, args):
    if args.lr_scheduler == "type1":
        lr = args.learning_rate * (0.5 ** (epoch - 1))
    elif args.lr_scheduler == "cosine":
        lr = args.learning_rate * 0.5 * (1 + math.cos(math.pi * epoch / args.train_epochs))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

        if args.verbose:
            print(f"Updating learning rate to {lr}")