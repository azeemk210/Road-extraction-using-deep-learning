#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 20:36:05 2021

@author: azeem
"""
CHECKPOINT_DIR = "checkpoints"
import torch,os
import torchvision
from dataset import Road_dataset
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    if not os.path.exists(CHECKPOINT_DIR):
        os.mkdir(CHECKPOINT_DIR)
    print("=> Saving checkpoint")
    
    torch.save(state, os.path.join(CHECKPOINT_DIR,filename))

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = Road_dataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = Road_dataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def get_data_loader(
    data_dir,
    data_maskdir,
    batch_size,
    transforms,
    split='train',
    num_workers=4,
    pin_memory=True,
):
    ds = Road_dataset(
        image_dir=data_dir,
        mask_dir=data_maskdir,
        transform=transforms,
    )
    if split == 'train':
        data_loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=True,
        )
    else:
        data_loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False,
        )

    return data_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            # print(f"preds.shape={preds.shape}, y.shape={y.shape}")
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    if not os.path.exists(folder):
        os.mkdir(folder)
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, os.path.join(folder,f"pred_{idx}.png")
        )
        torchvision.utils.save_image(y, os.path.join(folder,f"{idx}.png"))

    model.train()
def printt(suoo):
    print(suoo)


def compute_f1_iou(loader, model, device="cuda"):
    # global_tp = 0
    # global_p = 0
    # global_p_pred = 0
    instance_f1_scores = []
    instance_ious = []
    model.eval()

    with torch.no_grad():
        for x, y in loader: 
            x = x.to(device)
            y = y.to(device)
            y_pred = torch.sigmoid(model(x))
            y_pred_class = (y_pred>0.5).int()
            f1 = f1_score(y.cpu().ravel(), y_pred_class.cpu().ravel(), labels=[0,1])
            instance_f1_scores.append(f1)

            tn, fp, fn, tp = confusion_matrix(y.cpu().ravel(), y_pred_class.cpu().ravel(), labels=[0,1]).ravel()
            iou = tp/(tp+fn+fp + 1e-8) 
            instance_ious.append(iou)

            # global_tp += tp
            # global_p += y.sum()
            # global_p_pred += y_pred_class.sum()

        # prec = global_tp/global_p_pred
        # rec = global_tp/global_p
        # global_f1_score = 2*prec*rec/(prec+rec)
        # print(f"global f1 score: {global_f1_score:.2f}")
        mean_f1_score  = np.mean(instance_f1_scores)
        mean_iou = np.mean(instance_ious)
        return mean_f1_score, mean_iou

    