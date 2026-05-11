import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


import sys
import os
import math
import random
import numpy as np
from tqdm import tqdm
from thop import profile,clever_format
from sklearn import metrics
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, cohen_kappa_score,log_loss, brier_score_loss
)
import scipy.ndimage as ndi
from scipy.spatial import cKDTree



class SoftDiceLoss(torch.nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()
 
    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        m1 = logits.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
 
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

loss_function_s = SoftDiceLoss()
loss_function_c = torch.nn.CrossEntropyLoss()
consistency_loss = torch.nn.BCELoss()


def train_one_epoch_seg(model, optimizer, data_loader_s, device, epoch, epochs, frozen_layers):
    model.train()
    for layer in frozen_layers:
        layer.eval()
        for param in layer.parameters(): param.requires_grad = False
    if epoch == 0: 
        trainable_modules = [
            name for name, module in model.named_children()
            if any(param.requires_grad for param in module.parameters())
        ]
        print("Trainable layers:", ", ".join(trainable_modules))

    accu_loss_s = torch.zeros(1).to(device)
    accu_iou = torch.zeros(1).to(device)
    accu_dice = torch.zeros(1).to(device)
    accu_pre = torch.zeros(1).to(device)
    accu_rec = torch.zeros(1).to(device)
    accu_asd = torch.zeros(1).to(device)

    data_loader_s = tqdm(data_loader_s, file=sys.stdout)
    for i, data in enumerate(data_loader_s):
        images_s, NoduleMask = data
        images_s, NoduleMask = images_s.to(device), NoduleMask.to(device)

        _, pred_mask= model(images_s)
 
        loss_s = loss_function_s(pred_mask, NoduleMask)
        accu_loss_s += loss_s.detach()

        loss_s.backward()
        optimizer.step()
        optimizer.zero_grad()  

        # IoU
        iou = calculate_batch_iou(pred_mask, NoduleMask)
        accu_iou += iou

        # Dice / Pre / Rec
        dice, pre, rec = calculate_batch_metrics(pred_mask, NoduleMask)
        accu_dice += dice
        accu_pre += pre
        accu_rec += rec

        # ASD
        asd = calculate_batch_asd(pred_mask, NoduleMask)
        accu_asd += asd.to(device)
        
        data_loader_s.desc = "[epoch {}/{} train_seg]".format(epoch + 1, epochs)

    num = i + 1
    metrics = {
        "iou": (accu_iou / num).item(),
        "dice": (accu_dice / num).item(),
        "precision": (accu_pre / num).item(),
        "recall": (accu_rec / num).item(),
        "asd": (accu_asd / num).item(),
    }

    train_loss_s = accu_loss_s.item() / num
    return train_loss_s, metrics


@torch.no_grad()
def test_seg(model, data_loader_s, device, epoch, epochs):
    model.eval()

    accu_loss_s = torch.zeros(1).to(device)
    accu_iou = torch.zeros(1).to(device)
    accu_dice = torch.zeros(1).to(device)
    accu_pre = torch.zeros(1).to(device)
    accu_rec = torch.zeros(1).to(device)
    accu_asd = torch.zeros(1).to(device)

    data_loader_s = tqdm(data_loader_s, file=sys.stdout)

    for i, data in enumerate(data_loader_s):
        images_s, NoduleMask = data
        images_s = images_s.to(device)
        NoduleMask = NoduleMask.to(device)

        _, pred_mask = model(images_s)

        loss_s = loss_function_s(pred_mask, NoduleMask)
        accu_loss_s += loss_s.detach()

        # IoU
        iou = calculate_batch_iou(pred_mask, NoduleMask)
        accu_iou += iou

        # Dice / Pre / Rec
        dice, pre, rec = calculate_batch_metrics(pred_mask, NoduleMask)
        accu_dice += dice
        accu_pre += pre
        accu_rec += rec

        # ASD
        asd = calculate_batch_asd(pred_mask, NoduleMask)
        accu_asd += asd.to(device)

        data_loader_s.desc = f"[epoch {epoch+1}/{epochs} test_seg]"

    num = i + 1
    metrics = {
        "iou": (accu_iou / num).item(),
        "dice": (accu_dice / num).item(),
        "precision": (accu_pre / num).item(),
        "recall": (accu_rec / num).item(),
        "asd": (accu_asd / num).item(),
    }

    val_loss_s = accu_loss_s.item() / num
    return val_loss_s, metrics

def train_one_epoch_cls(model, optimizer, data_loader_c, device, epoch, epochs, frozen_layers=[]):
    model.train()
    for layer in frozen_layers:
        layer.eval()
        for param in layer.parameters(): param.requires_grad = False
    if epoch == 0: 
        trainable_modules = [
            name for name, module in model.named_children()
            if any(param.requires_grad for param in module.parameters())
        ]
        print("Trainable layers:", ", ".join(trainable_modules))

    accu_loss_c = torch.zeros(1).to(device) 
    pred = []
    true = []
    all_probs = []

    data_loader_c = tqdm(data_loader_c, file=sys.stdout)
    for i, data in enumerate(data_loader_c):
        images_c, labels = data
        images_c, labels = images_c.to(device), labels.to(device)

        cls, _= model(images_c)

        loss_c = loss_function_c(cls, labels)
        accu_loss_c += loss_c.detach()

        pred_classes = torch.max(cls, dim=1)[1]
        pred.extend(pred_classes.cpu().numpy())
        true.extend(labels.cpu().numpy())

        probs = F.softmax(cls, dim=1)[:, 1]
        all_probs.extend(probs.detach().cpu().numpy())

        loss_c.backward()
        optimizer.step()
        optimizer.zero_grad()  
        data_loader_c.desc = "[epoch {}/{} train_cls]".format(epoch + 1, epochs)

    train_loss_c = accu_loss_c.item() / (i + 1)

    accuracy = accuracy_score(true, pred)
    precision = precision_score(true, pred)
    recall = recall_score(true, pred)
    f1 = f1_score(true, pred)
    kappa = cohen_kappa_score(true, pred)


    ap = average_precision_score(true, all_probs)
    auc = roc_auc_score(true, all_probs)
    logloss = log_loss(true, all_probs)
    brier = brier_score_loss(true, all_probs)

    result = [accuracy,precision,recall,f1,kappa,ap,auc,logloss,brier]

    return train_loss_c, result

@torch.no_grad()
def test_cls(model, data_loader_c, device, epoch, epochs, return_preds=False):
    model.eval()
    accu_loss_c = torch.zeros(1).to(device) 
    pred = []
    true = []
    all_probs = []

    data_loader_c = tqdm(data_loader_c, file=sys.stdout)
    for i, data in enumerate(data_loader_c):
        images_c, labels = data
        images_c, labels = images_c.to(device), labels.to(device)
        
        cls, _= model(images_c)

        loss_c = loss_function_c(cls, labels)
        accu_loss_c += loss_c.detach()

        pred_classes = torch.max(cls, dim=1)[1]
        pred.extend(pred_classes.cpu().numpy())
        true.extend(labels.cpu().numpy())

        probs = F.softmax(cls, dim=1)[:, 1]
        all_probs.extend(probs.cpu().numpy())

        data_loader_c.desc = "[epoch {}/{} test_cls]".format(epoch + 1, epochs)
        
    val_loss_c = accu_loss_c.item() / (i + 1)

    accuracy = accuracy_score(true, pred)
    precision = precision_score(true, pred)
    recall = recall_score(true, pred)
    f1 = f1_score(true, pred)
    kappa = cohen_kappa_score(true, pred)


    ap = average_precision_score(true, all_probs)
    auc = roc_auc_score(true, all_probs)
    logloss = log_loss(true, all_probs)
    brier = brier_score_loss(true, all_probs)

    result = [accuracy,precision,recall,f1,kappa,ap,auc,logloss,brier]
    if return_preds: 
        return val_loss_c, result, pred, true, all_probs
    else: 
        return val_loss_c, result

def show_flops_params(model, device):
    input = torch.randn(1, 1, 224, 224).to(device)
    model = model.to(device)
    model.eval()

    for m in model.modules():
        if 'total_ops' in m._buffers:
            del m._buffers['total_ops']
        if 'total_params' in m._buffers:
            del m._buffers['total_params']

    with open(os.devnull, 'w') as fnull:
        original_stdout = sys.stdout
        sys.stdout = fnull
        macs, params = profile(model, inputs=(input,))
        sys.stdout = original_stdout

    macs, params = clever_format([macs, params], "%.3f")
    print(f"flops={macs}  params={params}")





def calculate_batch_iou(image, mask, threshold=0.5):
    batch_size = image.size(0)

    image = image.detach().cpu().numpy()
    binary_image = (image > threshold).astype(np.uint8)

    mask = mask.detach().cpu().numpy()
    binary_mask = (mask > threshold).astype(np.uint8)

    iou_values = []
    
    for i in range(batch_size):
        img1 = binary_image[i, 0, :, :]
        img2 = binary_mask[i, 0, :, :]
        
        intersection = np.logical_and(img1, img2)
        union = np.logical_or(img1, img2)
        
        iou = np.sum(intersection) / (np.sum(union) + 1e-6)
        iou_values.append(iou)
    
    mean_iou = np.mean(iou_values)
    return mean_iou

def calculate_batch_metrics(pred, target, eps=1e-6):
    """
    pred, target: [B, 1, H, W] or [B, H, W]
    """
    pred = (pred > 0.5).float()
    target = target.float()

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    TP = (pred * target).sum(dim=1)
    FP = (pred * (1 - target)).sum(dim=1)
    FN = ((1 - pred) * target).sum(dim=1)

    dice = (2 * TP + eps) / (2 * TP + FP + FN + eps)
    precision = (TP + eps) / (TP + FP + eps)
    recall = (TP + eps) / (TP + FN + eps)

    return dice.mean(), precision.mean(), recall.mean()

def get_surface(mask):
    return mask ^ ndi.binary_erosion(mask)

def asd_single(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan

    pred_surf = get_surface(pred)
    gt_surf = get_surface(gt)

    pred_pts = np.argwhere(pred_surf)
    gt_pts = np.argwhere(gt_surf)

    tree_pred = cKDTree(pred_pts)
    tree_gt = cKDTree(gt_pts)

    d1, _ = tree_gt.query(pred_pts, k=1)
    d2, _ = tree_pred.query(gt_pts, k=1)

    return (d1.mean() + d2.mean()) / 2

def calculate_batch_asd(pred, target):
    pred = (pred > 0.5).cpu().numpy()
    target = target.cpu().numpy()

    asd_list = []
    for i in range(pred.shape[0]):
        asd = asd_single(pred[i, 0], target[i, 0])
        if not np.isnan(asd):
            asd_list.append(asd)

    if len(asd_list) == 0:
        return torch.tensor(0.0)

    return torch.tensor(np.mean(asd_list))
