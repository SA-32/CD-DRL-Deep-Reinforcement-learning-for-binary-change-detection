"""
metrics.py
----------
Implements the evaluation metrics defined in the paper's "Evaluation"
subsection: DICE, PPV, SEN, IoU, Boundary IoU (BIoU), and 95% Hausdorff
Distance (HD95). All are computed per-image and can be averaged over a
test set exactly as in Table 2 / Table 3 / Table 4.
"""

import numpy as np
import torch
from scipy.ndimage import binary_erosion, distance_transform_edt

EPS = 1e-4  # matches the paper's epsilon


def _to_numpy_binary(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    return (x > 0.5).astype(np.uint8)


def dice_ppv_sen_iou(pred, gt):
    """
    pred, gt: binary masks (numpy arrays or torch tensors), same shape (H, W)
    Returns dict with DICE, PPV, SEN, IoU as defined in the paper:

        DICE = (2*TP + eps) / (T + P + eps)
        PPV  = (TP + eps) / (TP + FP + eps)
        SEN  = (TP + eps) / (TP + FN + eps)
        IoU  = (TP + eps) / (T + P - TP + eps)
    """
    pred = _to_numpy_binary(pred)
    gt = _to_numpy_binary(gt)

    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))

    T = np.sum(gt == 1)          # number of ground-truth positive points
    P = np.sum(pred == 1)        # number of predicted positive points

    dice = (2 * tp + EPS) / (T + P + EPS)
    ppv = (tp + EPS) / (tp + fp + EPS)
    sen = (tp + EPS) / (tp + fn + EPS)
    iou = (tp + EPS) / (T + P - tp + EPS)

    return {"DICE": dice, "PPV": ppv, "SEN": sen, "IoU": iou}


def _boundary_map(mask, dilation_ratio=0.02):
    """
    Extracts the boundary pixels of a binary mask, following the standard
    Boundary IoU definition (Cheng et al., CVPR 2021), used by the paper.
    dilation_ratio controls boundary thickness relative to image diagonal.
    """
    mask = mask.astype(np.uint8)
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = max(1, int(round(dilation_ratio * img_diag)))

    eroded = binary_erosion(mask, iterations=dilation, border_value=0)
    boundary = mask - eroded.astype(np.uint8)
    return boundary


def boundary_iou(pred, gt, dilation_ratio=0.02):
    """
    BIoU = |Gd ∩ Pd| / |Gd ∪ Pd|
    where Gd, Pd are the boundary regions of ground truth / prediction.
    """
    pred = _to_numpy_binary(pred)
    gt = _to_numpy_binary(gt)

    pred_b = _boundary_map(pred, dilation_ratio)
    gt_b = _boundary_map(gt, dilation_ratio)

    inter = np.sum((pred_b == 1) & (gt_b == 1))
    union = np.sum((pred_b == 1) | (gt_b == 1))
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


def hausdorff_distance_95(pred, gt):
    """
    HD95 = max( 95th percentile of d(P,G), 95th percentile of d(G,P) )
    Surface distances computed via Euclidean distance transform.
    Returns np.nan if either mask is empty (undefined HD).
    """
    pred = _to_numpy_binary(pred)
    gt = _to_numpy_binary(gt)

    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")

    # distance transform of the complement gives, at each foreground pixel
    # of the other mask, the distance to the nearest pixel of *this* mask.
    dt_gt = distance_transform_edt(1 - gt)
    dt_pred = distance_transform_edt(1 - pred)

    d_pred_to_gt = dt_gt[pred == 1]
    d_gt_to_pred = dt_pred[gt == 1]

    hd95_pred_to_gt = np.percentile(d_pred_to_gt, 95)
    hd95_gt_to_pred = np.percentile(d_gt_to_pred, 95)

    return max(hd95_pred_to_gt, hd95_gt_to_pred)


def compute_all_metrics(pred, gt):
    """Convenience wrapper returning all six metrics used in the paper."""
    out = dice_ppv_sen_iou(pred, gt)
    out["BIoU"] = boundary_iou(pred, gt)
    out["HD95"] = hausdorff_distance_95(pred, gt)
    return out


def average_metrics(list_of_dicts):
    """Averages a list of per-image metric dicts, ignoring NaNs (e.g. HD95
    when a slice has no foreground)."""
    keys = list_of_dicts[0].keys()
    out = {}
    for k in keys:
        vals = np.array([d[k] for d in list_of_dicts], dtype=np.float64)
        out[k] = float(np.nanmean(vals))
    return out
