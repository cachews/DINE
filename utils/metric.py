import torch
import torch.nn as nn
import numpy as np

def select_criterion_dual(loss="MAE", alpha=1, beta=1, weight=None, label_smooth=0):
    if loss == "MAE":
        loss_demand = nn.L1Loss()
    if loss == "MSE":
        loss_demand = nn.MSELoss()
    elif loss == "Huber":
        loss_demand = nn.HuberLoss()
    else:
        raise ValueError(f"{loss} training criterion not found")

    loss_occurrence = nn.BCEWithLogitsLoss(pos_weight=weight)

    return DualIntermittentLoss(loss_occurrence, loss_demand, alpha=alpha, beta=beta, label_smooth=label_smooth)

class DualIntermittentLoss(nn.Module):
    def __init__(self, loss_occurrence, loss_demand, alpha=1, beta=1, label_smooth=0):
        super().__init__()
        self.loss_demand = loss_demand # MSE
        self.loss_occurrence = loss_occurrence #BCE
        self.alpha = alpha
        self.beta = beta
        self.label_smooth = label_smooth

    def forward(self, pred_occurrence, pred_demand, true):
        true_occurrence_hard = (true > 0)
        true_occurrence = true_occurrence_hard.float()
        if self.label_smooth > 0:
            true_occurrence = true_occurrence * (1 - self.label_smooth) + 0.5 * self.label_smooth

        loss_o = self.loss_occurrence(pred_occurrence, true_occurrence)

        mask_occurrence = true_occurrence_hard.bool()

        if mask_occurrence.any():
            loss_d = self.loss_demand(pred_demand[mask_occurrence], true[mask_occurrence])
        else:
            loss_d = torch.tensor(0.0, device=pred_demand.device)

        return self.alpha * loss_o + self.beta * loss_d
    
def metric_for_series_scaled(pred, true):
    # ensure floats
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)

    metrics = {
        "MAE": MAE(pred, true),
        "RMSE": RMSE(pred, true),
        "MAAPE": MAAPE(pred, true),
        "ME": ME(pred, true),
    }

    try:
        metrics["SPEC"] = SPEC_fast(pred, true, a1=0.5, a2=0.5)
    except:
        print("SPEC calculation failed.")
    
    try:
        metrics["SPEC_windowed"] = SPEC_windowed(pred, true, window_size=28, a1=0.5, a2=0.5)
    except:
        print("SPEC windowed calculation failed.")


def MAE(pred, true):
    return np.mean(np.abs(true - pred))

def RMSE(pred, true):
    return np.sqrt(np.mean((true - pred) ** 2))

def ME(pred, true):
    return np.mean(true - pred)

def MAAPE(pred, true, eps=1e-8):
    percentage_error = np.abs((true - pred) / (true + eps))
    return np.mean(np.arctan(percentage_error))

def SPEC(y_pred, y_true, a1=0.5, a2=0.5):
    """
    Stock-keeping-oriented Prediction Error Costs (SPEC)
    Read more in the :ref:`https://arxiv.org/abs/2004.10537`.
    Lifted from DominikMartin/spec_metric
    """

    assert len(y_true) > 0 and len(y_pred) > 0
    assert len(y_true) == len(y_pred)

    sum_n = 0
    for t in range(1, len(y_true) + 1):
        sum_t = 0
        for i in range(1, t + 1):
            delta1 = np.sum([y_k for y_k in y_true[:i]]) - np.sum([f_j for f_j in y_pred[:t]])
            delta2 = np.sum([f_k for f_k in y_pred[:i]]) - np.sum([y_j for y_j in y_true[:t]])

            sum_t = sum_t + np.max([0, a1 * np.min([y_true[i - 1], delta1]), a2 * np.min([y_pred[i - 1], delta2])]) * (
                        t - i + 1)
        sum_n = sum_n + sum_t
    return sum_n / len(y_true)

def SPEC_fast(y_pred, y_true, a1=0.5, a2=0.5):
    """
    Optimized SPEC implementation using prefix sums.
    Semantically equivalent to the reference version.
    """

    T = len(y_true)
    assert T > 0 and len(y_pred) == T

    # Prefix sums (1-based indexing via padding)
    Y = np.concatenate(([0.0], np.cumsum(y_true)))
    F = np.concatenate(([0.0], np.cumsum(y_pred)))

    spec_total = 0.0

    for t in range(1, T + 1):
        sum_t = 0.0
        Ft = F[t]
        Yt = Y[t]

        for i in range(1, t + 1):
            delta1 = Y[i] - Ft
            delta2 = F[i] - Yt

            term1 = a1 * min(y_true[i - 1], delta1)
            term2 = a2 * min(y_pred[i - 1], delta2)

            cost = max(0.0, term1, term2)
            sum_t += cost * (t - i + 1)

        spec_total += sum_t

    return spec_total / T

# spec applied on rolling window then averaged
def SPEC_windowed(y_pred, y_true, window_size=28, a1=0.5, a2=0.5, fast=True):
    preds_len = len(y_pred)
    spec_series = []

    for i in range(preds_len - window_size + 1):
        pred_window = y_pred[i:i + window_size]
        true_window = y_true[i:i + window_size]

        if fast:
            spec_w = SPEC_fast(pred_window, true_window, a1=a1, a2=a2)
        else:
            spec_w = SPEC(pred_window, true_window, a1=a1, a2=a2)
        spec_series.append(spec_w)

    return np.mean(spec_series)