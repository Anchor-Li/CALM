from monai.metrics import ROCAUCMetric
from monai.data import decollate_batch
import torch
import numpy as np

def calcAUC(cfg, y, y_pred):
    try:
        y_pred_softmax = torch.softmax(y_pred, dim=1)
        
        if cfg.param.cls == 2:
            y_pred_probs = y_pred_softmax[:, 1].cpu().numpy().flatten()
            y_labels = y.cpu().numpy().flatten()
        else:
            y_pred_probs = y_pred_softmax.cpu().numpy().flatten()
            y_labels = y.cpu().numpy().flatten()
        
        if len(np.unique(y_labels)) < 2:
            return 0.5
        
        try:
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_labels, y_pred_probs)
        except ImportError:
            auc_metric = ROCAUCMetric()
            y_pred_list = [torch.tensor([prob], device=y_pred.device) for prob in y_pred_probs]
            y_label_list = [torch.tensor([label], device=y.device) for label in y_labels]
            
            auc_metric(y_pred_list, y_label_list)
            auc = auc_metric.aggregate().item()
            auc_metric.reset()
            return auc
    except Exception as e:
        print(f"AUC calculation error: {str(e)}")
        return 0.5


def calcACC(y, y_pred):
    acc_value = torch.eq(y_pred.argmax(dim=1), y)
    acc_metric = acc_value.sum().item() / len(acc_value)
    return acc_metric


def calcMetrics(cfg, y_pred_bins, y_gt_bins, y, y_pred, e=1e-6):

    y_pred_bins = np.array(y_pred_bins, dtype = np.int8)
    y_gt_bins = np.array(y_gt_bins, dtype = np.int8)
    
    tp = int(((y_pred_bins == 1) * (y_gt_bins == 1)).sum())
    fp = int(((y_pred_bins == 1) * (y_gt_bins == 0)).sum())
    tn = int(((y_pred_bins == 0) * (y_gt_bins == 0)).sum())
    fn = int(((y_pred_bins == 0) * (y_gt_bins == 1)).sum())

    sensitivity = float(tp / (tp + fn + e))
    specificity = float(tn / (tn + fp + e))
    acc = float((tp + tn) / (tp + fp + tn + fn + e))

    try:
        auc_metric = calcAUC(cfg, y, y_pred)
    except:
        auc_metric = 0

    try:
        from sklearn.metrics import average_precision_score
        probs = torch.softmax(y_pred, dim=1)[:, 1].detach().cpu().numpy()
        labels_np = y.detach().cpu().numpy()
        pr_auc = float(average_precision_score(labels_np, probs))
    except Exception:
        pr_auc = 0.0

    return sensitivity, specificity, acc, auc_metric, pr_auc
