import os, easydict, monai, torch, random, json
import numpy as np
import matplotlib.pyplot as plt
from utils.network import CALM
from loguru import logger
from monai.networks.utils import eval_mode
from dataset.dataloader import getDataLoader, val_transform
from utils.metrics import calcMetrics
from utils.tools import load_config, resolve_device, warmup_cuda_linalg, wrap_model_for_devices, unwrap_model
from utils.losses import TMCL

def _to_serializable(obj):
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    try:
        import numpy as _np
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    try:
        return _to_serializable(vars(obj))
    except Exception:
        return str(obj)

def _save_config(cfg, out_path):
    cfg_dict = {}
    for k, v in cfg.items():
        if k == "transforms":
            continue
        cfg_dict[k] = _to_serializable(v)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, ensure_ascii=False, indent=2)

def test(epoch, cfg, logger, net, dataloader, loss_func, which_loader=None, save_dir=None):

    assert which_loader in [None, 'train'], f'Type {which_loader} Error.'

    with eval_mode(net):

        y_gt_bins = list()
        y_pred_bins = list()

        y_pred = torch.tensor([], dtype=torch.float32, device=cfg.param.device)
        y = torch.tensor([], dtype=torch.long, device=cfg.param.device)

        val_loss = 0

        for val_data in dataloader:

            dce, dwi, t2, val_labels, dce_aff, dwi_aff, t2_aff = [d.to(cfg.param.device) for d in val_data]
            out = net(dce, dwi, t2, dce_aff=dce_aff, dwi_aff=dwi_aff, t2_aff=t2_aff)
            if isinstance(out, tuple):
                outputs, aux_logits = out[0], out[1]
                loss_fused = loss_func(outputs, val_labels)
                aux_loss = sum(loss_func(lg, val_labels) for lg in aux_logits) / len(aux_logits)
                loss = loss_fused + 0.2 * aux_loss
            else:
                outputs = out
                loss = loss_func(outputs, val_labels)
            val_loss += loss.item()

            y_pred = torch.cat([y_pred, outputs], dim=0)
            y = torch.cat([y, val_labels], dim=0)

            if cfg.param.cls == 2 and hasattr(cfg, 'infer') and hasattr(cfg.infer, 'threshold'):
                probs = torch.softmax(outputs, dim=1)[:, 1]
                y_pred_bin = (probs >= cfg.infer.threshold).long()
            else:
                y_pred_bin = outputs.argmax(dim=1)
            for i in range(len(y_pred_bin)):
                y_gt_bins.append(val_labels[i].item())
                y_pred_bins.append(y_pred_bin[i].item())

        val_loss /= len(dataloader)

        sensitivity, specificity, acc_metric, auc_metric, pr_auc = \
            calcMetrics(cfg, y_pred_bins, y_gt_bins, y, y_pred, e=1e-6)

        save_preds_csv = bool(getattr(getattr(cfg, "train", None), "save_preds_csv", False))
        if save_preds_csv and save_dir is not None:
            try:
                import os as _os, csv as _csv
                _os.makedirs(save_dir, exist_ok=True)
                out_csv = _os.path.join(save_dir, f'preds_epoch_{epoch}_{"train" if which_loader=="train" else "val"}.csv')
                with open(out_csv, 'w', newline='') as f:
                    w = _csv.writer(f)
                    w.writerow(['prob_pCR','label'])
                    probs_list = torch.softmax(y_pred, dim=1)[:, 1].detach().cpu().numpy().tolist()
                    labels_list = y.detach().cpu().numpy().tolist()
                    for p, l in zip(probs_list, labels_list):
                        w.writerow([p, l])
            except Exception as e:
                logger.warning(f'Fail: {e}')

        if which_loader == 'train':
            logger.info(f'Epoch:{epoch}, Train AUC:{auc_metric:.4f}, Train ACC:{acc_metric:.4f}')
        
        return sensitivity, specificity, acc_metric, auc_metric, pr_auc, val_loss  

def train(cfg_pth):
    if isinstance(cfg_pth, easydict.EasyDict):
        cfg = cfg_pth
    else:
        cfg = load_config(cfg_pth)

    monai.utils.set_determinism(cfg.param.seed)
    device_ids_override = os.environ.get("CODEMIX_DEVICE_IDS")
    device_ids = device_ids_override if device_ids_override is not None else getattr(cfg.param, "device_ids", None)
    cfg.param.device, cfg.param.device_ids = resolve_device(
        getattr(cfg.param, "device", "cuda:0"),
        device_ids,
    )
    warmup_cuda_linalg(cfg.param.device_ids)

    saved_model_root = cfg.param.saved_model_dir
    os.makedirs(saved_model_root, exist_ok=True)
    logger.add(os.path.join(saved_model_root, 'train.log'))

    try:
        cfg_out_pth = os.path.join(saved_model_root, "config_used.json")
        _save_config(cfg, cfg_out_pth)
    except Exception as e:
        logger.warning(f'Fail: {e}')

    train_folds = cfg.dataset.get('train_folds', cfg.dataset.k_fold)
    logger.info(f"Training on {train_folds} out of {cfg.dataset.k_fold} folds.")

    all_folds_metrics = []

    for fold in range(train_folds):
        logger.info(f"========== Begin the training of fold {fold+1}/{cfg.dataset.k_fold} ==========")
        fold_saved_dir = os.path.join(saved_model_root, f'fold_{fold+1}')
        os.makedirs(fold_saved_dir, exist_ok=True)
        
        if cfg.model.arch == 'calm':
            geo_cfg = getattr(cfg.model, 'geo', {})
            net = CALM(
                num_classes=cfg.param.cls,
                norm=getattr(cfg, 'model', {}).get('norm', 'instance'),
                dce_base=getattr(geo_cfg, 'dce_base', 12),
                dwi_base=getattr(geo_cfg, 'dwi_base', 12),
                t2_base=getattr(geo_cfg, 't2_base', 12),
                embed_dim=getattr(geo_cfg, 'embed_dim', 256),
                dist_mm=getattr(geo_cfg, 'dist_mm', 5.0),

                attn_q_chunk=getattr(geo_cfg, 'q_chunk', 512),
                attn_q_topk=getattr(geo_cfg, 'q_topk', 0),
                sag_pool=getattr(geo_cfg, 'sag_pool', (1, 1, 1)),
                use_checkpoint=getattr(geo_cfg, 'use_checkpoint', False),
                ablate_pgm=getattr(geo_cfg, 'ablate_pgm', False),
                ablate_pcia=getattr(geo_cfg, 'ablate_pcia', False),
            ).to(cfg.param.device)
            net = wrap_model_for_devices(net, cfg.param.device_ids)
        else:
            raise ValueError(f"Unknown architecture: {cfg.model.arch}")

        use_tmcl = bool(getattr(cfg.loss, "tmcl_enable", False)) and (cfg.model.arch == "calm")
        tmcl = None
        if use_tmcl:
            geo_cfg = getattr(cfg.model, "geo", {})
            token_dim = int(getattr(geo_cfg, "embed_dim", 256))
            tmcl = TMCL(
                token_dim=token_dim,
                proj_dim=int(getattr(cfg.loss, "tmcl_proj_dim", 128)),
                temperature=float(getattr(cfg.loss, "tmcl_temp", 0.1)),
                margin=float(getattr(cfg.loss, "tmcl_margin", 0.5)),
                intra_weight=float(getattr(cfg.loss, "tmcl_intra_w", 1.0)),
                inter_weight=float(getattr(cfg.loss, "tmcl_inter_w", 0.0)),
                dropout=float(getattr(cfg.loss, "tmcl_dropout", 0.0)),
            ).to(cfg.param.device)

        params = list(net.parameters()) + (list(tmcl.parameters()) if tmcl is not None else [])
        opt = torch.optim.AdamW(params, lr=cfg.loss.lr, weight_decay=cfg.loss.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=getattr(cfg.train, 'scheduler_mode', 'min'),
            factor=getattr(cfg.train, 'scheduler_factor', 0.5),
            patience=getattr(cfg.train, 'scheduler_patience', 10)
        )
        scaler = torch.amp.GradScaler('cuda', enabled=getattr(cfg.train, 'amp', True))
        accum_steps = getattr(cfg.train, 'accum_steps', 1)
        warmup_epochs = getattr(cfg.train, 'warmup_epochs', 0)

        train_loader, val_loader = getDataLoader(cfg, fold)

        if getattr(cfg.loss, 'loss_type', 'weight_cross_entropy') == 'focal':
            from torch.nn.modules.loss import _Loss
            class FocalLoss(_Loss):
                def __init__(self, gamma=2.0, alpha=None, reduction='mean', label_smoothing=0.0):
                    super().__init__(reduction=reduction)
                    self.gamma = gamma
                    self.alpha = alpha
                    self.label_smoothing = label_smoothing
                def forward(self, input, target):
                    ce = torch.nn.functional.cross_entropy(input, target, weight=self.alpha, reduction='none', label_smoothing=self.label_smoothing)
                    pt = torch.softmax(input, dim=1).gather(1, target.view(-1,1)).squeeze(1).clamp(min=1e-6, max=1-1e-6)
                    loss = ((1-pt)**self.gamma) * ce
                    if self.reduction == 'mean':
                        return loss.mean()
                    elif self.reduction == 'sum':
                        return loss.sum()
                    else:
                        return loss
            alpha = None
            if getattr(cfg.loss, 'use_class_weight', True):
                num_classes = cfg.param.cls
                import torch as _torch
                lbls = _torch.tensor(train_loader.dataset.labels, dtype=_torch.long)
                counts = _torch.bincount(lbls, minlength=num_classes).float().clamp(min=1)
                total = counts.sum()
                alpha = (total / (counts * num_classes)).to(cfg.param.device)
            loss_func = FocalLoss(gamma=getattr(cfg.loss, 'focal_gamma', 2.0), alpha=alpha, label_smoothing=getattr(cfg.loss, 'label_smoothing', 0.0))
        else:
            if getattr(cfg.loss, 'use_class_weight', True):
                num_classes = cfg.param.cls
                import torch as _torch
                lbls = _torch.tensor(train_loader.dataset.labels, dtype=_torch.long)
                counts = _torch.bincount(lbls, minlength=num_classes).float().clamp(min=1)
                total = counts.sum()
                weights = (total / (counts * num_classes)).to(cfg.param.device)
                loss_func = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=getattr(cfg.loss, 'label_smoothing', 0.0))
            else:
                loss_func = torch.nn.CrossEntropyLoss(label_smoothing=getattr(cfg.loss, 'label_smoothing', 0.0))

        best_auc_metric = -1
        best_auc_metric_epoch = -1
        best_acc_metric = -1
        best_acc_metric_epoch = -1
        best_val_loss = float('inf')
        best_val_loss_epoch = -1
        train_losses = []
        val_losses = []

        patience = getattr(cfg.early_stopping, 'patience', 0)
        no_improve = 0
        for epoch in range(cfg.param.epochs):
            epoch += 1
            net.train()
            epoch_loss = 0
            
            opt.zero_grad(set_to_none=True)
            for batch_idx, batch_data in enumerate(train_loader):
                dce, dwi, t2, labels, dce_aff, dwi_aff, t2_aff = [d.to(cfg.param.device) for d in batch_data]

                if warmup_epochs and epoch <= warmup_epochs:
                    for pg in opt.param_groups:
                        base_lr = cfg.loss.lr
                        pg['lr'] = base_lr * float(epoch) / float(warmup_epochs)

                with torch.amp.autocast('cuda', enabled=getattr(cfg.train, 'amp', True)):
                    if tmcl is not None:
                        outputs, feats = net(dce, dwi, t2, dce_aff=dce_aff, dwi_aff=dwi_aff, t2_aff=t2_aff, return_features=True)
                    else:
                        outputs = net(dce, dwi, t2, dce_aff=dce_aff, dwi_aff=dwi_aff, t2_aff=t2_aff)
                    ce_loss = loss_func(outputs, labels)
                    loss_raw = ce_loss
                    if tmcl is not None:
                        tmcl_total, tmcl_intra, tmcl_inter = tmcl(feats["tok_pre"], feats["tok_post"], labels)
                        loss_raw = loss_raw + float(getattr(cfg.loss, "tmcl_alpha", 0.1)) * tmcl_total
                    loss = loss_raw / accum_steps

                scaler.scale(loss).backward()
                is_update = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(train_loader))
                if is_update:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                epoch_loss += loss_raw.item()
            epoch_loss /= len(train_loader)
            train_losses.append(epoch_loss)
            
            # torch.save(net.state_dict(), os.path.join(fold_saved_dir, f"epoch_{epoch}.pth"))

            if epoch % cfg.dataset.val_interval == 0:
                sensitivity, specificity, acc, auc, pr_auc, val_loss = test(epoch, cfg, logger, net, val_loader, loss_func, save_dir=fold_saved_dir)
                logger.info(f'Epoch:{epoch}, Train Loss:{epoch_loss:.4f}, Val Loss:{val_loss:.4f}')
                val_losses.append(val_loss)

                if auc > best_auc_metric:
                    best_auc_metric = auc
                    best_auc_metric_epoch = epoch
                    torch.save(unwrap_model(net).state_dict(), os.path.join(fold_saved_dir, "best_auc_model.pth"))
                if acc > best_acc_metric:
                    best_acc_metric = acc
                    best_acc_metric_epoch = epoch
                    torch.save(unwrap_model(net).state_dict(), os.path.join(fold_saved_dir, "best_acc_model.pth"))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_loss_epoch = epoch

                logger.info(f"Epoch:{epoch}, AUC:{auc:.4f}, PR-AUC:{pr_auc:.4f}, ACC:{acc:.4f}, Sens:{sensitivity:.4f}, Spec:{specificity:.4f}")

                logger.info(
                f"Best AUC: {best_auc_metric:.4f} at Epoch: {best_auc_metric_epoch}, "
                f"Best ACC: {best_acc_metric:.4f} at Epoch: {best_acc_metric_epoch}."
                )
                
                monitor = getattr(cfg.train, 'monitor', 'val_loss')

                improved = False
                if monitor == 'val_loss':
                    if val_loss < best_val_loss + 1e-6:
                        improved = True
                elif monitor == 'auc':
                    if auc > best_auc_metric - 1e-6:
                        improved = True
                elif monitor == 'acc':
                    if acc > best_acc_metric - 1e-6:
                        improved = True
                else:
                    if val_loss < best_val_loss + 1e-6:
                        improved = True
                if improved:
                    no_improve = 0
                else:
                    no_improve += 1
                if patience and no_improve >= patience:
                    logger.info(f"Early Stopping")
                    break
                if monitor == 'train_loss':
                    metric = epoch_loss
                elif monitor == 'val_loss':
                    metric = val_loss
                elif monitor == 'auc':
                    metric = auc
                elif monitor == 'acc':
                    metric = acc
                else:
                    metric = val_loss
                scheduler.step(metric)

                if cfg.dataset.train_as_val:
                    test(epoch, cfg, logger, net, train_loader, loss_func, which_loader='train')
            
        
        logger.info(f"========== The training of fold {fold+1} is over ==========")
        logger.info(f"Best AUC: {best_auc_metric:.4f} at Epoch: {best_auc_metric_epoch}")
        logger.info(f"Best ACC: {best_acc_metric:.4f} at Epoch: {best_acc_metric_epoch}")
        
        if cfg.model.arch == 'calm':
            geo_cfg = getattr(cfg.model, 'geo', {})
            best_net = CALM(
                num_classes=cfg.param.cls,
                norm=getattr(cfg, 'model', {}).get('norm', 'instance'),
                dce_base=getattr(geo_cfg, 'dce_base', 12),
                dwi_base=getattr(geo_cfg, 'dwi_base', 12),
                t2_base=getattr(geo_cfg, 't2_base', 12),
                embed_dim=getattr(geo_cfg, 'embed_dim', 256),
                dist_mm=getattr(geo_cfg, 'dist_mm', 5.0),

                attn_q_chunk=getattr(geo_cfg, 'q_chunk', 512),
                attn_q_topk=getattr(geo_cfg, 'q_topk', 0),
                sag_pool=getattr(geo_cfg, 'sag_pool', (1, 1, 1)),
                use_checkpoint=getattr(geo_cfg, 'use_checkpoint', False),
                ablate_pgm=getattr(geo_cfg, 'ablate_pgm', False),
                ablate_pcia=getattr(geo_cfg, 'ablate_pcia', False),
            ).to(cfg.param.device)
            best_net = wrap_model_for_devices(best_net, cfg.param.device_ids)
        else:
            raise ValueError(f"Unknown architecture: {cfg.model.arch}")
        unwrap_model(best_net).load_state_dict(torch.load(os.path.join(fold_saved_dir, "best_auc_model.pth"), map_location=cfg.param.device))
        sensitivity, specificity, acc, auc, _, _ = test(best_auc_metric_epoch, cfg, logger, best_net, val_loader, loss_func)

        all_folds_metrics.append({
            'fold': fold + 1,
            'best_auc': best_auc_metric,
            'best_acc': best_acc_metric,
            'best_auc_epoch': best_auc_metric_epoch,
            'best_acc_epoch': best_acc_metric_epoch
        })

    logger.info("========== The training of all folds is over ==========")
    aucs = []
    accs = []
    for m in all_folds_metrics:
        logger.info(f"Fold {m['fold']}: Best AUC = {m['best_auc']:.4f} (Epoch {m['best_auc_epoch']}), Best ACC = {m['best_acc']:.4f} (Epoch {m['best_acc_epoch']})")
        aucs.append(m['best_auc'])
        accs.append(m['best_acc'])
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    logger.info(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}, Mean ACC: {mean_acc:.4f} ± {std_acc:.4f}")
    logger.info("============================================")


if __name__ == '__main__':
    cfg_pth = '/path/to/your/config.py' 
    train(cfg_pth)
