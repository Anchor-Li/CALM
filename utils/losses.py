import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dists, targets):
        batch_size = dists.size(0)
        targets = targets.long()
        correct_class_dists = dists[torch.arange(batch_size), targets]
        loss = correct_class_dists.mean()
        return loss

class ContrastivePrototypeLoss(nn.Module):
    def __init__(self, lambda_proto=0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.proto = PrototypeLoss()
        self.lambda_proto = lambda_proto
        
    def forward(self, logits, dists, targets):
        ce_loss = self.ce(logits, targets)
        proto_loss = self.proto(dists, targets)
        return ce_loss + self.lambda_proto * proto_loss, ce_loss, proto_loss


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, eps=1e-8):
        super().__init__()
        self.temperature = float(temperature)
        self.eps = float(eps)

    def forward(self, z, labels):
        if z.dim() != 2:
            raise ValueError(f"z must be [B,D], got {tuple(z.shape)}")
        b = z.size(0)
        if b <= 1:
            return z.new_tensor(0.0)
        if labels.dim() != 1 or labels.size(0) != b:
            raise ValueError(f"labels must be [B], got {tuple(labels.shape)}")

        z = F.normalize(z, dim=1)
        sim = torch.matmul(z, z.t()) / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True).values

        device = z.device
        eye = torch.eye(b, device=device, dtype=torch.bool)
        valid = ~eye
        labels = labels.view(-1, 1)
        pos = (labels == labels.t()) & valid

        denom = torch.logsumexp(sim.masked_fill(~valid, float("-inf")), dim=1, keepdim=True)
        log_prob = sim - denom

        pos_count = pos.sum(dim=1)
        has_pos = pos_count > 0
        if not has_pos.any():
            return z.new_tensor(0.0)

        loss_i = -(log_prob.masked_fill(~pos, 0.0).sum(dim=1) / pos_count.clamp_min(1).to(log_prob.dtype))
        loss = loss_i[has_pos].mean()
        return loss


class TMCL(nn.Module):
    def __init__(
        self,
        token_dim,
        proj_dim=128,
        temperature=0.1,
        margin=0.5,
        intra_weight=1.0,
        inter_weight=0.0,
        dropout=0.0,
    ):
        super().__init__()
        td = int(token_dim)
        pd = int(proj_dim)
        dp = float(dropout)
        self.margin = float(margin)
        self.intra_weight = float(intra_weight)
        self.inter_weight = float(inter_weight)
        self.supcon = SupervisedContrastiveLoss(temperature=float(temperature))
        self.proj = nn.Sequential(
            nn.Linear(td * 3, pd),
            nn.GELU(),
            nn.LayerNorm(pd),
            nn.Dropout(dp) if dp > 0 else nn.Identity(),
            nn.Linear(pd, pd),
        )

    def forward(self, tok_pre, tok_post, labels, eps=1e-6):
        if tok_pre.dim() != 2 or tok_post.dim() != 2:
            raise ValueError(f"tok_pre/tok_post must be [B,D], got {tuple(tok_pre.shape)} and {tuple(tok_post.shape)}")
        if tok_pre.shape != tok_post.shape:
            raise ValueError(f"tok_pre/tok_post shape mismatch: {tuple(tok_pre.shape)} vs {tuple(tok_post.shape)}")
        b, d = tok_pre.shape
        if labels.dim() != 1 or labels.size(0) != b:
            raise ValueError(f"labels must be [B], got {tuple(labels.shape)}")

        y = labels.to(dtype=tok_pre.dtype)
        cos = F.cosine_similarity(tok_pre, tok_post, dim=1).clamp(-1.0 + eps, 1.0 - eps)
        dist = 1.0 - cos
        m = tok_pre.new_tensor(self.margin)
        intra = (1.0 - y) * dist.pow(2) + y * (torch.clamp(m - dist, min=0.0)).pow(2)
        intra_loss = intra.mean()

        baseline = 0.5 * (tok_pre + tok_post)
        delta = tok_post - tok_pre
        norm_delta = delta / (tok_pre.abs() + tok_post.abs() + eps)
        traj_raw = torch.cat([baseline, delta, norm_delta], dim=1)
        z = self.proj(traj_raw)
        inter_loss = self.supcon(z, labels)

        total = tok_pre.new_tensor(0.0)
        if self.intra_weight != 0.0:
            total = total + float(self.intra_weight) * intra_loss
        if self.inter_weight != 0.0:
            total = total + float(self.inter_weight) * inter_loss
        return total, intra_loss, inter_loss
