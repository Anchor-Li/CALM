import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from contextlib import nullcontext

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, max(1, channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class SiameseEncoder(nn.Module):
    def __init__(self, in_channels, base_filters=16, norm='instance'):
        super(SiameseEncoder, self).__init__()
        nk = str(norm).lower()
        if nk in ('instance', 'instancenorm', 'in'):
            def Norm3d(c):
                return nn.InstanceNorm3d(c)
        elif nk in ('batch', 'batchnorm', 'bn'):
            def Norm3d(c):
                return nn.BatchNorm3d(c)
        elif nk in ('layer', 'layernorm', 'ln'):
            def Norm3d(c):
                return nn.GroupNorm(1, c)
        else:
            raise ValueError(f'Unsupported norm: {norm}')
        
        self.conv1 = nn.Conv3d(in_channels, base_filters, kernel_size=3, padding=1)
        self.bn1 = Norm3d(base_filters)
        self.act1 = nn.PReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.se1 = SEBlock(base_filters)
        self.skip1 = nn.Conv3d(in_channels, base_filters, kernel_size=1)

        self.conv2 = nn.Conv3d(base_filters, base_filters*2, kernel_size=3, padding=1)
        self.bn2 = Norm3d(base_filters*2)
        self.act2 = nn.PReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.skip2 = nn.Conv3d(base_filters, base_filters*2, kernel_size=1)

        self.conv3 = nn.Conv3d(base_filters*2, base_filters*4, kernel_size=3, padding=1)
        self.bn3 = Norm3d(base_filters*4)
        self.act3 = nn.PReLU()
        self.pool3 = nn.AdaptiveAvgPool3d((8, 8, 8)) 
        self.skip3 = nn.Conv3d(base_filters*2, base_filters*4, kernel_size=1)

    def forward(self, x):
        x1 = self.act1(self.bn1(self.conv1(x)))
        x1 = self.se1(x1)
        res1 = self.skip1(x)
        x = x1 + res1
        x = self.pool1(x)
        
        x2 = self.act2(self.bn2(self.conv2(x)))
        res2 = self.skip2(x)
        x = x2 + res2
        x = self.pool2(x)
        
        x3 = self.act3(self.bn3(self.conv3(x)))
        res3 = self.skip3(x)
        x = x3 + res3
        x = self.pool3(x)
        
        return x

class DCEPhaseEncoder3D(nn.Module):
    def __init__(self, in_channels=1, base=12, norm='instance'):
        super().__init__()
        nk = str(norm).lower()
        if nk in ('instance', 'instancenorm', 'in'):
            def Norm3d(c): return nn.InstanceNorm3d(c)
        elif nk in ('batch', 'batchnorm', 'bn'):
            def Norm3d(c): return nn.BatchNorm3d(c)
        else:
            def Norm3d(c): return nn.GroupNorm(1, c)
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base, 3, padding=1),
            Norm3d(base),
            nn.GELU(),
            nn.Conv3d(base, base, 3, padding=1, groups=base),
            Norm3d(base),
            nn.GELU(),
            nn.Conv3d(base, base*2, 1),
            Norm3d(base*2),
            nn.GELU(),
        )
        self.down = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(base*2, base*4, 3, padding=1),
            Norm3d(base*4),
            nn.GELU(),
            nn.MaxPool3d(2),
            nn.Conv3d(base*4, base*4, 3, padding=1),
            Norm3d(base*4),
            nn.GELU(),
        )
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.out_dim = base*4

    def forward(self, x):
        x = self.stem(x)
        x = self.down(x)
        feat = x
        vec = self.gap(x).view(x.size(0), -1)
        return feat, vec

class TimeTransformer(nn.Module):
    def __init__(self, token_dim=64, nhead=4, layers=2, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=nhead, dim_feedforward=token_dim*2, dropout=dropout, batch_first=True)
        self.tx = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.pool = nn.Linear(token_dim, token_dim)

    def forward(self, tokens):
        x = self.tx(tokens)
        x = self.pool(x.mean(dim=1))
        return x

class GridSampler3D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, affine_x, affine_target, shape_target):
        B, C, H, W, D = x.shape
        Ht, Wt, Dt = shape_target
        device = x.device
        out_dtype = x.dtype

        autocast_ctx = (
            torch.amp.autocast(device_type=device.type, enabled=False)
            if device.type in ("cuda", "cpu")
            else nullcontext()
        )
        with autocast_ctx:
            x_f = x.float()
            affine_x_f = affine_x.float()
            affine_target_f = affine_target.float()

            i = torch.arange(Ht, device=device, dtype=torch.float32)
            j = torch.arange(Wt, device=device, dtype=torch.float32)
            k = torch.arange(Dt, device=device, dtype=torch.float32)
            ii, jj, kk = torch.meshgrid(i, j, k, indexing='ij')

            coords = torch.stack([ii, jj, kk, torch.ones_like(ii)], dim=0).view(4, -1)
            coords = coords.unsqueeze(0).expand(B, -1, -1)

            P_world = torch.bmm(affine_target_f, coords)
            P_src = torch.linalg.solve(affine_x_f, P_world)

            isrc = P_src[:, 0, :]
            jsrc = P_src[:, 1, :]
            ksrc = P_src[:, 2, :]

            x_grid = 2.0 * ksrc / (D - 1) - 1.0
            y_grid = 2.0 * jsrc / (W - 1) - 1.0
            z_grid = 2.0 * isrc / (H - 1) - 1.0

            grid = torch.stack([x_grid, y_grid, z_grid], dim=-1).view(B, Ht, Wt, Dt, 3)
            out = F.grid_sample(x_f, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        return out.to(dtype=out_dtype)

class PatchEmbed3D(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, img_size):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_size = tuple(patch_size)
        self.img_size = tuple(img_size)
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=True)

        if any(s % p != 0 for s, p in zip(self.img_size, self.patch_size)):
            raise ValueError(f'img_size {self.img_size} must be divisible by patch_size {self.patch_size}')
        self.grid_size = tuple(s // p for s, p in zip(self.img_size, self.patch_size))
        self.num_patches = int(self.grid_size[0] * self.grid_size[1] * self.grid_size[2])

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Warp3D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, flow):
        raise NotImplementedError("Warp3D is deprecated.")

def _affine_scale(affine, scale_h=1.0, scale_w=1.0, scale_d=1.0):
    if affine.dim() != 3 or affine.size(-1) != 4 or affine.size(-2) != 4:
        raise ValueError(f"affine must be [B,4,4], got {tuple(affine.shape)}")
    s = torch.tensor(
        [[scale_h, 0.0, 0.0, 0.0], [0.0, scale_w, 0.0, 0.0], [0.0, 0.0, scale_d, 0.0], [0.0, 0.0, 0.0, 1.0]],
        device=affine.device,
        dtype=affine.dtype,
    )
    return affine @ s

def _world_coords_from_affine(affine, shape):
    if len(shape) != 3:
        raise ValueError(f"shape must be (H,W,D), got {shape}")
    b = affine.size(0)
    h, w, d = int(shape[0]), int(shape[1]), int(shape[2])
    device = affine.device
    dtype = affine.dtype
    ii = torch.arange(h, device=device, dtype=dtype)
    jj = torch.arange(w, device=device, dtype=dtype)
    kk = torch.arange(d, device=device, dtype=dtype)
    I, J, K = torch.meshgrid(ii, jj, kk, indexing="ij")
    ones = torch.ones_like(I)
    P = torch.stack([I, J, K, ones], dim=0).view(4, -1)
    W = torch.matmul(affine, P)
    xyz = W[:, :3, :].transpose(1, 2).contiguous()
    return xyz

def _world_bounds_from_affine(affine, shape):
    if len(shape) != 3:
        raise ValueError(f"shape must be (H,W,D), got {shape}")
    b = affine.size(0)
    h, w, d = int(shape[0]), int(shape[1]), int(shape[2])
    device = affine.device
    dtype = affine.dtype
    corners = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [float(h - 1), 0.0, 0.0],
            [0.0, float(w - 1), 0.0],
            [0.0, 0.0, float(d - 1)],
            [float(h - 1), float(w - 1), 0.0],
            [float(h - 1), 0.0, float(d - 1)],
            [0.0, float(w - 1), float(d - 1)],
            [float(h - 1), float(w - 1), float(d - 1)],
        ],
        device=device,
        dtype=dtype,
    )
    ones = torch.ones(8, 1, device=device, dtype=dtype)
    P = torch.cat([corners, ones], dim=1).t().unsqueeze(0).expand(b, -1, -1)
    W = torch.bmm(affine, P)[:, :3, :].transpose(1, 2).contiguous()
    mn = W.amin(dim=1, keepdim=True)
    mx = W.amax(dim=1, keepdim=True)
    return mn, mx

def _world_coords_from_affine_select(affine, shape, flat_idx):
    if len(shape) != 3:
        raise ValueError(f"shape must be (H,W,D), got {shape}")
    if flat_idx.dim() != 2:
        raise ValueError(f"flat_idx must be [B,K], got {tuple(flat_idx.shape)}")
    b = affine.size(0)
    h, w, d = int(shape[0]), int(shape[1]), int(shape[2])
    if flat_idx.size(0) != b:
        raise ValueError(f"flat_idx batch {flat_idx.size(0)} must match affine batch {b}")
    device = affine.device
    dtype = affine.dtype
    idx = flat_idx.to(device=device, dtype=torch.long)
    k = idx.remainder(d)
    tmp = idx.div(d, rounding_mode="floor")
    j = tmp.remainder(w)
    i = tmp.div(w, rounding_mode="floor")
    ones = torch.ones_like(i, device=device, dtype=dtype)
    P = torch.stack([i.to(dtype), j.to(dtype), k.to(dtype), ones], dim=1)
    W = torch.bmm(affine, P)
    xyz = W[:, :3, :].transpose(1, 2).contiguous()
    return xyz

def _normalize_coords_with_bounds(xyz, mn, mx, eps=1e-6):
    return 2.0 * (xyz - mn) / (mx - mn + eps) - 1.0

def _normalize_coords_pair(xyz_a, xyz_b, eps=1e-6):
    mn = torch.minimum(xyz_a.amin(dim=1, keepdim=True), xyz_b.amin(dim=1, keepdim=True))
    mx = torch.maximum(xyz_a.amax(dim=1, keepdim=True), xyz_b.amax(dim=1, keepdim=True))
    return 2.0 * (xyz_a - mn) / (mx - mn + eps) - 1.0, 2.0 * (xyz_b - mn) / (mx - mn + eps) - 1.0

class SiameseGridEncoder(nn.Module):
    def __init__(self, in_channels, base_filters=16, norm="instance", pool3d=(2, 2, 1), layers=2):
        super().__init__()
        nk = str(norm).lower()
        if nk in ("instance", "instancenorm", "in"):
            def Norm3d(c): return nn.InstanceNorm3d(c)
        elif nk in ("batch", "batchnorm", "bn"):
            def Norm3d(c): return nn.BatchNorm3d(c)
        elif nk in ("layer", "layernorm", "ln"):
            def Norm3d(c): return nn.GroupNorm(1, c)
        else:
            raise ValueError(f"Unsupported norm: {norm}")

        blocks = []
        c_in = int(in_channels)
        c = int(base_filters)
        for i in range(int(layers)):
            blocks.append(nn.Conv3d(c_in, c, kernel_size=3, padding=1))
            blocks.append(Norm3d(c))
            blocks.append(nn.GELU())
            blocks.append(nn.Conv3d(c, c, kernel_size=3, padding=1, groups=c))
            blocks.append(Norm3d(c))
            blocks.append(nn.GELU())
            blocks.append(nn.MaxPool3d(kernel_size=pool3d, stride=pool3d))
            c_in = c
            c = c * 2
        self.net = nn.Sequential(*blocks)
        self.out_channels = c_in
        self.pool3d = tuple(int(x) for x in pool3d)
        self.layers = int(layers)

    def downsample_factor(self):
        sh, sw, sd = self.pool3d
        return (sh ** self.layers, sw ** self.layers, sd ** self.layers)

    def forward(self, x):
        return self.net(x)

class PGM(nn.Module):
    """Physiology-gated Guidance Module"""

    def __init__(self, in_channels=3, base=12, norm="instance"):
        super().__init__()
        nk = str(norm).lower()
        if nk in ("instance", "instancenorm", "in"):
            def Norm3d(c): return nn.InstanceNorm3d(c)
        elif nk in ("batch", "batchnorm", "bn"):
            def Norm3d(c): return nn.BatchNorm3d(c)
        elif nk in ("layer", "layernorm", "ln"):
            def Norm3d(c): return nn.GroupNorm(1, c)
        else:
            raise ValueError(f"Unsupported norm: {norm}")

        self.net = nn.Sequential(
            nn.Conv3d(in_channels, base, 3, padding=1),
            Norm3d(base),
            nn.GELU(),
            nn.Conv3d(base, base, 3, padding=1, groups=base),
            Norm3d(base),
            nn.GELU(),
            nn.Conv3d(base, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

class CoordMLP(nn.Module):
    def __init__(self, out_dim, hidden_dim=64, dropout=0.0):
        super().__init__()
        od = int(out_dim)
        hd = int(hidden_dim)
        dp = float(dropout)
        self.net = nn.Sequential(
            nn.Linear(3, hd),
            nn.GELU(),
            nn.LayerNorm(hd),
            nn.Dropout(dp) if dp > 0 else nn.Identity(),
            nn.Linear(hd, od),
        )

    def forward(self, xyz_norm):
        return self.net(xyz_norm)

class PCIA(nn.Module):
    """Physical-distance-weighted Cross-view Inter-modal Attention"""

    def __init__(self, q_channels, kv_channels=None, attn_dim=None, coord_hidden=64, dropout=0.0, dist_mm=5.0, q_chunk=512):
        super().__init__()
        qc = int(q_channels)
        kc = int(kv_channels) if kv_channels is not None else qc
        d = int(attn_dim) if attn_dim is not None else qc
        self.q_channels = qc
        self.kv_channels = kc
        self.attn_dim = d
        self.q_proj = nn.Linear(qc, d, bias=True)
        self.k_proj = nn.Linear(kc, d, bias=True)
        self.v_proj = nn.Linear(kc, d, bias=True)
        self.out_proj = nn.Linear(d, qc, bias=True)
        self.coord_mlp = CoordMLP(out_dim=d, hidden_dim=int(coord_hidden), dropout=float(dropout))
        self.dist_mm = float(dist_mm)
        self.q_chunk = int(q_chunk)

    def _attend_chunked(self, q, k, v, q_xyz, k_xyz):
        b, nq, d = q.shape
        nk = k.shape[1]
        out = torch.zeros(b, nq, d, device=q.device, dtype=q.dtype)
        scale = 1.0 / math.sqrt(float(d))
        thr2 = (self.dist_mm ** 2)
        finfo = torch.finfo(q.dtype)
        neg_inf = finfo.min
        for s in range(0, nq, self.q_chunk):
            e = min(s + self.q_chunk, nq)
            qc = q[:, s:e, :]
            qxc = q_xyz[:, s:e, :]
            dist2 = (qxc.unsqueeze(2) - k_xyz.unsqueeze(1)).pow(2).sum(dim=-1)
            mask = dist2 <= thr2
            logits = torch.bmm(qc, k.transpose(1, 2)) * scale
            logits = logits.masked_fill(~mask, neg_inf)
            attn = torch.softmax(logits, dim=-1)
            row_has = mask.any(dim=-1, keepdim=True)
            attn = torch.where(row_has, attn, torch.zeros_like(attn))
            out[:, s:e, :] = torch.bmm(attn, v)
        return out

    def forward(self, axial_feat, axial_affine, sagittal_feat, sagittal_affine, q_idx=None, q_mask=None):
        b, c, ha, wa, da = axial_feat.shape
        if c != self.q_channels:
            raise ValueError(f"axial_feat channels {c} must equal q_channels {self.q_channels}")
        _, _, hs, ws, ds = sagittal_feat.shape
        ax_tok = axial_feat.flatten(2).transpose(1, 2)
        sg_tok = sagittal_feat.flatten(2).transpose(1, 2)

        sg_xyz = _world_coords_from_affine(sagittal_affine, (hs, ws, ds))
        if q_idx is not None and q_idx.numel() == 0:
            q_idx = None
        if q_mask is not None and q_mask.numel() == 0:
            q_mask = None

        if q_idx is not None:
            if q_idx.dim() != 2:
                raise ValueError(f"q_idx must be [B,K], got {tuple(q_idx.shape)}")
            if q_idx.size(0) != b:
                raise ValueError(f"q_idx batch {q_idx.size(0)} must match B={b}")
            q_idx = q_idx.to(device=ax_tok.device, dtype=torch.long)

            ax_mn, ax_mx = _world_bounds_from_affine(axial_affine, (ha, wa, da))
            sg_mn, sg_mx = _world_bounds_from_affine(sagittal_affine, (hs, ws, ds))
            mn = torch.minimum(ax_mn, sg_mn)
            mx = torch.maximum(ax_mx, sg_mx)

            ax_xyz_sel = _world_coords_from_affine_select(axial_affine, (ha, wa, da), q_idx)
            ax_norm_sel = _normalize_coords_with_bounds(ax_xyz_sel, mn, mx)
            sg_norm = _normalize_coords_with_bounds(sg_xyz, mn, mx)

            k = self.k_proj(sg_tok) + self.coord_mlp(sg_norm)
            v = self.v_proj(sg_tok)

            idx_feat = q_idx.unsqueeze(-1).expand(-1, -1, c)
            ax_tok_sel = ax_tok.gather(1, idx_feat)
            q = self.q_proj(ax_tok_sel) + self.coord_mlp(ax_norm_sel)
            out_sel = self._attend_chunked(q, k, v, ax_xyz_sel, sg_xyz)
            delta_sel = self.out_proj(out_sel)

            delta_full = torch.zeros_like(ax_tok)
            delta_full.scatter_(1, idx_feat, delta_sel)
            fused = ax_tok + delta_full
        else:
            ax_xyz = _world_coords_from_affine(axial_affine, (ha, wa, da))
            ax_norm, sg_norm = _normalize_coords_pair(ax_xyz, sg_xyz)
            q = self.q_proj(ax_tok) + self.coord_mlp(ax_norm)
            k = self.k_proj(sg_tok) + self.coord_mlp(sg_norm)
            v = self.v_proj(sg_tok)
            out = self._attend_chunked(q, k, v, ax_xyz, sg_xyz)

            if q_mask is not None:
                if q_mask.dim() == 5:
                    qb = q_mask.view(b, -1) > 0
                elif q_mask.dim() == 2:
                    qb = q_mask > 0
                else:
                    raise ValueError(f"Unsupported q_mask shape: {tuple(q_mask.shape)}")
                if qb.shape != (b, ax_tok.size(1)):
                    raise ValueError(f"q_mask flattened shape {tuple(qb.shape)} must match (B, Nq)=({b}, {ax_tok.size(1)})")
                out = out * qb.unsqueeze(-1)

            fused = ax_tok + self.out_proj(out)
        fused_map = fused.transpose(1, 2).view(b, c, ha, wa, da)
        return fused_map

class TimeFusionTransformer(nn.Module):
    def __init__(self, token_dim=256, nhead=8, layers=2, dropout=0.1):
        super().__init__()
        d = int(token_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.time_embed = nn.Embedding(2, d)
        enc = nn.TransformerEncoderLayer(d_model=d, nhead=int(nhead), dim_feedforward=d * 4, dropout=float(dropout), batch_first=True, activation="gelu")
        self.tx = nn.TransformerEncoder(enc, num_layers=int(layers))
        self.norm = nn.LayerNorm(d)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.time_embed.weight, std=0.02)

    def forward(self, tok_pre, tok_post, extra_pre=None, extra_post=None):
        b = tok_pre.size(0)
        t0 = self.time_embed.weight[0].view(1, -1)
        t1 = self.time_embed.weight[1].view(1, -1)
        x_pre = (tok_pre + t0).unsqueeze(1)
        x_post = (tok_post + t1).unsqueeze(1)

        if extra_pre is not None:
            if extra_pre.dim() != 3 or extra_pre.size(0) != b:
                raise ValueError(f"extra_pre must be [B,N,D], got {tuple(extra_pre.shape)}")
            x_pre = torch.cat([x_pre, extra_pre + t0.view(1, 1, -1)], dim=1)
        if extra_post is not None:
            if extra_post.dim() != 3 or extra_post.size(0) != b:
                raise ValueError(f"extra_post must be [B,N,D], got {tuple(extra_post.shape)}")
            x_post = torch.cat([x_post, extra_post + t1.view(1, 1, -1)], dim=1)

        x = torch.cat([x_pre, x_post], dim=1)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.tx(x)
        x = self.norm(x)
        return x[:, 0]

class CALM(nn.Module):
    """Cross-View Spatial-Temporal Aligning for Longitudinal Multimodal data"""

    def __init__(
        self,
        num_classes=2,
        norm="instance",
        dce_base=12,
        dwi_base=12,
        t2_base=12,
        embed_dim=256,
        dist_mm=5.0,
        attn_q_chunk=512,
        attn_q_topk=0,
        sag_pool=(1, 1, 1),
        use_checkpoint=False,
        ablate_pgm=False,
        ablate_pcia=False,
    ):
        super().__init__()
        self.norm = norm
        self.use_checkpoint = bool(use_checkpoint)
        self.ablate_pgm = bool(ablate_pgm)
        self.ablate_pcia = bool(ablate_pcia)
        self.q_topk = int(attn_q_topk) if attn_q_topk is not None else 0
        if isinstance(sag_pool, (tuple, list)):
            self.sag_pool = (int(sag_pool[0]), int(sag_pool[1]), int(sag_pool[2]))
        else:
            sp = int(sag_pool)
            self.sag_pool = (sp, sp, 1)
        if self.sag_pool == (1, 1, 1):
            self.sag_pool_layer = nn.Identity()
        else:
            self.sag_pool_layer = nn.AvgPool3d(kernel_size=self.sag_pool, stride=self.sag_pool)

        self.phase_enc = DCEPhaseEncoder3D(in_channels=1, base=int(dce_base), norm=norm)
        self.time_tx = TimeTransformer(token_dim=self.phase_enc.out_dim, nhead=8, layers=3, dropout=0.1)

        self.pgm = PGM(in_channels=3, base=int(dwi_base), norm=norm)
        self.sampler = GridSampler3D()
        self.dwi_enc = SiameseEncoder(in_channels=3, base_filters=int(dwi_base), norm=norm)
        self.dwi_proj = nn.Sequential(
            nn.Linear(int(dwi_base) * 4, int(embed_dim)),
            nn.GELU(),
            nn.LayerNorm(int(embed_dim)),
        )

        self.t2_enc = SiameseGridEncoder(in_channels=1, base_filters=int(t2_base), norm=norm, pool3d=(2, 2, 1), layers=2)
        self.pcia = PCIA(q_channels=self.phase_enc.out_dim, kv_channels=self.t2_enc.out_channels, attn_dim=embed_dim, coord_hidden=64, dropout=0.0, dist_mm=float(dist_mm), q_chunk=int(attn_q_chunk))
        self.sag_proj = nn.Sequential(
            nn.Linear(int(self.t2_enc.out_channels), int(embed_dim)),
            nn.GELU(),
            nn.LayerNorm(int(embed_dim)),
        )

        self.axial_proj = nn.Sequential(
            nn.Linear(self.phase_enc.out_dim, int(embed_dim)),
            nn.GELU(),
            nn.LayerNorm(int(embed_dim)),
        )
        self.fusion_time = TimeFusionTransformer(token_dim=int(embed_dim), nhead=8, layers=2, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(int(embed_dim), int(embed_dim)),
            nn.GELU(),
            nn.LayerNorm(int(embed_dim)),
            nn.Dropout(0.2),
            nn.Linear(int(embed_dim), int(num_classes)),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _encode_dce_timepoint(self, dce_tp, dwi_tp, dce_aff_tp, dwi_aff_tp, t2_tp, t2_aff_tp):
        b, ph, _, _, _ = dce_tp.shape
        if self.ablate_pgm:
            pgm_gate = None
        else:
            pgm_gate = self.pgm(dwi_tp)
        feats_vec = []
        feats_map = []
        pgm_aligned = None
        dce_feat_aff = None
        for p in range(ph):
            x = dce_tp[:, p:p + 1, ...]
            if self.use_checkpoint and self.training:
                try:
                    from torch.utils.checkpoint import checkpoint as _ckpt
                    f, v = _ckpt(self.phase_enc, x, use_reentrant=False)
                except TypeError:
                    from torch.utils.checkpoint import checkpoint as _ckpt
                    f, v = _ckpt(self.phase_enc, x)
            else:
                f, v = self.phase_enc(x)
            if dce_feat_aff is None:
                dce_feat_aff = _affine_scale(dce_aff_tp, 4.0, 4.0, 4.0)
            if pgm_gate is not None:
                if pgm_aligned is None:
                    pgm_aligned = self.sampler(pgm_gate, dwi_aff_tp, dce_feat_aff, f.shape[2:])
                f = f * (1.0 + pgm_aligned)
            feats_vec.append(F.adaptive_avg_pool3d(f, 1).view(b, -1))
            feats_map.append(f)
        seq = torch.stack(feats_vec, dim=1)
        v_dce = self.time_tx(seq)
        f_axial = torch.stack(feats_map, dim=1).mean(dim=1)

        if self.use_checkpoint and self.training:
            try:
                from torch.utils.checkpoint import checkpoint as _ckpt
                f_sag = _ckpt(self.t2_enc, t2_tp, use_reentrant=False)
            except TypeError:
                from torch.utils.checkpoint import checkpoint as _ckpt
                f_sag = _ckpt(self.t2_enc, t2_tp)
        else:
            f_sag = self.t2_enc(t2_tp)
        f_sag = self.sag_pool_layer(f_sag)
        sh, sw, sd = self.t2_enc.downsample_factor()
        sh *= int(self.sag_pool[0])
        sw *= int(self.sag_pool[1])
        sd *= int(self.sag_pool[2])
        t2_feat_aff = _affine_scale(t2_aff_tp, float(sh), float(sw), float(sd))

        q_idx = None
        q_mask = None
        if pgm_aligned is not None:
            g = pgm_aligned.view(b, -1)
            if self.q_topk is not None and self.q_topk > 0 and self.q_topk < g.size(1):
                q_idx = torch.topk(g, self.q_topk, dim=1).indices
            else:
                q_mask = self._make_q_mask(pgm_aligned)
        else:
            if self.q_topk is not None and self.q_topk > 0:
                score = f_axial.abs().mean(dim=1).view(b, -1)
                k = min(int(self.q_topk), score.size(1))
                if k > 0:
                    q_idx = torch.topk(score, k, dim=1).indices

        extra = []
        if self.ablate_pgm:
            if self.use_checkpoint and self.training:
                try:
                    from torch.utils.checkpoint import checkpoint as _ckpt
                    f_dwi = _ckpt(self.dwi_enc, dwi_tp, use_reentrant=False)
                except TypeError:
                    from torch.utils.checkpoint import checkpoint as _ckpt
                    f_dwi = _ckpt(self.dwi_enc, dwi_tp)
            else:
                f_dwi = self.dwi_enc(dwi_tp)
            v_dwi = F.adaptive_avg_pool3d(f_dwi, 1).view(b, -1)
            tok_dwi = self.dwi_proj(v_dwi)
            extra.append(tok_dwi.unsqueeze(1))

        if self.ablate_pcia:
            tok_axial = self.axial_proj(v_dce)
            v_sag = F.adaptive_avg_pool3d(f_sag, 1).view(b, -1)
            tok_sag = self.sag_proj(v_sag)
            extra.insert(0, tok_sag.unsqueeze(1))
            extra_tokens = torch.cat(extra, dim=1) if len(extra) > 0 else None
            return tok_axial, extra_tokens

        if self.use_checkpoint and self.training:
            q_idx_t = q_idx if q_idx is not None else f_axial.new_empty((b, 0), dtype=torch.long)
            q_mask_t = q_mask if q_mask is not None else f_axial.new_empty((b, 0), dtype=torch.bool)

            def _cross(axial_feat, axial_affine, sag_feat, sag_affine, q_idx_in, q_mask_in):
                return self.pcia(axial_feat, axial_affine, sag_feat, sag_affine, q_idx=q_idx_in, q_mask=q_mask_in)

            try:
                from torch.utils.checkpoint import checkpoint as _ckpt
                f_fused = _ckpt(_cross, f_axial, dce_feat_aff, f_sag, t2_feat_aff, q_idx_t, q_mask_t, use_reentrant=False)
            except TypeError:
                from torch.utils.checkpoint import checkpoint as _ckpt
                f_fused = _ckpt(_cross, f_axial, dce_feat_aff, f_sag, t2_feat_aff, q_idx_t, q_mask_t)
        else:
            f_fused = self.pcia(f_axial, dce_feat_aff, f_sag, t2_feat_aff, q_idx=q_idx, q_mask=q_mask)
        v_fused = F.adaptive_avg_pool3d(f_fused, 1).view(b, -1)
        tok = self.axial_proj(v_fused)
        extra_tokens = torch.cat(extra, dim=1) if len(extra) > 0 else None
        return tok, extra_tokens

    def _make_q_mask(self, pgm_aligned):
        b, _, h, w, d = pgm_aligned.shape
        g = pgm_aligned.view(b, -1)
        thr = g.mean(dim=1, keepdim=True)
        mask = g >= thr
        return mask.view(b, 1, h, w, d)

    def forward(self, dce, dwi, t2, dce_aff=None, dwi_aff=None, t2_aff=None, return_features=False):
        dce_pre = dce[:, 0, ...]
        dce_post = dce[:, 1, ...]
        dwi_pre = dwi[:, 0, ...]
        dwi_post = dwi[:, 1, ...]
        t2_pre = t2[:, 0, ...]
        t2_post = t2[:, 1, ...]
        if dce_aff is None or dwi_aff is None or t2_aff is None:
            raise ValueError("CALM requires dce_aff, dwi_aff, t2_aff")
        tok_pre, extra_pre = self._encode_dce_timepoint(dce_pre, dwi_pre, dce_aff[:, 0], dwi_aff[:, 0], t2_pre, t2_aff[:, 0])
        tok_post, extra_post = self._encode_dce_timepoint(dce_post, dwi_post, dce_aff[:, 1], dwi_aff[:, 1], t2_post, t2_aff[:, 1])
        fused = self.fusion_time(tok_pre, tok_post, extra_pre=extra_pre, extra_post=extra_post)
        out = self.head(fused)
        if return_features:
            return out, {"tok_pre": tok_pre, "tok_post": tok_post, "fused": fused}
        return out
