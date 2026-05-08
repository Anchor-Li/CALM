from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch, monai
import numpy as np
from monai.transforms import (Compose, LoadImage, AddChannel, ScaleIntensity, ToTensor, RandGaussianNoise, RandAdjustContrast, RandRicianNoise, Resize, Affine, Flip, Rotate90)
from monai.transforms.transform import Randomizable
from sklearn.model_selection import StratifiedKFold

class MyTransform:
    def __call__(self, image):
        return (image - image.mean()) / (image.std() + 1e-8)

class SharedTransform(Randomizable):
    def __init__(self, aug_cfg=None, target_size=(128, 128, 16), enable_resize=True, enable_geom=True):
        super().__init__()
        self.loader = LoadImage(image_only=False)
        self.add_ch = AddChannel()
        self.target_size = tuple(target_size) if target_size is not None else None
        self.enable_resize = bool(enable_resize)
        self.enable_geom = bool(enable_geom)
        self.resize = Resize(spatial_size=self.target_size) if (self.enable_resize and self.target_size is not None) else None
        self.scale = ScaleIntensity()
        self.my_trans = MyTransform()
        
        self.affine_t = None
        if aug_cfg is not None:
            self.geom_prob = getattr(aug_cfg, 'geom_prob', 0.3)
            self.rotate_lim = getattr(aug_cfg, 'rotate', (-0.1, 0.1))
            self.translate_lim = getattr(aug_cfg, 'translate', (-3.0, 3.0))
            self.scale_lim = getattr(aug_cfg, 'scale', (0.98, 1.02))
            self.flip_prob = getattr(aug_cfg, 'flip_prob', 0.5)
            self.rotate90_prob = getattr(aug_cfg, 'rotate90_prob', 0.3)
            self.gaussian_noise_prob = getattr(aug_cfg, 'gaussian_noise_prob', 0.1)
            self.rician_noise_prob = getattr(aug_cfg, 'rician_noise_prob', 0.0)
            self.gamma_prob = getattr(aug_cfg, 'gamma_prob', 0.2)
        else:
            self.geom_prob = 0.3
            self.rotate_lim = (-0.1, 0.1)
            self.translate_lim = (-3.0, 3.0)
            self.scale_lim = (0.98, 1.02)
            self.flip_prob = 0.5
            self.rotate90_prob = 0.3
            self.gaussian_noise_prob = 0.1
            self.rician_noise_prob = 0.0
            self.gamma_prob = 0.2
        
        self.flipper_0 = Flip(spatial_axis=0)
        self.flipper_1 = Flip(spatial_axis=1)
        self.current_rotator = None

        self.do_flip_0 = False
        self.do_flip_1 = False
        
        post = []
        if self.gaussian_noise_prob > 0:
            post.append(RandGaussianNoise(prob=self.gaussian_noise_prob, std=0.005))
        if self.rician_noise_prob > 0:
            post.append(RandRicianNoise(prob=self.rician_noise_prob))
        if self.gamma_prob > 0:
            post.append(RandAdjustContrast(prob=self.gamma_prob, gamma=(0.95, 1.05)))
        post.append(ToTensor())
        self.post = Compose(post)

    def randomize(self):
        if not self.enable_geom:
            self.affine_t = None
            self.do_flip_0 = False
            self.do_flip_1 = False
            self.current_rotator = None
            return

        do_geom = self.R.rand() < self.geom_prob
        if do_geom:
            rx = self.R.uniform(self.rotate_lim[0], self.rotate_lim[1])
            ry = self.R.uniform(self.rotate_lim[0], self.rotate_lim[1])
            rz = self.R.uniform(self.rotate_lim[0], self.rotate_lim[1])
            tx = self.R.uniform(self.translate_lim[0], self.translate_lim[1])
            ty = self.R.uniform(self.translate_lim[0], self.translate_lim[1])
            tz = self.R.uniform(-2.0, 2.0)
            s = self.R.uniform(self.scale_lim[0], self.scale_lim[1])
            self.affine_t = Affine(
                rotate_params=(rx, ry, rz),
                translate_params=(tx, ty, tz),
                scale_params=(s, s, s),
                mode='bilinear',
                padding_mode='border'
            )
        else:
            self.affine_t = None
            
        self.do_flip_0 = self.R.rand() < self.flip_prob
        self.do_flip_1 = self.R.rand() < self.flip_prob
        
        do_rotate90 = self.R.rand() < self.rotate90_prob
        if do_rotate90:
            k = self.R.randint(1, 4)
            self.current_rotator = Rotate90(k=k, spatial_axes=(0, 1))
        else:
            self.current_rotator = None

    def __call__(self, image_paths):
        out_imgs = []
        out_affines = []
        for p in image_paths:
            data = self.loader(p)
            if isinstance(data, (tuple, list)):
                img, meta = data
            else:
                img = data
                meta = getattr(data, 'meta', {})
            
            affine = meta.get('affine', np.eye(4))
            if isinstance(affine, torch.Tensor):
                affine = affine.numpy()
            orig_shape = img.shape
            img = self.add_ch(img)
            if self.resize is not None:
                img = self.resize(img)
                s = np.array(orig_shape) / np.array(self.target_size)
                if len(orig_shape) == 3:
                    scale_mat = np.diag([s[0], s[1], s[2], 1.0])
                    new_affine = affine @ scale_mat
                else:
                    new_affine = affine
            else:
                new_affine = affine

            img = self.scale(img)
            img = self.my_trans(img)

            if self.affine_t is not None:
                img = self.affine_t(img)
                if isinstance(img, (list, tuple)):
                    img = img[0]
            
            if self.do_flip_0:
                img = self.flipper_0(img)
            if self.do_flip_1:
                img = self.flipper_1(img)
            if self.current_rotator is not None:
                img = self.current_rotator(img)
                
            img = self.post(img)
            out_imgs.append(img)
            out_affines.append(torch.from_numpy(new_affine).float())
            
        return out_imgs, out_affines

class PatientConsistentTransform:
    def __init__(self, is_train=True, aug_cfg=None, target_size=(128, 128, 16), enable_resize=True, enable_geom=True):
        self.is_train = is_train
        if is_train:
            self.shared_transform = SharedTransform(aug_cfg, target_size=target_size, enable_resize=enable_resize, enable_geom=enable_geom)
        else:
            class NoAug:
                geom_prob = 0.0
                flip_prob = 0.0
                rotate90_prob = 0.0
                gaussian_noise_prob = 0.0
                rician_noise_prob = 0.0
                gamma_prob = 0.0
            self.shared_transform = SharedTransform(NoAug, target_size=target_size, enable_resize=enable_resize, enable_geom=False)
    
    def __call__(self, image_paths):
        return self.shared_transform(image_paths)

class ImageDataset(Dataset):
    def __init__(self, dce_pre, dwi_pre, t2_pre, dce_post, dwi_post, t2_post, labels, transforms):
        self.dce_pre = dce_pre
        self.dwi_pre = dwi_pre
        self.t2_pre = t2_pre
        self.dce_post = dce_post
        self.dwi_post = dwi_post
        self.t2_post = t2_post
        self.labels = labels
        self.transforms = transforms
        self.is_patient_consistent = isinstance(transforms, PatientConsistentTransform)

    def __len__(self):
        return len(self.dce_pre)

    def __getitem__(self, index):
        dce_pre_paths = self.dce_pre[index].split(';')
        dwi_pre_paths = self.dwi_pre[index].split(';')
        t2_pre_paths = self.t2_pre[index].split(';')
        
        dce_post_paths = self.dce_post[index].split(';')
        dwi_post_paths = self.dwi_post[index].split(';')
        t2_post_paths = self.t2_post[index].split(';')
        
        if self.is_patient_consistent:
            if hasattr(self.transforms, 'shared_transform'):
                self.transforms.shared_transform.randomize()
            dce_pre_imgs, dce_pre_affs = self.transforms(dce_pre_paths)
            dwi_pre_imgs, dwi_pre_affs = self.transforms(dwi_pre_paths)
            t2_pre_imgs, t2_pre_affs = self.transforms(t2_pre_paths)
            dce_post_imgs, dce_post_affs = self.transforms(dce_post_paths)
            dwi_post_imgs, dwi_post_affs = self.transforms(dwi_post_paths)
            t2_post_imgs, t2_post_affs = self.transforms(t2_post_paths)
        else:
            raise NotImplementedError("Only PatientConsistentTransform is supported with Affine loading")
        
        dce_pre_tensor = torch.cat(dce_pre_imgs, dim=0)
        dwi_pre_tensor = torch.cat(dwi_pre_imgs, dim=0)
        t2_pre_tensor = torch.cat(t2_pre_imgs, dim=0)
        
        dce_post_tensor = torch.cat(dce_post_imgs, dim=0)
        dwi_post_tensor = torch.cat(dwi_post_imgs, dim=0)
        t2_post_tensor = torch.cat(t2_post_imgs, dim=0)

        def _pad_upper_to(x, target_hwd):
            th, tw, td = int(target_hwd[0]), int(target_hwd[1]), int(target_hwd[2])
            h, w, d = x.shape[-3], x.shape[-2], x.shape[-1]
            ph = max(0, th - int(h))
            pw = max(0, tw - int(w))
            pd = max(0, td - int(d))
            if ph == 0 and pw == 0 and pd == 0:
                return x
            return torch.nn.functional.pad(x, (0, pd, 0, pw, 0, ph), mode="constant", value=0.0)

        dce_hwd = (
            max(int(dce_pre_tensor.shape[1]), int(dce_post_tensor.shape[1])),
            max(int(dce_pre_tensor.shape[2]), int(dce_post_tensor.shape[2])),
            max(int(dce_pre_tensor.shape[3]), int(dce_post_tensor.shape[3])),
        )
        dwi_hwd = (
            max(int(dwi_pre_tensor.shape[1]), int(dwi_post_tensor.shape[1])),
            max(int(dwi_pre_tensor.shape[2]), int(dwi_post_tensor.shape[2])),
            max(int(dwi_pre_tensor.shape[3]), int(dwi_post_tensor.shape[3])),
        )
        t2_hwd = (
            max(int(t2_pre_tensor.shape[1]), int(t2_post_tensor.shape[1])),
            max(int(t2_pre_tensor.shape[2]), int(t2_post_tensor.shape[2])),
            max(int(t2_pre_tensor.shape[3]), int(t2_post_tensor.shape[3])),
        )
        dce_pre_tensor = _pad_upper_to(dce_pre_tensor, dce_hwd)
        dce_post_tensor = _pad_upper_to(dce_post_tensor, dce_hwd)
        dwi_pre_tensor = _pad_upper_to(dwi_pre_tensor, dwi_hwd)
        dwi_post_tensor = _pad_upper_to(dwi_post_tensor, dwi_hwd)
        t2_pre_tensor = _pad_upper_to(t2_pre_tensor, t2_hwd)
        t2_post_tensor = _pad_upper_to(t2_post_tensor, t2_hwd)

        dce_pre_aff = dce_pre_affs[0]
        dwi_pre_aff = dwi_pre_affs[0]
        t2_pre_aff = t2_pre_affs[0]
        dce_post_aff = dce_post_affs[0]
        dwi_post_aff = dwi_post_affs[0]
        t2_post_aff = t2_post_affs[0]

        eps = 1e-6
        bval = 800.0
        def _adc_from_pair(b0_b800):
            s0 = torch.clamp(b0_b800[0], min=eps)
            s800 = torch.clamp(b0_b800[1], min=eps)
            adc = -torch.log(s800 / s0) / bval
            return adc.unsqueeze(0)
        adc_pre = _adc_from_pair(dwi_pre_tensor)
        adc_post = _adc_from_pair(dwi_post_tensor)
        dwi_pre_tensor = torch.cat([dwi_pre_tensor, adc_pre], dim=0)   # [3,H,W,D]
        dwi_post_tensor = torch.cat([dwi_post_tensor, adc_post], dim=0) # [3,H,W,D]
        
        dce_out = torch.stack([dce_pre_tensor, dce_post_tensor], dim=0)
        dwi_out = torch.stack([dwi_pre_tensor, dwi_post_tensor], dim=0)
        t2_out = torch.stack([t2_pre_tensor, t2_post_tensor], dim=0)
        
        dce_aff = torch.stack([dce_pre_aff, dce_post_aff], dim=0)
        dwi_aff = torch.stack([dwi_pre_aff, dwi_post_aff], dim=0)
        t2_aff = torch.stack([t2_pre_aff, t2_post_aff], dim=0)
        
        return dce_out, dwi_out, t2_out, self.labels[index], dce_aff, dwi_aff, t2_aff

def train_transform(cfg=None):
    aug_cfg = getattr(cfg, 'aug', None) if cfg is not None else None
    arch = getattr(getattr(cfg, 'model', None), 'arch', None) if cfg is not None else None
    enable_resize = True
    enable_geom = True
    if arch == 'calm':
        enable_resize = False
        enable_geom = False
    target_size = getattr(getattr(cfg, 'dataset', None), 'target_size', (128, 128, 16)) if cfg is not None else (128, 128, 16)
    return PatientConsistentTransform(is_train=True, aug_cfg=aug_cfg, target_size=target_size, enable_resize=enable_resize, enable_geom=enable_geom)

def val_transform(cfg=None):
    arch = getattr(getattr(cfg, 'model', None), 'arch', None) if cfg is not None else None
    enable_resize = True
    if arch == 'calm':
        enable_resize = False
    target_size = getattr(getattr(cfg, 'dataset', None), 'target_size', (128, 128, 16)) if cfg is not None else (128, 128, 16)
    return PatientConsistentTransform(is_train=False, aug_cfg=None, target_size=target_size, enable_resize=enable_resize, enable_geom=False)

def getDataLoader(cfg, fold):

    data_df = pd.read_csv(cfg.dataset.data_csv)
    labels = data_df['label'].to_numpy()

    skf = StratifiedKFold(n_splits=cfg.dataset.k_fold, shuffle=True, random_state=cfg.param.seed)
    train_indices, val_indices = list(skf.split(data_df, labels))[fold]
    
    train_df = data_df.iloc[train_indices]
    val_df = data_df.iloc[val_indices]
    
    def create_dataset(df, transforms):
        return ImageDataset(
            dce_pre=df['dce_pre_paths'].to_list(),
            dwi_pre=df['dwi_pre_paths'].to_list(),
            t2_pre=df['t2_pre_paths'].to_list(),
            dce_post=df['dce_post_paths'].to_list(),
            dwi_post=df['dwi_post_paths'].to_list(),
            t2_post=df['t2_post_paths'].to_list(),
            labels=df['label'].to_list(),
            transforms=transforms,
        )

    train_ds = create_dataset(train_df, train_transform(cfg))
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.param.batchsize,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=cfg.dataset.get('drop_last', True)
    )
    
    val_ds = create_dataset(val_df, val_transform(cfg))
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.param.batchsize,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader
