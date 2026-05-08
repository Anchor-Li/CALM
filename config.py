import monai
import numpy as np
from easydict import EasyDict as edict
from dataset.dataloader import train_transform, val_transform

config = edict()
__C = config

# Basic Parameter
__C.param = {}
__C.param.epochs = 100
__C.param.batchsize = 4
__C.param.device = '/'
__C.param.device_ids = [0,1]
__C.param.cls = 2
__C.param.seed = 1
__C.param.saved_model_dir = '/'
monai.utils.set_determinism(seed=config.param.seed)

# Datasets
__C.dataset = {}
__C.dataset.data_csv = '/path/to/your/data_list.csv'
__C.dataset.num_workers = 8
__C.dataset.train_as_val = False
__C.dataset.val_interval = 1
__C.dataset.k_fold = 5
__C.dataset.train_folds = 5 # Number of folds to train (<= k_fold)
__C.dataset.drop_last = True

# Augmentation
__C.aug = {}
__C.aug.geom_prob = 0.3
__C.aug.rotate = (-0.1, 0.1)
__C.aug.translate = (-3.0, 3.0)
__C.aug.scale = (0.98, 1.02)
__C.aug.flip_prob = 0.5
__C.aug.rotate90_prob = 0.3
__C.aug.gaussian_noise_prob = 0.1
__C.aug.rician_noise_prob = 0.1
__C.aug.gamma_prob = 0.2

# loss & opt
__C.loss={}
__C.loss.loss_type = 'weight_cross_entropy'
__C.loss.lr = 3e-4  
__C.loss.weight_decay = 1e-5 
__C.loss.n = np.random.randn(1)
__C.loss.label_smoothing = 0.05
__C.loss.use_class_weight = True
__C.loss.focal_gamma = 2.0

__C.loss.tmcl_enable = True
__C.loss.tmcl_alpha = 0.5
__C.loss.tmcl_proj_dim = 128
__C.loss.tmcl_temp = 0.1
__C.loss.tmcl_margin = 0.6
__C.loss.tmcl_dropout = 0.2
__C.loss.tmcl_intra_w = 1.0
__C.loss.tmcl_inter_w = 1.0

# Transforms
__C.transforms = {}
__C.transforms.train = train_transform()
__C.transforms.val = val_transform()  

# Early stopping
__C.early_stopping = {}
__C.early_stopping.patience = 30 

# Training control
__C.train = {}
__C.train.monitor = 'val_loss'
__C.train.scheduler_mode = 'min'
__C.train.scheduler_patience = 10
__C.train.scheduler_factor = 0.5
__C.train.amp = True
__C.train.accum_steps = 4
__C.train.warmup_epochs = 5
# __C.train.save_preds_csv = True

# Model config
__C.model = {}
__C.model.norm = 'instance' 
__C.model.arch = 'calm' 

__C.model.geo = {}
__C.model.geo.dce_base = 12
__C.model.geo.dwi_base = 12
__C.model.geo.t2_base = 12
__C.model.geo.embed_dim = 256
__C.model.geo.dist_mm = 5.0
__C.model.geo.q_chunk = 256
__C.model.geo.q_topk = 2048
__C.model.geo.sag_pool = (2, 2, 1)
__C.model.geo.use_checkpoint = False
__C.model.geo.ablate_pgm = False
__C.model.geo.ablate_pcia = False

# Inference config
__C.infer = {}
__C.infer.threshold = 0.5

if __name__ == '__main__':
    print(config)
