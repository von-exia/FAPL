import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from test_loader import Get_DataLoader
from FFPP_Dataset import FFPP_Dataset, VideoFFPP_Dataset
import random
import time
from sklearn import metrics
from torch.utils import data
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

seed = 10
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
print('Random seed :', seed)
# for determinstic reproduction
torch.use_deterministic_algorithms(True)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from collections import OrderedDict
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def compute_AUC(y, pred, n_class=1):
    # compute one score
    if n_class == 1:
        auc = metrics.roc_auc_score(y, pred)

    # compute two-class
    elif n_class == 2:
        # pos = pred[:, 1]
        auc = metrics.roc_auc_score(y, pred)
    return auc


def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
                

image_size = 224
train_dataset = FFPP_Dataset(phase='train',image_size=image_size, compress='c23', tamper='all')
batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size = batch_size//2,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=8,
                                            collate_fn=train_dataset.collate_fn,
                                            worker_init_fn=train_dataset.worker_init_fn,
                                            drop_last = True
                                            )


TestSet = Get_DataLoader(dataset_name="VideoCDF", image_size=224, normalize=True)
train_len = len(train_loader)
print('length of TrainSet is ', train_len)

criterion_dict = dict()
criterion_dict['bce'] = nn.BCEWithLogitsLoss().cuda()
criterion_dict['mse'] = nn.MSELoss().cuda()
criterion_dict['L1'] = nn.L1Loss().cuda()
# -----------------Build optimizerr-----------------

device = "cuda" if torch.cuda.is_available() else "cpu"
from CLIP import clip
from CLIP_Model import CLIP_Detector, MultiClassFocalLossWithAlpha, FocalLoss_BCE
clip_model, preprocess = clip.load("ViT-B/16", \
    device=torch.device("cpu"), download_root="/home/liu/fcb1/clip_model")#ViT-B/16

from CLIP import longclip
clip_model, preprocess = longclip.load("/home/liu/fcb1/clip_model/longclip-B.pt",  # longclip-L@336px.pt  longclip-B.pt
                                  device=torch.device("cpu"), download_root="/home/liu/fcb1/clip_model")
sd = torch.load("/home/liu/fcb1/reft/trained_weight/longclip_ViT-B_ft_0_frames1.pth")
clip_model.load_state_dict(sd['model'])
model = CLIP_Detector(clip_model).cuda()

# criterion_dict['focal_ce'] = MultiClassFocalLossWithAlpha(gamma=2)
# criterion_dict['focal_bce'] = FocalLoss_BCE(alpha=0.25, gamma=2) # V2 second-best
# criterion_dict['focal_bce'] = FocalLoss_BCE(alpha=[0.8, 0.2], gamma=2) # V2 best

best_auc = 0.

from adan import Adan
optimizer = Adan(model.parameters(), lr=1e-5, betas=(0.98, 0.92, 0.99), weight_decay=1e-2, eps=1e-8)


Epoch = 300
step = 0
step2 = 0
step_v = 0

device = torch.device('cuda')
gc.collect()
torch.cuda.empty_cache()
print("Let's start training!")

end_decay = 500
half_decay = 40
start_decay = 0

import torchvision.transforms as tr
normalize = tr.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

max_val_auc = 0.
e_cnt = 1
train_start_time = time.time()
for e in range(0, Epoch):
    start = time.time()
    model.train()
    for step_id, data in enumerate(train_loader):
        # start = time.time()
        img = data['img'].to(device, non_blocking=True).float()
        label = data['label'].to(device, non_blocking=True).float() 
        margin_mask = data['margin_mask'].to(device, non_blocking=True).float()
        cls_mask = data['cls_mask'].to(device, non_blocking=True).float()
        cls_mask[batch_size//2:, ...] = margin_mask # best

        soft_label = data['soft_labels'].to(device, non_blocking=True).float()
        # soft_label = data['mlab_list'].to(device, non_blocking=True).float()
        soft_label_df = soft_label[:, 0]
        soft_label_fs = soft_label[:, 1]
        soft_label_ff = soft_label[:, 2]
        soft_label_nt = soft_label[:, 3]

        for i in range(len(img)):
            img[i] = normalize(img[i])
        probs, cls1, cls2, cls3, cls4, fi = model(img)
        
        loss_cls = criterion_dict['bce'](probs, label.reshape(-1, 1))
        # loss_cls = criterion_dict['focal_bce'](probs, label.reshape(-1, 1))
        
        loss_cls1 = criterion_dict['bce'](cls1, soft_label_df.reshape(-1, 1))
        loss_cls2 = criterion_dict['bce'](cls2, soft_label_fs.reshape(-1, 1))
        loss_cls3 = criterion_dict['bce'](cls3, soft_label_ff.reshape(-1, 1))
        loss_cls4 = criterion_dict['bce'](cls4, soft_label_nt.reshape(-1, 1))
        loss_spc = (loss_cls1*1. + loss_cls2*1. + loss_cls3*1. + loss_cls4*1.) / 4.
        
        # loss_cls1 = F.cross_entropy(cls1, soft_label_df.long())
        # loss_cls2 = F.cross_entropy(cls2, soft_label_fs.long())
        # loss_cls3 = F.cross_entropy(cls3, soft_label_ff.long())
        # loss_cls4 = F.cross_entropy(cls4, soft_label_nt.long())
        # loss_cls1 = criterion_dict['focal_ce'](cls1, soft_label_df.long())
        # loss_cls2 = criterion_dict['focal_ce'](cls2, soft_label_fs.long())
        # loss_cls3 = criterion_dict['focal_ce'](cls3, soft_label_ff.long())
        # loss_cls4 = criterion_dict['focal_ce'](cls4, soft_label_nt.long())

        loss_fi = criterion_dict['L1'](fi, cls_mask) # best
        loss =  loss_cls*1. + loss_spc + loss_fi*1.

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        if not (step_id+1) % 30:
            print(f"epoch: {e} / {Epoch},step {step_id+1} / {len(train_loader)}, loss: {loss.detach().cpu().numpy():.4f}")


        if (step_id+1) % 90 == 0:
            model.eval()
            outputs = None
            testargets = None
            print('testing......')
            with torch.no_grad():
                for step_id, datas in enumerate(TestSet):
                    img = datas[0].cuda()
                    targets = datas[1].cuda()
                    output = model.test_forward(img)
                    cls_final = torch.sigmoid(output)
                    
                    n_frames = img.shape[0]
                    targets = targets.expand(n_frames, -1)
                    outputs = cls_final if outputs is None else torch.cat(
                        (outputs, cls_final), dim=0)
                    testargets = targets if testargets is None else torch.cat(
                        (testargets, targets), dim=0)
            cur_auc = compute_AUC(testargets.cpu().detach(), outputs.cpu().detach())
            print(f'Frame-level CDF test AUC:{cur_auc:.4f}')
            if best_auc < cur_auc:
                # best_auc = cur_auc
                torch.save(model.state_dict(
                ), f'/home/liu/fcb1/CPL/trained_weights/CPL_Epoch{e_cnt}_cdf{cur_auc:.4f}.pth')
 
            step_v += 1

            end = time.time()
            print(f"epoch: {e_cnt} end ; cost time: {(end - start)/60.:.4f} min")
            start = time.time()
            e_cnt += 1
            gc.collect()
            torch.cuda.empty_cache()
            if e_cnt >= end_decay:
                print("training end.")
                train_end_time = time.time()
                training_time = train_end_time - train_start_time
                training_time /= 60. # min
                training_hours = training_time // 60
                training_min = training_time % 60
                print(f"training spent {training_hours} hs; {training_min:.4f} mins...")
                exit(0)
            model.train()
print('train ended !')
