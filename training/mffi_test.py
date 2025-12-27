import gc
import time
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as tr
from sklearn import metrics
import random
import numpy as np
import cv2
import os
import pandas as pd
from torchvision.transforms import InterpolationMode
import sys
import warnings
import albumentations as alb
warnings.filterwarnings('ignore')

seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
# CUDA_LAUNCH_BLOCKING = 1

def compute_F1(y, pred, n_class=1, t=0.5):
    ## compute one score
    if n_class == 1:
        pred[pred >= t] = 1
        pred[pred < t] = 0
        f1 = metrics.f1_score(y, pred)
        return f1
    else:
        ## compute two-class
        index = torch.argmax(pred, dim=1)
        # index[index!=0]=1
        f1 = metrics.f1_score(y, index)
    
    return f1


def compute_AP(y, pred):
    ap = metrics.average_precision_score(y, pred)
    return ap

 
def compute_ACC(y, pred, n_class=1, t=0.5):
    ## compute one score
    if n_class == 1:
        pred[pred >= t] = 1
        pred[pred < t] = 0
        acc = metrics.accuracy_score(y, pred)

    ## compute two-class
    elif n_class == 2:
        index = torch.argmax(pred, dim=1)
        # index[index!=0]=1
        acc = metrics.accuracy_score(y, index)
        # acc = metrics.f1_score(y, index)
    
    return acc


def compute_AUC(y, pred, n_class=1):
    ## compute one score
    if n_class == 1:
        auc = metrics.roc_auc_score(y, pred)

    ## compute two-class
    elif n_class == 2:
        # pos = pred[:, 1]
        auc = metrics.roc_auc_score(y, pred)
    return auc


def Tensor2cv(img_tensor):
    img_tensor = img_tensor.permute(1, 2, 0).cpu()
    img_numpy = img_tensor.numpy() * 255
    img_numpy = np.uint8(img_numpy)
    img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
    return img_numpy

def crop_face(img,landmark=None,bbox=None,margin=False,crop_by_bbox=True,abs_coord=False,only_img=False,phase='train'):
    assert phase in ['train','val','test']

    #crop face------------------------------------------
    H,W=len(img),len(img[0])

    assert landmark is not None or bbox is not None

    H,W=len(img),len(img[0])
    
    if crop_by_bbox:
        x0,y0=bbox[0]
        x1,y1=bbox[1]
        w=x1-x0
        h=y1-y0
        w0_margin=w/4#0#np.random.rand()*(w/8)
        w1_margin=w/4
        h0_margin=h/4#0#np.random.rand()*(h/5)
        h1_margin=h/4
    else:
        x0,y0=landmark[:68,0].min(),landmark[:68,1].min()
        x1,y1=landmark[:68,0].max(),landmark[:68,1].max()
        w=x1-x0
        h=y1-y0
        w0_margin=w/32#w/8#0#np.random.rand()*(w/8)
        w1_margin=w/32#w/8
        h0_margin=h/4.5#h/2#0#np.random.rand()*(h/5)
        h1_margin=h/16# h/5

    if margin:
        w0_margin*=4
        w1_margin*=4
        h0_margin*=2
        h1_margin*=2
    elif phase=='train':
        # print('dynamic cropping')
        w0_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
        w1_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
        h0_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
        h1_margin*=(np.random.rand()*0.6+0.2)#np.random.rand()
        
        # w0_margin*=(np.random.rand()*.6+3.7)#np.random.rand()
        # w1_margin*=(np.random.rand()*.6+3.7)#np.random.rand()
        # h0_margin*=(np.random.rand()*.6+1.7)#np.random.rand()
        # h1_margin*=(np.random.rand()*.6+1.7)#np.random.rand()
     
    else:
        w0_margin*=0.5
        w1_margin*=0.5
        h0_margin*=0.5
        h1_margin*=0.5
            
    y0_new=max(0,int(y0-h0_margin))
    y1_new=min(H,int(y1+h1_margin)+1)
    x0_new=max(0,int(x0-w0_margin))
    x1_new=min(W,int(x1+w1_margin)+1)
    
    img_cropped=img[y0_new:y1_new,x0_new:x1_new]
    if landmark is not None:
        landmark_cropped=np.zeros_like(landmark)
        for i,(p,q) in enumerate(landmark):
            landmark_cropped[i]=[p-x0_new,q-y0_new]
    else:
        landmark_cropped=None
    if bbox is not None:
        bbox_cropped=np.zeros_like(bbox)
        for i,(p,q) in enumerate(bbox):
            bbox_cropped[i]=[p-x0_new,q-y0_new]
    else:
        bbox_cropped=None

    if only_img:
        return img_cropped
    if abs_coord:
        return img_cropped,landmark_cropped,bbox_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1),y0_new,y1_new,x0_new,x1_new
    else:
        return img_cropped,landmark_cropped,bbox_cropped,(y0-y0_new,x0-x0_new,y1_new-y1,x1_new-x1)

from glob import glob
import json
def init_ff_real(phase, dataset_paths, n_frames=8):
    landmark_path = '/home/liu/fcb1/dataset/FFPP_16/landmarks/'
    image_list = []
    landmark_list=[]
    folder_list = sorted(glob(os.path.join(dataset_paths,'*')))
    landmark_folder_list = sorted(glob(landmark_path+'*'))
    filelist = []
    list_dict = json.load(open(f'/home/liu/fcb1/dataset/FFplus/{phase}.json','r'))
    for i in list_dict:
        filelist+=i
    folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]
    landmark_folder_list = [i for i in landmark_folder_list if os.path.basename(i)[:3] in filelist]

    for i in range(len(folder_list)):
        images_temp=sorted(glob(folder_list[i]+'/*.png'))
        landmarks_temp=sorted(glob(landmark_folder_list[i]+'/*.npy'))
        if n_frames<len(images_temp):
            images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
            landmarks_temp=[landmarks_temp[round(i)] for i in np.linspace(0,len(landmarks_temp)-1,n_frames)]     
        image_list+=images_temp
        landmark_list+=landmarks_temp
        
        
    return image_list,landmark_list

class Phase1Trainset(Dataset):
    def __init__(self, root_dir, img_size=224):
        self.root_dir = root_dir
        self.phase = "train"
        self.data = pd.read_csv(os.path.join(self.root_dir, 'phase1', 'trainset_label.txt'))
        self.real_list = []
        self.fake_list = []
        for img_name, lab in zip(self.data['img_name'], self.data['target']):
            if lab == 0:
                self.real_list.append(img_name)
            else:
                self.fake_list.append(img_name)
                
        # self.original_root = f'/home/liu/sdb/FFPP/original_sequences/youtube/c23/images/'
        # original_list, _ = init_ff_real(phase='train',
        #                                             dataset_paths=self.original_root,
        #                                             n_frames=8)
        # self.landmark_root = "/home/liu/fcb1/dataset/FFPP_16/landmarks/"
        # self.real_list += original_list
        
        print('Real samples : {}'.format(len(self.real_list)))
        print('Fake samples : {}'.format(len(self.fake_list)))

        # print(self.data)
        # self.norm = tr.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.img_size = img_size
        self.augmentation = self.get_transforms()
        self.rid = 0
        
    def __len__(self):
        return len(self.fake_list)

    def get_transforms(self):
        return alb.Compose([
            # alb.RandomResizedCrop(224, 224, scale=(0.97, 1), ratio=(0.97, 1),interpolation=cv2.INTER_CUBIC, p=1.), 
            alb.HorizontalFlip(p=0.5),
            alb.RGBShift((-20,20),(-20,20),(-20,20), p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
            alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
            alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.2),
            alb.GaussianBlur(p=0.05),
            alb.GaussNoise(p=0.03),
            # alb.OneOf([
            #         alb.GaussNoise(p=1.),
            #     alb.ISONoise(p=1.),
            #     alb.MultiplicativeNoise(p=1.)
            #     ],p=0.03
            # ),
            alb.RandomGridShuffle((self.img_size//16, self.img_size//16), p=0.15),
        ], 
        additional_targets={f'image1': 'image'},
        p=1.)
    
    def __getitem__(self, idx):
        fimg_path = os.path.join(self.root_dir, 'phase1', self.phase + 'set', self.fake_list[idx])
        real_path = self.real_list[self.rid]
        if 'original_sequences' not in real_path:
            rimg_path = os.path.join(self.root_dir, 'phase1', self.phase + 'set', self.real_list[self.rid])
        self.rid += 1
        if self.rid >= len(self.real_list): self.rid = 0
        
        fimg = cv2.imread(fimg_path)
        fimg = cv2.cvtColor(fimg, cv2.COLOR_BGR2RGB)
        rimg = cv2.imread(rimg_path)
        rimg = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
        # if 'original_sequences' in real_path:
        #     real_id = real_path.split('/')[-2]
        #     frame_id = real_path.split('/')[-1].split('.')[0] + '.npy'
        #     lamk_path = self.landmark_root + real_id + '/' + frame_id
        #     landmark=np.load(lamk_path)[0]
        #     rimg, _, _, _, y0_new,y1_new,x0_new,x1_new=crop_face(rimg,landmark,bbox=None,margin=True,crop_by_bbox=False,abs_coord=True,phase=self.phase)

        inter_flag = cv2.INTER_CUBIC
        fimg = cv2.resize(fimg, dsize=(self.img_size, self.img_size), interpolation=inter_flag)
        rimg = cv2.resize(rimg, dsize=(self.img_size, self.img_size), interpolation=inter_flag)
        
        transformed = self.augmentation(image=fimg.astype('uint8'),\
                image1=rimg.astype('uint8'))
        rimg=transformed['image1'].transpose((2, 0, 1)) / 255.
        fimg=transformed['image'].transpose((2, 0, 1)) / 255. 
        
        # fimg = fimg.transpose((2, 0, 1)) / 255.
        # rimg = rimg.transpose((2, 0, 1)) / 255.
        
        return fimg, rimg, 
    
    def collate_fn(self, batch):
        img_f, img_r = zip(*batch)
        data={}
        data['img']=torch.cat([torch.tensor(img_r).float(),torch.tensor(img_f).float()],0)
        data['label']=torch.tensor([0]*len(img_r)+[1]*len(img_f))
        return data

    def worker_init_fn(self,worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)
        
        
        

class phase1Dataset(Dataset):
    def __init__(self, root_dir, img_size=224, phase='val'):
        self.root_dir = root_dir
        self.phase = phase
        
        if self.phase == 'train':
            self.data = pd.read_csv(os.path.join(self.root_dir, 'phase1', 'trainset_label.txt'))
        elif self.phase == 'val':
            self.data = pd.read_csv(os.path.join(self.root_dir, 'phase1', 'valset_label.txt'))
        # print(self.data)
        self.norm = tr.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.img_size = img_size
        
    def __len__(self):
        return len(self.data)
    
    def center_crop(self, img, crop_size):
        h, w = img.shape[:2]
        crop_h, crop_w = crop_size
        
        # 计算裁剪的起始和结束位置
        start_h = max(h // 2 - (crop_h // 2), 0)
        start_w = max(w // 2 - (crop_w // 2), 0)
        end_h = start_h + crop_h
        end_w = start_w + crop_w
        
        # 进行裁剪
        cropped_img = img[start_h:end_h, start_w:end_w]
        
        return cropped_img
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, 'phase1', self.phase + 'set', self.data.iloc[idx, 0])

        label = torch.tensor(int(self.data.iloc[idx, 1]))
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = self.center_crop(img, (384, 384))
        
        inter_flag = cv2.INTER_CUBIC
        img = cv2.resize(img, dsize=(self.img_size, self.img_size), interpolation=inter_flag)
        img = torch.tensor(img.transpose((2, 0, 1)) / 255., dtype=torch.float32)
        img = self.norm(img)
        
        img_name = self.data.iloc[idx, 0]
        
        return img, label, img_name


if __name__ == "__main__":
    # train_dataset=Phase1Trainset(root_dir='/home/liu/sdb')
    # train_loader=DataLoader(train_dataset,
    #                         batch_size=32,
    #                         num_workers=8,
    #                         pin_memory=True,
    #                         shuffle=True,
    #                         drop_last=True,
    #                         collate_fn=train_dataset.collate_fn,
    #                         worker_init_fn=train_dataset.worker_init_fn)
    val_dataset = phase1Dataset(root_dir='/home/liu/sdb',phase='val')
    val_loader = DataLoader(val_dataset,
                             batch_size=64,
                             num_workers=8,
                             pin_memory=True,
                             shuffle=False)
    print("start")

    from CLIP import clip
    from CLIP_Model import CLIP_Detector
    clip_model, preprocess = clip.load("ViT-B/16", \
        device=torch.device("cpu"), download_root="/home/liu/fcb1/decouple/clip_model")#ViT-B/16
    model = CLIP_Detector(clip_model).cuda()
    wp = '/home/liu/fcb1/CPL/trained_weights/FAPL.pth'
    model.init_text_embed_with_pretrainweight(wp)
    print('parameters inherited from ', wp)
    model.eval()

    all_label = None
    all_probs = None
    
    all_img_name = []
    # print(len(train_loader))
    print(len(val_loader))
    st = time.time()
    
    prediction = {'img_name': [],
        'y_pred': []}
    with torch.no_grad():
        # for step_id, datas in enumerate(train_loader):
        for step_id, datas in enumerate(val_loader):
            img = datas[0].cuda()
            lab = datas[1]
            img_name = datas[2]
            
            probs = model.test_forward(img)
            probs = torch.sigmoid(probs)
            
            all_probs = probs if all_probs is None else torch.cat((all_probs, probs), dim=0)
            all_label = lab if all_label is None else torch.cat((all_label, lab), dim=0)
            all_img_name += img_name
            
            
            if not (step_id+1) % 100:
                now_percent = int(step_id / len(val_loader) * 100)
                print(f"Test: complete {now_percent} %")
                
    auc = compute_AUC(all_label.cpu().detach(), all_probs.cpu().detach())
    acc = compute_ACC(all_label.cpu().detach(), all_probs.cpu().detach())
    
    for i, imgn in enumerate(all_img_name):
        prediction['img_name'].append(imgn)
        prediction['y_pred'].append(all_probs[i, 0].cpu().detach().numpy())
    df = pd.DataFrame(prediction)
    df.to_csv('/home/liu/fcb1/CPL/output.csv', index=False)
    
    end = time.time()
    print(f'test AUC: {auc: .4f}; ACC: {acc: .4f}')
    print(f'{(end-st)/60.: .4f} min')
    