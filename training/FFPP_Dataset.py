import torch
from torchvision import datasets,transforms,utils
from torch.utils.data import Dataset,IterableDataset
import torchvision.transforms as tr
from glob import glob
import os
import numpy as np
import random
import cv2
import json
import io
from PIL import Image
import time
from torchvision.transforms import InterpolationMode
import sys
import warnings
import albumentations as alb
warnings.filterwarnings('ignore')

def IoUfrom2bboxes(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

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
        
        # w0_margin*=(np.random.rand()*4.+0.2)#np.random.rand()
        # w1_margin*=(np.random.rand()*4.+0.2)#np.random.rand()
        # h0_margin*=(np.random.rand()*2.+0.2)#np.random.rand()
        # h1_margin*=(np.random.rand()*2.+0.2)#np.random.rand()
     
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

def init_fff(phase, dataset_paths, DF, NT, FS, FF, n_frames=8):
    landmark_path = '/home/liu/fcb1/dataset/FFPP_16/landmarks/'
    image_list = []
    DFs_list = []
    NTs_list = []
    FSs_list = []
    FFs_list = []
    landmark_list=[]
    folder_list = sorted(glob(os.path.join(dataset_paths,'*')))
    landmark_folder_list = sorted(glob(landmark_path+'*'))
    DF_list = sorted(glob(os.path.join(DF,'*')))
    DF_list = sorted(DF_list, key=lambda x: int(x.split('/')[-1].split('_')[0]))
    NT_list = sorted(glob(os.path.join(NT,'*')))
    NT_list = sorted(NT_list, key=lambda x: int(x.split('/')[-1].split('_')[0]))
    FS_list = sorted(glob(os.path.join(FS,'*')))
    FS_list = sorted(FS_list, key=lambda x: int(x.split('/')[-1].split('_')[0]))
    FF_list = sorted(glob(os.path.join(FF,'*')))
    FF_list = sorted(FF_list, key=lambda x: int(x.split('/')[-1].split('_')[0]))
    filelist = []
    list_dict = json.load(open(f'/home/liu/fcb1/dataset/FFplus/{phase}.json','r'))
    for i in list_dict:
        filelist+=i
    folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]
    landmark_folder_list = [i for i in landmark_folder_list if os.path.basename(i)[:3] in filelist]
    DF_list = [i for i in DF_list if os.path.basename(i)[:7].split('/')[-1].split('_')[0] in filelist]
    NT_list = [i for i in NT_list if os.path.basename(i)[:7].split('/')[-1].split('_')[0] in filelist]
    FS_list = [i for i in FS_list if os.path.basename(i)[:7].split('/')[-1].split('_')[0] in filelist]
    FF_list = [i for i in FF_list if os.path.basename(i)[:7].split('/')[-1].split('_')[0] in filelist]
    for i in range(len(folder_list)):
        images_temp=sorted(glob(folder_list[i]+'/*.png'))
        DF_temp = sorted(glob(DF_list[i]+'/*.png'))
        NT_temp = sorted(glob(NT_list[i]+'/*.png'))
        FS_temp = sorted(glob(FS_list[i]+'/*.png'))
        FF_temp = sorted(glob(FF_list[i]+'/*.png'))
        landmarks_temp=sorted(glob(landmark_folder_list[i]+'/*.npy'))
        if n_frames<len(images_temp):
            images_temp=[images_temp[round(i)] for i in np.linspace(0,len(images_temp)-1,n_frames)]
            DF_temp=[DF_temp[round(i)] for i in np.linspace(0,len(DF_temp)-1,n_frames)]
            NT_temp=[NT_temp[round(i)] for i in np.linspace(0,len(NT_temp)-1,n_frames)]
            FS_temp=[FS_temp[round(i)] for i in np.linspace(0,len(FS_temp)-1,n_frames)]
            FF_temp=[FF_temp[round(i)] for i in np.linspace(0,len(FF_temp)-1,n_frames)]
            landmarks_temp=[landmarks_temp[round(i)] for i in np.linspace(0,len(landmarks_temp)-1,n_frames)]     
        image_list+=images_temp
        DFs_list+=DF_temp
        NTs_list+=NT_temp
        FSs_list+=FS_temp
        FFs_list+=FF_temp
        landmark_list+=landmarks_temp
    return image_list,DFs_list,NTs_list,FSs_list,FFs_list,landmark_list

# def landmark_fff(phase, n_frames=8):
#     print(phase)
#     landmark_path = '/home/liu/fcb1/dataset/FFPP_16/landmarks/'
#     landmark_list=[]
#     folder_list = sorted(glob(landmark_path+'*'))
#     filelist = []
#     list_dict = json.load(open(f'/home/liu/fcb1/dataset/FFplus/{phase}.json','r'))
#     for i in list_dict:
#         filelist+=i
#     folder_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]
#     for i in range(len(folder_list)):
#         landmarks_temp=sorted(glob(folder_list[i]+'/*.npy'))
#         landmark_list+=landmarks_temp
#     return landmark_list

class FFPP_Dataset(Dataset):
    def __init__(self,compress='raw',image_size=256,phase = "train", tamper="all"):# raw、c23、c40
        super().__init__()
        self.original_root = f'/home/liu/sdb/FFPP/original_sequences/youtube/{compress}/images/'
        self.Deepfakes_root = f'/home/liu/sdb/FFPP/manipulated_sequences/Deepfakes/{compress}/images/'
        self.NeuralTextures_root = f'/home/liu/sdb/FFPP/manipulated_sequences/NeuralTextures/{compress}/images/'
        self.FaseSwap_root = f'/home/liu/sdb/FFPP/manipulated_sequences/FaceSwap/{compress}/images/'
        self.Face2Face_root = f'/home/liu/sdb/FFPP/manipulated_sequences/Face2Face/{compress}/images/'
 
        print('start')
        st = time.time()
        # self.original_list = init_fff(phase=phase, dataset_paths=self.original_root)
        # Deepfakes_list = init_fff(phase=phase, dataset_paths=self.Deepfakes_root)
        # NeuralTextures_list = init_fff(phase=phase, dataset_paths=self.NeuralTextures_root)
        # FaseSwap_list = init_fff(phase=phase, dataset_paths=self.FaseSwap_root)
        # Face2Face_list = init_fff(phase=phase, dataset_paths=self.Face2Face_root)
        if phase == 'val':
            n_frames = 4
        else:
            n_frames = 8
        original_list,Deepfakes_list,NeuralTextures_list,FaseSwap_list,Face2Face_list,landmark_list = init_fff(phase=phase,
                                                                                                               dataset_paths=self.original_root,
                                                                                                               DF=self.Deepfakes_root,
                                                                                                               NT=self.NeuralTextures_root,
                                                                                                               FS=self.FaseSwap_root,
                                                                                                               FF=self.Face2Face_root,
                                                                                                               n_frames=n_frames)
        self.real_frame_list = original_list
        self.landmark_list = landmark_list
        if tamper == 'all':
            self.fake_frame_list = Deepfakes_list + NeuralTextures_list + FaseSwap_list + Face2Face_list
            # Deepfakes_list[:len(Deepfakes_list)//4] + NeuralTextures_list[len(NeuralTextures_list)//4:len(NeuralTextures_list)//2] + FaseSwap_list[len(FaseSwap_list)//2:len(FaseSwap_list)*3//4] + Face2Face_list[len(Face2Face_list)*3//4:]
            # self.fake_frame_list = Deepfakes_list[:len(Deepfakes_list)//4] + FaseSwap_list[len(FaseSwap_list)//4:len(FaseSwap_list)//2] + FaseSwap_list[len(FaseSwap_list)//2:len(FaseSwap_list)*3//4] + Face2Face_list[len(Face2Face_list)*3//4:]
        elif tamper == 'DF':
            self.fake_frame_list = Deepfakes_list
        elif tamper == 'NT':
            self.fake_frame_list = NeuralTextures_list
        elif tamper == 'FS':
            self.fake_frame_list = FaseSwap_list
        elif tamper == 'FF':
            self.fake_frame_list = Face2Face_list
        elif tamper == 'DF+FS':
            self.fake_frame_list = FaseSwap_list + Deepfakes_list
        elif tamper == 'FS+FF+NT':
            self.fake_frame_list = FaseSwap_list + Face2Face_list + NeuralTextures_list
            
            
        ed = time.time()
        print(f'load... {ed -st:.2f} s')
        # if phase == 'train':
        #     self.fake_frame_list = [self.fake_frame_list[i] for i in range(0, len(self.fake_frame_list), 2)]
        # elif phase == 'val':
        #     self.fake_frame_list = [self.fake_frame_list[i] for i in range(0, len(self.fake_frame_list), 8)]
        print('Real samples : {}'.format(len(self.real_frame_list)))
        print('Fake samples : {}'.format(len(self.fake_frame_list)))

        
        self.idx_real = 0
        self.max_real = len(self.real_frame_list)
        self.img_size = image_size
        
        self.landmark_root = "/home/liu/fcb1/dataset/FFPP_16/landmarks/"
        self.real_root = self.original_root
        self.augmentation = self.get_transforms()
        self.phase = phase
        self.tamper = tamper
        
        
    def __len__(self):
        return len(self.fake_frame_list)
    
    # # v1 best    
    def get_transforms(self):
        return alb.Compose([
            alb.HorizontalFlip(p=0.5),
            alb.RGBShift((-20,20),(-20,20),(-20,20),p=0.3),
            alb.HueSaturationValue(hue_shift_limit=(-0.3,0.3), sat_shift_limit=(-0.3,0.3), val_shift_limit=(-0.3,0.3), p=0.3),
            alb.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3,0.3), p=0.3),
            alb.ImageCompression(quality_lower=40,quality_upper=100,p=0.2),
            alb.GaussianBlur(p=0.05),
            alb.GaussNoise(p=0.03),
            alb.RandomGridShuffle((self.img_size//16, self.img_size//16), p=0.15),
        ], 
        additional_targets={f'image1': 'image'},
        p=1.)
    
        
    def get_patch_mask(self, real, fake):   # for prediction of num of modified pixels in a patch
        ph = pw = 16
        pixel_num_of_patch = ph * pw
        mask = np.abs(real - fake)
        mask = np.sum(mask, axis=0)
        H, W = mask.shape
        mask[mask!=0] = 1
        mask = mask.reshape(H//ph, ph, W//pw, pw)
        mask = mask.transpose((0, 2, 1, 3)) # H, W, patch_h, patch_w
        mask = mask.reshape(H//ph, W//pw, ph*pw) # H, W, patch_h*patch_w
        patch_score = np.sum(mask, axis=2) / pixel_num_of_patch
        patch_score = patch_score.reshape((H//ph)*(W//pw))
        
        return patch_score
    
    # V2
    def get_patch_margin_and_cls_mask(self, residual):
        # pn = 24 # for ViT
        pn = 14 # for CLIP/b16
        # pn = 7 # for CLIP/b32
        ph = pw = self.img_size // pn
        pixel_num_of_patch = ph * pw
        C, H, W = residual.shape
        # mask = np.sum(residual, axis=0) # best
        mask = np.sum(np.abs(residual), axis=0)
        mask[mask!=0] = 1
        mask = mask.reshape(H//ph, ph, W//pw, pw)
        mask = mask.transpose((0, 2, 1, 3)) # H, W, patch_h, patch_w
        mask = mask.reshape(H//ph, W//pw, ph*pw) # H, W, patch_h*patch_w
        
        margin_mask = np.sum(mask, axis=2) / pixel_num_of_patch
        margin_mask = margin_mask[np.newaxis, ...]
        cls_mask = margin_mask.copy()
        cls_mask[cls_mask!=0] = 1
        
        
        #------------------stage 1-------------------
        residual_gt1 = residual.copy()
        # mask_gt1 = residual.copy()
        # mask_gt1[mask_gt1<0] = -1.
        # mask_gt1[mask_gt1>0] = 1.
        residual_gt1 = np.abs(residual_gt1)
        # margin = 16. * 1. / 255.
        margin = 16. * 2. / 255.   # best
        # margin = 16. * 3. / 255.
        residual_gt1[residual_gt1 < margin] = 0.
        # residual_gt1 *= mask_gt1
        
        mask = np.sum(residual_gt1, axis=0) # best
        # mask = np.sum(np.abs(residual_gt1), axis=0)
        mask[mask!=0] = 1
        mask = mask.reshape(H//ph, ph, W//pw, pw)
        mask = mask.transpose((0, 2, 1, 3)) # H, W, patch_h, patch_w
        mask = mask.reshape(H//ph, W//pw, ph*pw) # H, W, patch_h*patch_w
        
        margin_mask2 = np.sum(mask, axis=2) / pixel_num_of_patch
        margin_mask2 = margin_mask2[np.newaxis, ...]
        
        margin_mask[margin_mask>1] = 1.
        margin_mask2[margin_mask2>1] = 1.

        return margin_mask, cls_mask, margin_mask2 # , fsa_stage1_label, fsa_stage2_label
        
    
    def __getitem__(self, idx):
        flag = True
        ms_blend = False
        # if np.random.rand() < 0.5: # V2 best
        if np.random.rand() < 0.1: # V1 best
            ms_blend = True
        while flag:
            try:
                img_path = self.fake_frame_list[idx]
                real_id = img_path.split('/')[-2].split('_')[0]
                frame_id = img_path.split('/')[-1].split('.')[0] + '.npy'
                lamk_path = self.landmark_root + real_id + '/' + frame_id
                png_path = img_path.split('images')[-1][1:]
                forgery_type = img_path.split('/')[-5]
                fake_img = cv2.imread(img_path)
                fake_img = cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)
                
                real_img_path = self.real_root + real_id + '/' + frame_id.replace(".npy", ".png")
                img = cv2.imread(real_img_path)
                real_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                if ms_blend:
                    df_path = self.Deepfakes_root + png_path
                    fs_path = self.FaseSwap_root + png_path 
                    ff_path = self.Face2Face_root + png_path 
                    nt_path = self.NeuralTextures_root + png_path 
                    
                    df_img = cv2.imread(df_path)
                    df_img = cv2.cvtColor(df_img, cv2.COLOR_BGR2RGB)
                    fs_img = cv2.imread(fs_path)
                    fs_img = cv2.cvtColor(fs_img, cv2.COLOR_BGR2RGB)
                    ff_img = cv2.imread(ff_path)
                    ff_img = cv2.cvtColor(ff_img, cv2.COLOR_BGR2RGB)
                    nt_img = cv2.imread(nt_path)
                    nt_img = cv2.cvtColor(nt_img, cv2.COLOR_BGR2RGB)
                    fake_img, Multi_FI_map_list, soft_label_list, mlab_list = Multiple_Soft_Blend(df_img, fs_img, ff_img, nt_img, real_img)
                    
                    # fake_img = cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR)
                    # cv2.imwrite("/home/liu/fcb1/decouple/clip_dfd/visualize/msblending.png", np.uint8(fake_img))
                    # FI_map = cv2.cvtColor(np.abs(FI_map), cv2.COLOR_RGB2BGR) * 10
                    # FI_map[FI_map>255]=255
                    # cv2.imwrite("/home/liu/fcb1/decouple/clip_dfd/visualize/msblending_FT_map.png", np.uint8(FI_map))
                    # exit(0)
                else:
                    if forgery_type == "Deepfakes":
                        soft_label_list = [1., 0., 0., 0.]
                        mlab_list = [3, 0, 0, 0]
                    elif forgery_type == "FaceSwap":
                        soft_label_list = [0., 1., 0., 0.]
                        mlab_list = [0, 3, 0, 0]
                    elif forgery_type == "Face2Face":
                        soft_label_list = [0., 0., 1., 0.]
                        mlab_list = [0, 0, 3, 0]
                    elif forgery_type == "NeuralTextures":
                        soft_label_list = [0., 0., 0., 1.]
                        mlab_list = [0, 0, 0, 3]
                
                
                landmark=np.load(lamk_path)[0]
                flag = False
            except Exception as e:
                # print(e)
                idx = torch.randint(low=0,high=len(self.fake_frame_list),size=(1,)).item()
        

        if self.phase == 'train':
            real_img, landmark, _, _, y0_new,y1_new,x0_new,x1_new=crop_face(real_img,landmark,bbox=None,margin=True,crop_by_bbox=False,abs_coord=True,phase=self.phase)
            # real_img, landmark, _, _, y0_new,y1_new,x0_new,x1_new=crop_face(real_img,landmark,bbox=None,margin=False,crop_by_bbox=False,abs_coord=True,phase=self.phase)
        else:
            real_img, landmark, _, _, y0_new,y1_new,x0_new,x1_new=crop_face(real_img,landmark,bbox=None,margin=False,crop_by_bbox=False,abs_coord=True,phase=self.phase)
 

        fake_img = fake_img[y0_new:y1_new,x0_new:x1_new]

        transformed=self.augmentation(image=fake_img.astype('uint8'),\
            image1=real_img.astype('uint8'))
        real_img=transformed['image1'] 
        fake_img=transformed['image']
        
        inter_flag = cv2.INTER_CUBIC # best
        real_img = cv2.resize(real_img, dsize=(self.img_size, self.img_size), interpolation=inter_flag)
        real_img=real_img.transpose((2, 0, 1)) / 255.
        fake_img = cv2.resize(fake_img, dsize=(self.img_size, self.img_size), interpolation=inter_flag)
        fake_img=fake_img.transpose((2, 0, 1)) / 255.
            
        FI_map = real_img - fake_img
        margin_mask, cls_mask, _ = self.get_patch_margin_and_cls_mask(FI_map)
        return fake_img, real_img, FI_map, margin_mask, cls_mask, soft_label_list, mlab_list
        
    def reorder_landmark(self,landmark):
        landmark_add=np.zeros((13,2))
        for idx,idx_l in enumerate([77,75,76,68,69,70,71,80,72,73,79,74,78]):
            landmark_add[idx]=landmark[idx_l]
        landmark[68:]=landmark_add
        return landmark
    
    

    def collate_fn(self,batch):
        img_f,img_r, residual, margin_mask, cls_mask, soft_labels, mlab_list =zip(*batch)
        # img_f,img_r,patch_mask,hist_w=zip(*batch)
        data={}
        data['img']=torch.cat([torch.tensor(img_r).float(),torch.tensor(img_f).float()],0)
        data['label']=torch.tensor([0]*len(img_r)+[1]*len(img_f))
        
        residual = torch.tensor(residual).float()
        real_zero = torch.zeros_like(residual)
        data['residual_map'] = torch.cat([real_zero, residual], dim=0)
        
        cls_mask = torch.tensor(cls_mask).float()
        real_mask = torch.zeros_like(cls_mask)
        data['cls_mask'] = torch.cat([real_mask, cls_mask], dim=0)
        
        data['margin_mask'] = torch.tensor(margin_mask).float()
        
        soft_labels = torch.tensor(soft_labels)
        real_labels = torch.zeros_like(soft_labels)
        data['soft_labels'] = torch.cat([real_labels, soft_labels], dim=0)
        
        mlab_list = torch.tensor(mlab_list)
        real_labels = torch.zeros_like(mlab_list)
        data['mlab_list'] = torch.cat([real_labels, mlab_list], dim=0)
        return data

    def worker_init_fn(self,worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)


def randaffine(residual):
    if np.random.rand() < 0.5:
        f=alb.Affine(
                translate_percent={'x':(-0.01,0.01),'y':(-0.0075,0.0075)},
                scale=[0.98,1/0.98],
                interpolation=cv2.INTER_LINEAR,
                fit_output=False,
                p=1)
    else:
        f=alb.Affine(
                translate_percent={'x':(-0.01,0.01),'y':(-0.0075,0.0075)},
                scale=[0.98,1/0.98],
                interpolation=cv2.INTER_CUBIC,
                fit_output=False,
                p=1)
            

    transformed=f(image=residual)
    residual=transformed['image']
    return residual

from skimage.transform import PiecewiseAffineTransform, warp
def random_deform(imageSize, nrows, ncols, mean=0, std=5):
    '''
    e.g. where nrows = 6, ncols = 7
    *_______*______*_____*______*______*_________*
    |                                            |
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *       *      *     *      *      *         *
    |                                            |
    *_______*______*_____*______*______*_________*

    '''
    h, w = imageSize
    rows = np.linspace(0, h-1, nrows).astype(np.int32)   # 生成坐标
    cols = np.linspace(0, w-1, ncols).astype(np.int32)
    rows, cols = np.meshgrid(rows, cols)        # 得到对应的坐标矩阵，即两个矩阵中分别保存x轴坐标与y轴坐标

    anchors = np.vstack([rows.flat, cols.flat]).T
    assert anchors.shape[1] == 2 and anchors.shape[0] == ncols * nrows
    deformed = anchors + np.random.normal(mean, std, size=anchors.shape)
    np.clip(deformed[:,0], 0, h-1, deformed[:,0])
    np.clip(deformed[:,1], 0, w-1, deformed[:,1])

    return anchors, deformed.astype(np.int32)


def piecewise_affine_transform(image, srcAnchor, tgtAnchor):
    trans = PiecewiseAffineTransform()
    trans.estimate(srcAnchor, tgtAnchor)
    if np.random.rand() < 0.5:
        warped = warp(image, trans, order=3).astype(np.float32)
        warped = warp(warped, trans.inverse, order=3).astype(np.float32)
    else:
        warped = warp(image, trans, order=1).astype(np.float32)
        warped = warp(warped, trans.inverse, order=1).astype(np.float32)
    return warped


def Normalize(w1, w2, w3, w4):
    _sum = np.sum([w1, w2, w3, w4])
    w1 = w1 / _sum
    w2 = w2 / _sum
    w3 = w3 / _sum
    w4 = w4 / _sum
    return w1, w2, w3, w4

def softmax(w1, w2, w3, w4):
    # 避免数值稳定性问题，减去输入中的最大值
    # a1 = np.random.randint(1, 10)
    # a2 = np.random.randint(1, 10)
    # a3 = np.random.randint(1, 10)
    # a4 = np.random.randint(1, 10)
    # x = [w1*a1, w2*a2, w3*a3, w4*a4]
    
    x = [w1*5., w2*5., w3*5., w4*5.] # best
    # exps = np.exp(x - np.max(x))
    exps = np.exp(x)
    exps = exps / np.sum(exps, axis=0)
    return exps[0], exps[1], exps[2], exps[3]

def get_level_label(w):
    if w == 0:
        lab = 0
    elif 0. < w <= 0.3:
        lab = 1
    elif 0.3 < w <= 0.6:
        lab = 2
    elif 0.6 < w:
        lab = 3
    return lab
        

def Multiple_Soft_Blend(df, fs, ff, nt, real):
    
    real = real.astype(np.float32)
    df = df.astype(np.float32)
    fs = fs.astype(np.float32)
    ff = ff.astype(np.float32)
    nt = nt.astype(np.float32)
    
    FI_map_df = real - df
    FI_map_fs = real - fs
    FI_map_ff = real - ff
    FI_map_nt = real - nt
    
    w1 = np.random.rand()
    w2 = np.random.rand()
    w3 = np.random.rand()
    w4 = np.random.rand()
    
    if np.random.rand() < 0.5:  # best
        w1, w2, w3, w4 = Normalize(w1, w2, w3, w4)
    else:
        w1, w2, w3, w4 = softmax(w1, w2, w3, w4)
        w1, w2, w3, w4 = Normalize(w1, w2, w3, w4)
        

    FI_map = FI_map_df * w1 + FI_map_fs * w2 + FI_map_ff * w3 + FI_map_nt * w4
    
    lab1 = get_level_label(w1)
    lab2 = get_level_label(w2)
    lab3 = get_level_label(w3)
    lab4 = get_level_label(w4)
    
    fake = real - FI_map
    fake[fake > 255] = 255
    fake[fake < 0] = 0
    return fake, [FI_map_df * w1, FI_map_fs * w2, FI_map_ff * w3, FI_map_nt * w4], [w1, w2, w3, w4], [lab1,lab2,lab3,lab4]




#v1 best
def augment_fake(real, fake):
    
    real = real.astype(np.float32)
    fake = fake.astype(np.float32)
    img_size = real.shape[0]
    residual_map = real - fake
    
    grid_size_list = [0, 2, 3, 4] # best
    grid_size = np.random.choice(grid_size_list)
    if grid_size != 0:
        lamk, lamk_deformed = random_deform((img_size, img_size), grid_size, grid_size)
        warped = piecewise_affine_transform(residual_map, lamk, lamk_deformed)
    else:
        warped = residual_map
    
    weight_of_residual_list = [0.7, 0.8, 0.9, 1., 1.] # best 
    w = np.random.choice(weight_of_residual_list)
    warped = warped * w
    

    p = np.random.rand()
    if p < 0.25:
        # print("blur")
        kz_list = [5]  # best
        # kz_list = [5, 5, 5, 7, 9]  
        kz = np.random.choice(kz_list)
        warped = cv2.GaussianBlur(warped, ksize=(kz,kz), sigmaX=0)
    elif 0.25 <= p < 0.45:   # best 
        # print("enhanced")
        #---------------------------- V1 BEST -----------------------------
        p1 = np.random.rand()
        if p1 < 0.5: 
            # print("enhanced")
            kernel = np.array([[-1, -1, -1],
                                [-1, 9, -1],
                                [-1, -1, -1]], dtype=np.float32)  # 定义拉普拉斯算子
            warped = cv2.filter2D(warped, -1, kernel)
        else:
            kernel = np.array([[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]], dtype=np.float32)  # 定义拉普拉斯算子
            warped = cv2.filter2D(warped, -1, kernel)
    
    fake_warped = real - warped
    fake_warped[fake_warped > 255] = 255
    fake_warped[fake_warped < 0] = 0
    
    return fake_warped


class RandomDownScale(alb.core.transforms_interface.ImageOnlyTransform):
    def apply(self,img,**params):
        return self.randomdownscale(img)

    def randomdownscale(self,img):
        keep_ratio=True
        keep_input_shape=True
        H,W,C=img.shape
        ratio_list=[2,4]
        # ratio_list=[2]
        r=ratio_list[np.random.randint(len(ratio_list))]
        # img_ds=cv2.resize(img,(int(W/r),int(H/r)),interpolation=cv2.INTER_NEAREST)
        img_ds=cv2.resize(img,(int(W/r),int(H/r)),interpolation=cv2.INTER_LINEAR)
        if keep_input_shape:
            img_ds=cv2.resize(img_ds,(W,H),interpolation=cv2.INTER_LINEAR)

        return img_ds


#-------------------- For evaluation and test ------------------#
class VideoFFPP_Dataset(Dataset):
    def __init__(self,compress='raw',image_size=384,phase = "test", tamper="all", n_frames=16):# raw、c23、c40
        super().__init__()
        self.original_root = f'/home/liu/sdb/FFPP/original_sequences/youtube/{compress}/images/'
        self.Deepfakes_root = f'/home/liu/sdb/FFPP/manipulated_sequences/Deepfakes/{compress}/images/'
        self.NeuralTextures_root = f'/home/liu/sdb/FFPP/manipulated_sequences/NeuralTextures/{compress}/images/'
        self.FaseSwap_root = f'/home/liu/sdb/FFPP/manipulated_sequences/FaceSwap/{compress}/images/'
        self.Face2Face_root = f'/home/liu/sdb/FFPP/manipulated_sequences/Face2Face/{compress}/images/'
 
        print('start')
        st = time.time()
        # if phase == "test":
        #     n_frames = 2
        # else:
        #     n_frames = 4
        original_list,Deepfakes_list,NeuralTextures_list,FaseSwap_list,Face2Face_list,landmark_list = init_fff(phase=phase,
                                                                                                               dataset_paths=self.original_root,
                                                                                                               DF=self.Deepfakes_root,
                                                                                                               NT=self.NeuralTextures_root,
                                                                                                               FS=self.FaseSwap_root,
                                                                                                               FF=self.Face2Face_root,
                                                                                                               n_frames=n_frames)
        self.real_frame_list = original_list
        self.landmark_list = landmark_list
        if tamper == 'all':
            self.fake_frame_list = Deepfakes_list + NeuralTextures_list + FaseSwap_list + Face2Face_list
            # Deepfakes_list[:len(Deepfakes_list)//4] + NeuralTextures_list[len(NeuralTextures_list)//4:len(NeuralTextures_list)//2] + FaseSwap_list[len(FaseSwap_list)//2:len(FaseSwap_list)*3//4] + Face2Face_list[len(Face2Face_list)*3//4:]
            # self.fake_frame_list = Deepfakes_list[:len(Deepfakes_list)//4] + FaseSwap_list[len(FaseSwap_list)//4:len(FaseSwap_list)//2] + FaseSwap_list[len(FaseSwap_list)//2:len(FaseSwap_list)*3//4] + Face2Face_list[len(Face2Face_list)*3//4:]
        elif tamper == 'DF':
            self.fake_frame_list = Deepfakes_list
        elif tamper == 'NT':
            self.fake_frame_list = NeuralTextures_list
        elif tamper == 'FS':
            self.fake_frame_list = FaseSwap_list
        elif tamper == 'FF':
            self.fake_frame_list = Face2Face_list
        ed = time.time()
        print(f'load... {ed -st:.2f} s')
        print('Real samples : {}'.format(len(self.real_frame_list)))
        print('Fake samples : {}'.format(len(self.fake_frame_list)))
        
        last_name = self.real_frame_list[0].split('/')[-1]
        last_video_fold = self.real_frame_list[0].replace(last_name, "")
        self.video_list = [last_video_fold]
        self.label_list = [0]
        for i in range(1, len(self.real_frame_list)):
            cur_name =  self.real_frame_list[i].split('/')[-1]
            cur_video_fold = self.real_frame_list[i].replace(cur_name, "")
            if cur_video_fold == self.video_list[-1]:
                continue
            else:
                self.video_list.append(cur_video_fold)
                self.label_list.append(0)
        
        for i in range(0, len(self.fake_frame_list)):
            cur_name =  self.fake_frame_list[i].split('/')[-1]
            cur_video_fold = self.fake_frame_list[i].replace(cur_name, "")
            if cur_video_fold == self.video_list[-1]:
                continue
            else:
                self.video_list.append(cur_video_fold)
                self.label_list.append(1)

        trans = [
            tr.ToTensor(),
            # tr.Resize((image_size, image_size), antialias=True)
            # tr.Resize((384, 384), antialias=True),
            
            tr.Resize((224, 224), InterpolationMode.BICUBIC, antialias=True),
            tr.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

        ]
        self.trans = tr.Compose(trans)
        # self.tamper = tamper
        self.idx_real = 0
        self.max_real = len(self.real_frame_list)
        # self.compress = compress
        self.img_size = image_size
        
        self.landmark_root = "/home/liu/fcb1/dataset/FFPP_16/landmarks/"
        self.real_root = self.original_root
        self.phase = phase
        
    def __len__(self):
        return len(self.video_list)    
    
    def __getitem__(self, index):
        video_path = self.video_list[index]
        target = self.label_list[index]
        
        video = None
        try:
            for _root, _, files in os.walk(video_path):
                if files:
                    for file in files:
                        # file = file.decode()
                        img_path = os.path.join(_root, file)
                        
                        real_id = img_path.split('/')[-2].split('_')[0]
                        frame_id = img_path.split('/')[-1].split('.')[0] + '.npy'
                        lamk_path = self.landmark_root + real_id + '/' + frame_id
                        landmark=np.load(lamk_path)[0]
                        
                        img = cv2.imread(img_path)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        # img, _, _, _, y0_new,y1_new,x0_new,x1_new=crop_face(img,\
                        #     landmark,bbox=None,margin=False,crop_by_bbox=False,abs_coord=True,phase=self.phase)
                        img, _, _, _, y0_new,y1_new,x0_new,x1_new=crop_face(img,\
                            landmark,bbox=None,margin=True,crop_by_bbox=False,abs_coord=True,phase=self.phase)
                        
                        img = self.trans(img)
                        img = img.unsqueeze(0)
                        video = img if video is None else torch.cat((video, img), dim=0)
                break
        except Exception as e:
            # print(e)
            pass
                    
        return video, target

    def collate_fn(self, batch):
        video, target = zip(*batch)
        video = video[0]
        target = torch.tensor(target).unsqueeze(0)
        return video, target

    def worker_init_fn(self,worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)


       
if __name__ == "__main__":
    seed = 11
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    print('Random seed :', seed)
    # train_dataset = FFPP_Dataset(phase='train',image_size=224,compress='c23',tamper='all')
    train_dataset = VideoFFPP_Dataset(phase='test',image_size=224,compress='c23',tamper='all')
    # train_dataset.__getitem__(2)
    batch_size_sbi = 32
    train_set = torch.utils.data.DataLoader(train_dataset,
                                            batch_size = 1,
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=0,
                                            collate_fn=train_dataset.collate_fn,
                                            worker_init_fn=train_dataset.worker_init_fn
                                            )
    print(len(train_set))
    for step_id, data in enumerate(train_set):
        # img = data['img']
        # print(img.shape)
        # label = data['label']
        # print(label)
        # s_label = data['soft_labels']
        # print(s_label)
        
        # For videoFFPP test
        img = data[0]
        print(img.shape)
        label = data[1]
        print(label)

    