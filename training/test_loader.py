import os
import torch
import torch.utils.data as data
import torchvision.transforms as tr
from torchvision.transforms import InterpolationMode
from PIL import Image
import random
import numpy as np
import torch.nn.functional as F
import cv2
import torchvision.transforms.functional as TF
import io
from functools import reduce
from skimage.transform import PiecewiseAffineTransform, warp

from numpy.linalg import lstsq, inv
from numpy.linalg import matrix_rank as rank
import cv2
import os
import numpy as np

inter_flag = cv2.INTER_CUBIC
# inter_flag = cv2.INTER_LINEAR
dict_inter = {cv2.INTER_CUBIC:"CUBIC", cv2.INTER_LINEAR:"Linear"}
print(f"Currently, testing with inpterpolation {dict_inter[inter_flag]}")

class VideoCDF(data.Dataset):
    def __init__(self,image_size=224,normalize=True) -> None:
        super().__init__()
        # root = '/home/liu/fcb/dataset/VideoCDF'
        idx_root = '/home/liu/fcb1/dataset/VideoCDF/idx'
        txt_root = '/home/liu/fcb1/dataset/VideoCDF/all.txt'
        allData = np.loadtxt(txt_root, dtype='str', delimiter='\t')

        self.video_list = []
        self.target_list = []
        self.idx_list = []

        
        for _root, _, files in os.walk(idx_root):
            if files:
                for file in files:
                    idx_path = os.path.join(_root, file)
                    self.idx_list.append(idx_path)
        self.idx_list = list(sorted(self.idx_list))
        # print(len(self.idx_list))

        allData = list(sorted(allData, key=lambda x:x[0]))
        # print(len(allData))
    
        
        for i in range(len(self.idx_list)):
            video_path = self.idx_list[i].replace("/idx", "").replace(".npy", "")
            self.video_list.append(video_path)
            # print(self.idx_list[i])
            # print(self.video_list[i])
        for j in range(len(self.video_list)):
            for i in range(len(allData)):
                if self.video_list[j] == allData[i][0]:
                    self.target_list.append(int(allData[i][1]))
                    break
        self.length = len(self.video_list)
        self.normalize = normalize
        self.norm = tr.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.image_size = image_size
        
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        video_path = self.video_list[index]
        target = self.target_list[index]
        idx_path = self.idx_list[index]
        
        video = None
        # try:
        for _root, _, files in os.walk(video_path):
            if files:
                for file in files:
                    # file = file.decode()
                    img_path = os.path.join(_root, file)
                    # img = Image.open(img_path)
                    # img = self.totensor(img)
                    # img = img.unsqueeze(0)
                    
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=inter_flag)
            
                    image = image.transpose((2, 0, 1)) / 255.
                    img = torch.tensor(image).float()
                    if self.normalize:
                        img = self.norm(img)
                    img = img.unsqueeze_(0)
                    video = img if video is None else torch.cat((video, img), dim=0)
        if video is None:
            print(video_path)
        # except Exception as e:
        #     print(e)
                    
        return video, target, idx_path
                    

    
    def collate_fn(self, batch):
        video, target, idx_path = zip(*batch)
        video = video[0]
        idx_path = idx_path[0]
        target = torch.tensor(target).unsqueeze(0)
        return video, target, idx_path



class VideoDFDC(data.Dataset):
    def __init__(self,image_size=224,normalize=True) -> None:
        super().__init__()
        idx_root = '/home/liu/fcb1/dataset/VideoDFDC/idx'
        txt_root = '/home/liu/fcb1/dataset/VideoDFDC/all.txt'
        allData = np.loadtxt(txt_root, dtype='str', delimiter='\t')
        
        self.video_list = []
        self.target_list = []
        self.idx_list = []
        
        
        for _root, _, files in os.walk(idx_root):
            if files:
                for file in files:
                    idx_path = os.path.join(_root, file)
                    self.idx_list.append(idx_path)
        self.idx_list = list(sorted(self.idx_list))
        # print(len(self.idx_list))

        allData = list(sorted(allData, key=lambda x:x[0]))
        # print(len(allData))
    
        
        for i in range(len(self.idx_list)):
            video_path = self.idx_list[i].replace("/idx", "").replace(".npy", "")
            self.video_list.append(video_path)
            # print(self.idx_list[i])
            # print(self.video_list[i])
        for j in range(len(self.video_list)):
            for i in range(len(allData)):
                if self.video_list[j] == allData[i][0]:
                    self.target_list.append(int(allData[i][1]))
                    break
        self.length = len(self.video_list)
        self.normalize = normalize
        self.norm = tr.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.image_size = image_size
        
        
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        video_path = self.video_list[index]
        target = self.target_list[index]
        idx_path = self.idx_list[index]
        
        video = None
        # try:
        for _root, _, files in os.walk(video_path):
            if files:
                for file in files:
                    # file = file.decode()
                    img_path = os.path.join(_root, file)
                    # img = Image.open(img_path)
                    # img = self.totensor(img)
                    # img = img.unsqueeze(0)
                    
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=inter_flag)
            
                    image = image.transpose((2, 0, 1)) / 255.
                    img = torch.tensor(image).float()
                    if self.normalize:
                        img = self.norm(img)
                    img = img.unsqueeze_(0)
                    video = img if video is None else torch.cat((video, img), dim=0)
        if video is None:
            print(video_path)
        # except Exception as e:
        #     print(e)
                    
        return video, target, idx_path
    
    def collate_fn(self, batch):
        video, target, idx_path = zip(*batch)
        video = video[0]
        idx_path = idx_path[0]
        target = torch.tensor(target).unsqueeze(0)
        return video, target, idx_path

class VideoDFDCP(data.Dataset):
    def __init__(self,image_size=224,normalize=True) -> None:
        super().__init__()
        root = '/home/liu/fcb1/dataset/VideoDFDCP'
        txt_root = '/home/liu/fcb1/dataset/VideoDFDCP/all.txt'
        allData = np.loadtxt(txt_root, dtype='str', delimiter='\t')
        self.video_list = []
        self.target_list = []
        
        self.totensor = [
        tr.ToTensor(),
        tr.Resize((image_size, image_size),antialias=True)
        ]
        self.totensor = tr.Compose(self.totensor)
        
        for i in range(len(allData)):
            self.video_list.append(allData[i, 0])
            
            if allData[i, 1] == 'True':
                label = 1
            elif allData[i, 1] == 'False':
                label = 0
            self.target_list.append(label)
        self.length = len(self.video_list)
        self.normalize = normalize
        self.norm = tr.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.image_size = image_size

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        video_path = self.video_list[index]
        target = self.target_list[index]
        
        video = None
        # try:
        for _root, _, files in os.walk(video_path):
            if files:
                for file in files:
                    file = file.decode()
                    img_path = os.path.join(_root, file)
                    # img = Image.open(img_path)
                    # img = self.totensor(img)
                    # img = img.unsqueeze(0)
                    
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=inter_flag)
            
                    image = image.transpose((2, 0, 1)) / 255.
                    img = torch.tensor(image).float()
                    if self.normalize:
                        img = self.norm(img)
                    img = img.unsqueeze_(0)
                    video = img if video is None else torch.cat((video, img), dim=0)
        # except Exception as e:
        #     print(e)
                    
        return video, target
    
    def collate_fn(self, batch):
        video, target = zip(*batch)
        # video, target = batch
        video = video[0]
        target = torch.tensor(target).unsqueeze(0)
        return video, target

class VideoDFV1(data.Dataset):
    def __init__(self,image_size=224,normalize=True) -> None:
        super().__init__()
        idx_root = '/home/liu/fcb1/dataset/VideoDFV1/idx'
        txt_root = '/home/liu/fcb1/dataset/VideoDFV1/all.txt'
        allData = np.loadtxt(txt_root, dtype='str', delimiter='\t')
        self.video_list = []
        self.target_list = []
        self.idx_list = []
        
        
        for _root, _, files in os.walk(idx_root):
            if files:
                for file in files:
                    idx_path = os.path.join(_root, file)
                    self.idx_list.append(idx_path)
        self.idx_list = list(sorted(self.idx_list))
        # print(len(self.idx_list))

        allData = list(sorted(allData, key=lambda x:x[0]))
        # print(len(allData))
    
        
        for i in range(len(self.idx_list)):
            video_path = self.idx_list[i].replace("/idx", "").replace(".npy", "")
            self.video_list.append(video_path)
            # print(self.idx_list[i])
            # print(self.video_list[i])
        for j in range(len(self.video_list)):
            for i in range(len(allData)):
                if self.video_list[j] == allData[i][0]:
                    self.target_list.append(int(allData[i][1]))
                    break
        
        # for i in range(len(allData)):
        #     self.video_list.append(allData[i, 0])
        #     self.target_list.append(int(allData[i, 1]))
        self.length = len(self.video_list)
        self.normalize = normalize
        self.norm = tr.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.image_size = image_size

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        video_path = self.video_list[index]
        target = self.target_list[index]
        idx_path = self.idx_list[index]
        
        video = None
        # try:
        for _root, _, files in os.walk(video_path):
            if files:
                for file in files:
                    # file = file.decode()
                    img_path = os.path.join(_root, file)
                    # img = Image.open(img_path)
                    # img = self.totensor(img)
                    # img = img.unsqueeze(0)
                    
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=inter_flag)
            
                    image = image.transpose((2, 0, 1)) / 255.
                    img = torch.tensor(image).float()
                    if self.normalize:
                        img = self.norm(img)
                    img = img.unsqueeze_(0)
                    video = img if video is None else torch.cat((video, img), dim=0)
        # except Exception as e:
        #     print(e)
                    
        return video, target, idx_path
    
    def collate_fn(self, batch):
        video, target, idx_path = zip(*batch)
        video = video[0]
        idx_path = idx_path[0]
        target = torch.tensor(target).unsqueeze(0)
        return video, target, idx_path
    

class VideoDFD(data.Dataset):
    def __init__(self,image_size=224,normalize=True) -> None:
        super().__init__()
        idx_root = '/home/liu/fcb1/dataset/VideoDFD/idx'
        txt_root = '/home/liu/fcb1/dataset/VideoDFD/all.txt'
        allData = np.loadtxt(txt_root, dtype='str', delimiter='\t')
        
        self.video_list = []
        self.target_list = []
        
        self.idx_list = []
        
        
        for _root, _, files in os.walk(idx_root):
            if files:
                for file in files:
                    idx_path = os.path.join(_root, file)
                    self.idx_list.append(idx_path)
        self.idx_list = list(sorted(self.idx_list))
    
        
        # for i in range(len(allData)):
        #     self.video_list.append(allData[i, 0])
        #     self.target_list.append(int(allData[i, 1]))
            
        for i in range(len(self.idx_list)):
            video_path = self.idx_list[i].replace("/idx", "").replace(".npy", "")
            self.video_list.append(video_path)
            # print(self.idx_list[i])
            # print(self.video_list[i])
        for j in range(len(self.video_list)):
            for i in range(len(allData)):
                if self.video_list[j] == allData[i][0]:
                    self.target_list.append(int(allData[i][1]))
                    break
            
        self.length = len(self.video_list)
        self.normalize = normalize
        self.norm = tr.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        self.image_size = image_size
        
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        video_path = self.video_list[index]
        target = self.target_list[index]
        idx_path = self.idx_list[index]
        
        video = None
        # try:
        for _root, _, files in os.walk(video_path):
            if files:
                for file in files:
                    # file = file.decode()
                    img_path = os.path.join(_root, file)
                    # img = Image.open(img_path)
                    # img = self.totensor(img)
                    # img = img.unsqueeze(0)
                    
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=inter_flag)
            
                    image = image.transpose((2, 0, 1)) / 255.
                    img = torch.tensor(image).float()
                    if self.normalize:
                        img = self.norm(img)
                    img = img.unsqueeze_(0)
                    video = img if video is None else torch.cat((video, img), dim=0)
        # except Exception as e:
        #     print(e)
                    
        return video, target, idx_path
    
    def collate_fn(self, batch):
        video, target, idx_path = zip(*batch)
        video = video[0]
        idx_path = idx_path[0]
        target = torch.tensor(target).unsqueeze(0)
        return video, target, idx_path
    
    # def collate_fn(self, batch):
    #     video, target = zip(*batch)
    #     video = video[0]
    #     target = torch.tensor(target).unsqueeze(0)
    #     return video, target
    
class WildDeepfake(data.Dataset):
    def __init__(self,image_size=224,normalize=True) -> None:
        super().__init__()
        
        txt_root = '/home/liu/sdb/wildDeepfake/WildDeefake.txt'
        allData = np.loadtxt(txt_root, dtype='str', delimiter='\t')
        
        self.video_list = []
        self.target_list = []
        self.idx_list = []
        self.image_size = image_size
        self.normalize = normalize
        self.norm = tr.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))

        allData = list(sorted(allData, key=lambda x:x[0]))
        # print(allData)
        for i in range(len(allData)):
            parts = allData[i].rsplit(' ', 1)
            self.video_list.append(parts[0])
            self.target_list.append(int(parts[1].strip()))
            # break
        self.length = len(self.video_list)
        
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        video_path = self.video_list[index]
        target = self.target_list[index]
        
        video = None
        # try:
        for _root, _, files in os.walk(video_path):
            if files:
                for file in files:
                    # file = file.decode()
                    img_path = os.path.join(_root, file)
                    # img = Image.open(img_path)
                    # img = self.totensor(img)
                    # img = img.unsqueeze(0)
                    
                    image = cv2.imread(img_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    inter_flag = cv2.INTER_CUBIC
                    image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=inter_flag)
            
                    image = image.transpose((2, 0, 1)) / 255.
                    img = torch.tensor(image).float()
                    if self.normalize:
                        img = self.norm(img)
                    img = img.unsqueeze_(0)
                    video = img if video is None else torch.cat((video, img), dim=0)
                    if video.shape[0] >= 32:
                        video = video[:32]
                        break
                    break
        if video is None:
            print(video_path)
        # except Exception as e:
        #     print(e)
                    
        return video, target
                    

    def collate_fn(self, batch):
        video, target = zip(*batch)
        video = video[0]
        target = torch.tensor(target).unsqueeze(0)
        return video, target



def Get_DataLoader(dataset_name="VideoCDF",image_size=224,normalize=True):
    if dataset_name =="VideoCDF":
        dataset = VideoCDF(image_size=image_size, normalize=normalize)
        return data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=dataset.collate_fn)
    elif dataset_name =="VideoDFD":
        dataset = VideoDFD(image_size=image_size, normalize=normalize)
        return data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=dataset.collate_fn)
    elif dataset_name =="VideoDFDC":
        dataset = VideoDFDC(image_size=image_size, normalize=normalize)
        return data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=dataset.collate_fn)
    elif dataset_name =="VideoDFDCP":
        dataset = VideoDFDCP(image_size=image_size, normalize=normalize)
        return data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=dataset.collate_fn)
    elif dataset_name =="VideoDFV1":
        dataset = VideoDFV1(image_size=image_size, normalize=normalize)
        return data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=dataset.collate_fn)
    elif dataset_name =="WildDeepFake":
        dataset = WildDeepfake(image_size=image_size, normalize=normalize)
        return data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=dataset.collate_fn)
    else:
        raise NotImplementedError("No this kind of dataset!")


