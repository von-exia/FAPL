import gc
import time
import torch
import random
import numpy as np
import cv2
import random
from metrics_util import *
from test_loader import Get_DataLoader


seed = 1
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# CUDA_LAUNCH_BLOCKING=1

def Tensor2cv(img_tensor):
    img_tensor = img_tensor.permute(1, 2, 0).cpu()
    img_numpy = img_tensor.numpy() * 255
    img_numpy = np.uint8(img_numpy)
    img_numpy = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
    return img_numpy


img_size = 224
check = True
from CLIP import clip
clip_model, preprocess = clip.load("ViT-B/16", \
    device=torch.device("cpu"), download_root="/home/liu/fcb1/clip_model")#ViT-B/16
from CLIP_Model import CLIP_Detector
model = CLIP_Detector(clip_model).to('cuda')
if check:
    wp = '/home/liu/fcb1/CPL/trained_weights/FAPL.pth'
    model.init_text_embed_with_pretrainweight(wp)
    print('parameters inherited from ', wp)


'''
通过idx判断使用VideoCDF、VideoDFDC还是VideoDFV1、VideoDFD、VideoDFDCP
若idx为False,则测试VideoDFV1、VideoDFD、VideoDFDCP
若idx为True,则测试VideoCDF、VideoDFDC
'''
test_name = "VideoCDF" # 518
# test_name = "VideoDFDC" # 4986
test_name = 'VideoDFV1' # 2412
# test_name = 'VideoDFD' # 4068
# test_name = 'VideoDFDCP' # 777
test_name = 'WildDeepFake'
if test_name in ["VideoCDF","VideoDFDC","VideoDFD", "VideoDFV1"]:
    idx = True
if test_name in ["VideoDFDCP", 'WildDeepFake']:
    idx = False

TestSet = Get_DataLoader(dataset_name=test_name,
                          image_size=img_size,
                          normalize=True)

print(test_name)
print(idx)
# frame_level = False
frame_level = True

if frame_level:
    print('testing in frame level now!')
else:
    print('testing in video level now!')

model.eval()
outputs = None
testargets = None
print('start testing...')
print(len(TestSet))
cnt = 0
feats = None

with torch.no_grad():
    start = time.time()
    for step_id, datas in enumerate(TestSet):
        gc.collect()
        torch.cuda.empty_cache()
        if datas[0] is None:
            continue
        img = datas[0].cuda()
        targets = datas[1].cuda()
        if idx:
            idx_path = datas[2]
            idx_list = np.load(idx_path).tolist()
            
        output = model.test_forward(img)
        cls_final = torch.sigmoid(output)
        
        # output = model(img)
        # cls_final = torch.softmax(output, -1)[:, -1]
        
        if idx:
            pred_list=[]
            # feat_list=[]
            idx_img=-1
            for i in range(len(cls_final)):
                if idx_list[i]!=idx_img:
                    pred_list.append([])
                    # feat_list.append([])
                    idx_img=idx_list[i]
                pred_list[-1].append(cls_final[i].item())
                # feat_list[-1].append(feat[i, :].cpu().numpy())

            pred_res=np.zeros(len(pred_list))
            # pred_res_feat = np.zeros((len(pred_list),1536))
            # pred_res_feat = np.zeros((len(pred_list),1792))
            for i in range(len(pred_res)):
                pred_res[i]=max(pred_list[i])
                # idx = int(np.argmax(pred_list[i]))
                # pred_res_feat[i, :] = feat_list[i][idx]
                # print(feat_batch)
            if frame_level:
                pred=pred_res
                n_frames = len(pred)
                targets = targets.expand(n_frames, -1)
                # feat_batch = torch.tensor(pred_res_feat, dtype=torch.float32)
                # feats = feat_batch if feats is None else torch.cat((feats, feat_batch), dim=0)
            else:
                pred=pred_res.mean()
            cls_final = torch.tensor(pred, dtype=torch.float32).unsqueeze(-1)
        else:
            if frame_level:
                n_frames = img.shape[0]
                targets = targets.expand(n_frames, -1)
            else:
                cls_final = cls_final.mean().unsqueeze(0)
        outputs = cls_final if outputs is None else torch.cat((outputs, cls_final), dim=0)
        testargets = targets if testargets is None else torch.cat((testargets, targets), dim=0)
        
    
        if not (step_id+1) % 100:
            now_percent = int(step_id / len(TestSet) * 100)
            print(f"Test: complete {now_percent} %")

'''
在图像层面保存t_SNE
'''
# if frame_level:
#     save_tSNE(feats.detach().cpu().numpy(), testargets.detach().cpu(), save_name=test_name)
cdfauc = compute_AUC(testargets.cpu().detach(), outputs.cpu().detach(), n_class=2)
ap = compute_AP(testargets.cpu().detach(), outputs.cpu().detach())
'''
t对于acc、f1、recall、precision都有影响
'''
t = 0.5
print('threshold :', t)
acc = compute_ACC(testargets.cpu().detach(), outputs.cpu().detach(), n_class=1, t=t)
f1 = compute_F1(testargets.cpu().detach(), outputs.cpu().detach(), n_class=1, t=t)
recall = compute_recall(testargets.cpu().detach(), outputs.cpu().detach(), n_class=1, t=t)
pre = compute_precision(testargets.cpu().detach(), outputs.cpu().detach(), n_class=1, t=t)
print(f'{test_name} test AUC:{cdfauc:.4f}; AP: {ap:.4f}')
print(f'acc : {acc:.4f} ; f1 : {f1:.4f} ; recall : {recall:.4f} ; precision : {pre:.4f}')


end = time.time()
spent = (end - start) / 60
print(f'spent time: {spent:.4f} min')
print('ending')
gc.collect()
torch.cuda.empty_cache()
exit(0)


