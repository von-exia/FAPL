from thop import profile
import torch
device = "cuda"


from CLIP import clip
from CLIP_Model import CLIP_Detector
clip_model, preprocess = clip.load("ViT-B/16", \
    device=torch.device("cpu"), download_root="/home/liu/fcb1/clip_model")#ViT-B/16
model = CLIP_Detector(clip_model).to('cuda')
model.init_text_embed_with_pretrainweight("FAPL.pth")
img_size = 224

img = torch.randn(1, 3, img_size, img_size)      # (1, 3, 224, 224)分别表示输入的图片数量，图像通道处，图像高度和宽度
img= img.to(device)


flops, params = profile(model, inputs=(img,))

print('FLOPs = ' + str(flops/1000**3) + 'G') # GFLOPs
print('Params = ' + str(params/1000**2) + 'M')

# flops, params, ret = profile(model, inputs=(img,), ret_layer_info=True) # ours
# print('FLOPs = ' + str((flops-ret['text_encoderx'][0])/1000**3) + 'G') # GFLOPs
# print('Params = ' + str((params-ret['text_encoderx'][1])/1000**2) + 'M')
# print(ret)
