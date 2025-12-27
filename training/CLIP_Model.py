import sys
from CLIP import clip, longclip
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

from torch.autograd import Function
class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_
        self.begin = lambda_
        self.steps = 1440
        # self.add_step = 1e-7
        self.end = 5e-6
        self.add_step = (self.end - self.begin) / self.steps

    def forward(self, x):
        if self.lambda_ < self.end:
            self.lambda_ += self.add_step
        return GradientReversalFunction.apply(x, self.lambda_)


class MLP(nn.Module):
    def __init__(self, embed_dim=512, expand=4):
        super().__init__()
        
        self.fc1 = nn.Linear(embed_dim, embed_dim*expand)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(embed_dim*expand, embed_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
        


        
# # V1 best
class Image_Encoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual
        self.dtype = clip_model.dtype
        
        self.text_linear = nn.Sequential(
            nn.Linear(512, 768)
        )
        
        scale = 5 ** -0.5
        self.text_position_embed = nn.Parameter(scale * torch.randn(5, 768))

    
    def forward(self, x: torch.Tensor,text:torch.Tensor):
        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.visual.class_embedding.to(x.dtype) +\
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        n = x.shape[0]
        
        text = self.text_linear(text).unsqueeze(0).expand(n, -1, -1) + self.text_position_embed
        
        x = torch.cat([x, text], dim=1)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        img_tokens = x[:, 1:, :]
        # x = self.visual.ln_post(x[:, 0, :]) # for V2
        x = x[:, 0, :]
        
        if self.visual.proj is not None:
            x = x @ self.visual.proj
            
            
        return x, img_tokens
            

# V2    
# class Image_Encoder(nn.Module):
#     def __init__(self, clip_model):
#         super().__init__()
#         self.visual = clip_model.visual
#         self.dtype = clip_model.dtype
        
#         self.text_linear = nn.Linear(512, 768)
#         self.text_ln = nn.LayerNorm(768)
#         self.sa = MHSA(768, 1)
        
        
#         scale = 5 ** -0.5
#         self.text_position_embed = nn.Parameter(scale * torch.randn(5, 768))

    
#     def forward(self, x: torch.Tensor,text:torch.Tensor):
#         x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
#         x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#         x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
#         x = torch.cat([self.visual.class_embedding.to(x.dtype) +\
#             torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
#         x = x + self.visual.positional_embedding.to(x.dtype)
#         n = x.shape[0]
        
#         tex_res = self.text_linear(text).unsqueeze(0).expand(n, -1, -1) + self.text_position_embed
#         text = self.sa(self.text_ln(tex_res)) + tex_res
#         text = text.permute(1, 0, 2)
#         # x = torch.cat([x, text], dim=1)
        
#         x = self.visual.ln_pre(x)

#         x = x.permute(1, 0, 2)  # NLD -> LND
#         # x = self.visual.transformer(x)
#         for id, block in enumerate(self.visual.transformer.resblocks):
#             if id == 10:
#                 x = torch.cat([x, text], dim=0)
#             x = block(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD

#         img_tokens = x[:, 1:, :]
#         # x = self.visual.ln_post(x[:, 0, :])
#         x = x[:, 0, :]
        
#         if self.visual.proj is not None:
#             x = x @ self.visual.proj
            
            
#         return x, img_tokens


class Text_Encoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # print(tokenized_prompts.shape)
        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x.type(self.dtype)
        # x = self.ln_final(x) # for V2
        
        # x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        x = x[torch.arange(x.shape[0]), tokenized_prompts] @ self.text_projection
        
        return x

class QFormer_Layer(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.sa = MHSA(embed_dim, 8)
        
        self.ln_q = nn.LayerNorm(embed_dim)
        self.ln_fi = nn.LayerNorm(embed_dim)
        # self.cross_a = nn.MultiheadAttention(embed_dim, 8, batch_first=True)
        self.cross_a = nn.MultiheadAttention(embed_dim, 1, batch_first=True)
        
        self.ln3 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim)
    
    def forward(self, x, queries):
        
        residual = queries
        queries = self.ln1(queries)
        residual = self.sa(queries) + residual
        
        queries = self.ln_q(residual)
        x = self.ln_fi(x)
        cross_queries_residual, _ = self.cross_a(queries, x, x, need_weights=False)
        cross_queries_residual = cross_queries_residual + residual
        
        queries = self.ln3(cross_queries_residual)
        queries = self.mlp(queries) + cross_queries_residual
        return queries
        

# query_len = 16 # best
query_len = 64
n_l = 1
class QFormer(nn.Module):
    def __init__(self, n_q=query_len+2, embed_dim=512, n_layer=n_l): # n_q 18=16+2 best
        super().__init__()
        
        self.query = nn.Parameter(torch.randn(size=[n_q, embed_dim]), True)
        nn.init.normal_(self.query) # best
        # nn.init.kaiming_normal_(self.query, mode='fan_out', nonlinearity='relu')
        
        self.layer = nn.ModuleList()
        for i in range(n_layer):
            self.layer.append(QFormer_Layer(embed_dim))
        
    
    def forward(self, fi_feat):
        n = fi_feat.shape[0]
        
        queries = self.query.expand(n, -1, -1)
        for i in range(len(self.layer)):
            queries = self.layer[i](fi_feat, queries)
        
        # queries = self.proj(queries.permute(0, 2, 1)).permute(0, 2, 1)
        return queries   
    
    
# Modified by Text Encoder
class FI_Attn_Encoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        from copy import deepcopy
        clip_model = deepcopy(clip_model)
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        
        self.q_len = query_len+2
        zeros = torch.zeros(size=[77-self.q_len], dtype=torch.long)
        # zeros = torch.zeros(size=[248-self.q_len], dtype=torch.long)
        self.zeros = nn.Parameter(clip_model.token_embedding(zeros), False)
        

    def forward(self, prompts):
        # print(tokenized_prompts.shape)
        zeros = self.zeros.expand(prompts.shape[0], -1, -1).to(prompts.device)
        prompts = torch.cat([prompts, zeros], dim=1)

        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)  # NLD -> LND
        
        
        x = self.transformer(x)
        # for i in range(len(self.transformer.resblocks)):
        #     x = self.transformer.resblocks(x)
        #     if i >= 5:
        #         break


        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x.type(self.dtype)
        # x = self.ln_final(x) # for V2
        
        
        tokenized_prompts = torch.full(size=[prompts.shape[0]], fill_value=self.q_len-1, dtype=torch.long) # n_q 18
        x = x[torch.arange(x.shape[0]), tokenized_prompts] @ self.text_projection
        # x = torch.sigmoid(x) # for V2
        
        return x
    
    
    
    
    
class Prompt_processor(nn.Module):
    def __init__(self, model, initials=None):
        super(Prompt_processor,self).__init__()
        print("The initial prompts are:",initials)
        self.text_encoder = Text_Encoder(model)
        
        # if isinstance(initials,list):
        #     text = clip.tokenize(initials)
        #     self.embedding_prompt = nn.Parameter(model.token_embedding(text).requires_grad_())
        #     self.num_prompts = self.embedding_prompt.shape[0]
        if isinstance(initials,list):
            # text = clip.tokenize(initials)
            text = longclip.tokenize(initials, truncate=True)
            self.tokenized_text = text
            # self.teacher_embedding_prompt = nn.Parameter(model.token_embedding(text[:16]), False)
            self.embedding_prompt = nn.Parameter(model.token_embedding(text).requires_grad_())
            self.num_prompts = self.embedding_prompt
        elif isinstance(initials,str):
            prompt_path=initials

            state_dict = torch.load(prompt_path)
            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            self.embedding_prompt=nn.Parameter(new_state_dict['embedding_prompt']).cuda()
            self.embedding_prompt.requires_grad = True
        else:
            print([" ".join(["X"]*\
                16)," ".join(["X"]*16)])
            self.embedding_prompt=torch.nn.init.xavier_normal_(nn.Parameter(model.token_embedding([" ".join(["X"]*\
                16)," ".join(["X"]*16)]).requires_grad_())).cuda()
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self):
        tokenized_prompts = torch.cat([p.argmax(dim=-1).unsqueeze(0) for p in self.tokenized_text], dim=0)
        # print(tokenized_prompts)
        text_features = self.text_encoder(self.embedding_prompt,tokenized_prompts)
        # embedding_prompt = torch.cat([self.teacher_embedding_prompt, self.student_embedding_prompt], dim=0)
        # text_features = self.text_encoder(embedding_prompt,tokenized_prompts)
        return text_features


    
class MHSA(nn.Module):
    def __init__(self, input_dim=512, heads=8):
        super().__init__()
        # self.ln = nn.LayerNorm(input_dim)
        self.mhsa = nn.MultiheadAttention(embed_dim=input_dim, num_heads=heads, batch_first=True)
    
    def forward(self, x):
        # x = self.ln(x)
        x, _ = self.mhsa(x, x, x, need_weights=False)
        return x
 

 

class CLIP_Detector(nn.Module):
    def __init__(self, model):
        super(CLIP_Detector,self).__init__()
        self.text_encoder = Prompt_processor(model,
            [
            # Best
            # "The face image is forged by DeepFake.",
            # "The face image is forged by FaceSwap.",
            # "The face image is forged by Face2Face.",
            # "The face image is forged by NeuralTexture.",
            # "The face image is forged by Unknown method.",
            
            # Best
            "The face image is forged by DeepFake. In this image, there are several potential signs of forgery that suggest it might be created using deepfake technology:\n\n1. **Skin Texture and Lighting**: The skin appears unusually smooth with an even lighting across the face, which can be a sign of digital manipulation.\n2. **Facial Features**: The eyes seem slightly unnatural in shape or position, possibly indicating digital editing.\n3. **Background Integration**: The background seems to blend poorly with the subject's head, suggesting that the image was edited together from different sources.",
            "The face image is forged by FaceSwap. In this image, there are several potential clues that suggest it might be a forgery:\n\n1. **Skin Texture and Lighting**: The skin texture appears unusually smooth in certain areas, which can be an indicator of digital manipulation.\n2. **Eyelids and Eyebrows**: The eyelids seem slightly unnatural, possibly indicating that they were digitally altered or added.\n3. **Lips and Teeth**: The lips appear to have a slight sheen or reflection that might not be natural, suggesting digital enhancement.\n4. **Overall Consistency**: There is a noticeable lack of consistency between different facial features, such as the eyes, nose, and mouth, which can be a sign of digital manipulation.",
            "The face image is forged by Face2Face. The image appears to have several signs of forgery: 1. **Skin Texture and Lighting**: The skin texture looks unnatural, with an overly smooth appearance that is not typical of human skin. Additionally, the lighting seems inconsistent, which can be a sign of digital manipulation.\n\n2. **Facial Features**: The facial features appear exaggerated or distorted in some areas, particularly around the eyes and mouth, which might indicate digital editing.\n\n3. **Background Integration**: The background does not seamlessly integrate with the subject's skin tone and texture, suggesting possible digital compositing issues.",
            "The face image is forged by NeuralTexture. In this image, there are several potential clues that might indicate it has been forged using NeuralTextures or similar techniques:\n\n1. **Skin Texture**: The skin appears unusually smooth with no visible pores or natural imperfections.\n2. **Lighting and Shadows**: The lighting seems overly uniform, lacking the subtle variations typically seen in natural photographs.\n3. **Facial Features**: The edges of facial features (eyes, nose, mouth) appear slightly unnatural, possibly indicating digital manipulation.\n4. **Background Integration**: The background does not seamlessly integrate with the subject's skin tone and texture.\n\nThese elements suggest that the image may have been digitally altered or created rather than being a photograph of an actual person.",
            "The face image is forged by Unknown method. X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X",
            ]).cuda()
        
        set_requires_grad(self.text_encoder.text_encoder, False)
        self.image_encoder = Image_Encoder(model)
        
        self.decoder = nn.Sequential(*[
            nn.TransformerEncoderLayer(768, 12, 768*4, 0, F.gelu, 1e-5, True, True),
            nn.TransformerEncoderLayer(768, 12, 768*4, 0, F.gelu, 1e-5, True, True),
        ])
        

        self.type_linear = nn.Linear(512, 768)
        self.head1 = nn.Linear(768, 1)
        self.head2 = nn.Linear(768, 1)
        self.head3 = nn.Linear(768, 1)
        self.head4 = nn.Linear(768, 1)

        # self.head1 = nn.Linear(768, 4)
        # self.head2 = nn.Linear(768, 4)
        # self.head3 = nn.Linear(768, 4)
        # self.head4 = nn.Linear(768, 4)
        
        self.head = nn.Linear(512, 1)

        self.img2tex = nn.Linear(768, 512)
        self.q_former = QFormer()
        self.fi_attn_encoder = FI_Attn_Encoder(model)
        set_requires_grad(self.fi_attn_encoder, False)
        
        
    def init_text_embed_with_pretrainweight(self, path):
        # self.load_state_dict(torch.load(path)['model'])
        self.load_state_dict(torch.load(path))
        with torch.no_grad():
            all_text_feature = self.text_encoder()
            self.type_feat = all_text_feature[:5]
            
            
    def forward(self, x):
        all_text_feature = self.text_encoder()
        # for i in range(all_text_feature.shape[0]):
        #     sim_list = []
        #     cur_tex = all_text_feature[i]/all_text_feature[i].norm(dim=-1, keepdim=True)
        #     for j in range(all_text_feature.shape[0]):
        #         # if i != j:
        #         sim = torch.dot(cur_tex,\
        #                 all_text_feature[j]/all_text_feature[j].norm(dim=-1, keepdim=True))
        #         sim_list.append(np.float16(sim.detach().cpu().numpy()))
        #     print(sim_list)
        # exit(0)
        forgery_type_ori = all_text_feature[:5]
        image_feature, tokens = self.image_encoder(x, forgery_type_ori)


        # ------------ Specific part ---------------- #
        types = tokens[:,196:,:] # torch.Size([2, 768]) "The forgery type for this fake image is Deepfake",
        type1 = types[:, 0, :]
        type2 = types[:, 1, :]
        type3 = types[:, 2, :]
        type4 = types[:, 3, :]

        cls1 = self.head1(type1)
        cls2 = self.head2(type2)
        cls3 = self.head3(type3)
        cls4 = self.head4(type4)
        # cls5 = self.head5(type5)

        # ------------ Forgery Intensity part ---------------- # 
        forgery_type_visual = self.type_linear(all_text_feature[:5])

        patch_feat = tokens[:,:196,:]
        fi_feat = self.decoder(patch_feat)
        # fi_feat = patch_feat
        
        forgery_type_visual = forgery_type_visual / forgery_type_visual.norm(dim=-1, keepdim=True)
        fi_feat = fi_feat[:,:196,:]
        fi_feat = fi_feat / fi_feat.norm(dim=-1, keepdim=True)
        fi_feat = fi_feat @ (forgery_type_visual.T)
        fi = self.fi_unpatchify(fi_feat)
        fi = fi.sum(1, keepdim=True)
        

        fi_tex = self.img2tex(types)
        fi_w = self.q_former(fi_tex)
        fi_w = self.fi_attn_encoder(fi_w)
        

        # ------------ Similarity part ---------------- # 
        probs = self.head(image_feature * fi_w)

        self.type_feat = all_text_feature[:5]
        
        return probs, cls1, cls2, cls3, cls4, fi # , fi_probs
        # return probs, cls1, cls2, cls3, cls4, None
    
    @torch.no_grad()
    def test_forward(self, x):
        type = self.type_feat
        image_feature, tokens = self.image_encoder(x, type)
        
        fi_tex = self.img2tex(tokens[:,196:,:])
        fi_w = self.q_former(fi_tex)
        fi_w = self.fi_attn_encoder(fi_w)

        probs = self.head(image_feature * fi_w)
        # probs = self.head(image_feature)

        return probs
  
    
    def fi_unpatchify(self, x, c=5):
        """
        x: (N, T, 4)
        imgs: (N, H, W, 4)
        """
        p = 1
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        # c = self.out_channels
        c = 3
        # p = self.x_embedder.patch_size[0]
        p = 16
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    

class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, gamma=2, reduction='mean'):
        """
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        # self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        class_num = pred.shape[-1]
        bz = pred.shape[0]
        alpha = torch.bincount(target, minlength=class_num)
        alpha = 1. - alpha / bz
        
        alpha = alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        # focal_loss = 1. * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss

class FocalLoss_BCE(nn.Module):
    def __init__(self, gamma=2, alpha=[0.25, 0.75]):
        super(FocalLoss_BCE, self).__init__()
        self.gamma = gamma
        self.alpha = torch.tensor(alpha)
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        
        alpha = self.alpha.to(inputs.device)
        alpha = alpha[targets[:, 0].long()]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        focal_loss = alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()
 
    
if __name__ == '__main__':
    
    from CLIP import clip
    clip_model, preprocess = clip.load("ViT-B/16", \
    device=torch.device("cpu"), download_root="/home/liu/fcb1/clip_model")#ViT-B/16
    length_prompt = 16
    x = torch.randn([16, 3, 224, 224]).cuda()
    model = CLIP_Detector(clip_model).cuda()
    # for name, para in model.named_parameters():
    #     print(name)
    probs, cls1, cls2, cls3, cls4 ,fi_feat= model(x)
    _ = model.test_forward(x)
    print("ok")

    


        


    