from CLIP import clip
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
        

query_len = 16 # best
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
        

        if isinstance(initials,list):
            text = clip.tokenize(initials)
            self.tokenized_text = text
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
            self.embedding_prompt=nn.Parameter(new_state_dict['embedding_prompt'])
            self.embedding_prompt.requires_grad = True
        else:
            print([" ".join(["X"]*\
                16)," ".join(["X"]*16)])
            self.embedding_prompt=torch.nn.init.xavier_normal_(nn.Parameter(model.token_embedding([" ".join(["X"]*\
                16)," ".join(["X"]*16)]).requires_grad_()))
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
            "The face image is forged by DeepFake.",
            "The face image is forged by FaceSwap.",
            "The face image is forged by Face2Face.",
            "The face image is forged by NeuralTexture.",
            "The face image is forged by Unknown method.",
            
            ])
        
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
        
        self.head = nn.Linear(512, 1)

        self.img2tex = nn.Linear(768, 512)
        self.q_former = QFormer()
        self.fi_attn_encoder = FI_Attn_Encoder(model)
        set_requires_grad(self.fi_attn_encoder, False)
        
        
    def init_text_embed_with_pretrainweight(self, path):
        # self.load_state_dict(torch.load(path)['model'])
        self.load_state_dict(torch.load(path, map_location="cpu"))
        with torch.no_grad():
            all_text_feature = self.text_encoder()
            self.type_feat = all_text_feature[:5]
        del self.text_encoder
        del self.type_linear 
        del self.head1
        del self.head2
        del self.head3
        del self.head4
        del self.decoder
            
    @torch.no_grad()
    def forward(self, x):
        type = self.type_feat
        image_feature, tokens = self.image_encoder(x, type)
        fi_tex = self.img2tex(tokens[:,196:,:])
        fi_w = self.q_former(fi_tex)
        fi_w = self.fi_attn_encoder(fi_w)
        probs = self.head(image_feature * fi_w)
        probs = torch.sigmoid(probs)
        return probs
 
    
if __name__ == '__main__':
    print("PyTorch Version:",torch.__version__)
    from CLIP import clip
    clip_model, preprocess = clip.load("ViT-B/16",
                            device=torch.device("cpu"), 
                            download_root="./") # ViT-B/16

    import cv2
    imgpath = "imgs./real.png"
    image = cv2.imread(imgpath)
    #cv2 read as bgr format
    image = image[..., ::-1]
    #change to rgb format
    image = cv2.resize(image, (224, 224)) / 255.
    #resize to mobile_net tensor size
    image = image - (0.48145466, 0.4578275, 0.40821073)
    image = image / (0.26862954, 0.26130258, 0.27577711)
    x = torch.tensor(image).unsqueeze(0).permute(0,3,1,2).float()
    model = CLIP_Detector(clip_model)
    _ = model.init_text_embed_with_pretrainweight("./FAPL.pth")
    from time import time
    st = time()
    probs = model(x)
    ed = time()
    print("inference time: {:.3f} s".format((ed - st)))
    print(probs)
    
    # torch.onnx.export(model, 
    #                   (x,), 
    #                   './mnn_model/FAPL_detector.onnx',
    #                   opset_version=18,
    #                   input_names=["img"],
    #                   output_names=["prob"],
    #                   dynamo=True,
    #                   report=True
    #                   )
    
    print("ok")


    


        


    