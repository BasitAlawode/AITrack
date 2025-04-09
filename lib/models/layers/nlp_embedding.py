import torch
import torch.nn as nn
from torch.nn import functional as F
#from . import alpha_clip
from alpha_clip import load as load_alpha_clip
from PIL import Image
import numpy as np
from torchvision import transforms

mask_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Resize((224, 224)), # change to (336,336) when using ViT-L/14@336px
    transforms.Normalize(0.5, 0.26)
])

image_transform = transforms.ToPILImage()

class NLP_embedding(nn.Module):

    def __init__(self,training=False):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.clip_model, self.preprocess = clip.load('/home/yuqing/test2/CiteTracker/lib/models/layers/clip/ViT-B-32.pt', self.device)
        self.clip_model, self.preprocess = \
            load_alpha_clip("ViT-L/14", 
                            alpha_vision_ckpt_pth= \
                                "./alpha_clip_ckpt/clip_l14_grit20m_fultune_2xe.pth", 
                            device=self.device)  # change to your own ckpt path
        
        self.training = training
        
    def forward(self, template, template_mask, search, search_mask):
        # image-encoder
        if self.training is True or type(template)==torch.Tensor:
            batch_size = len(template)
        else:
            batch_size = 1
        
        #feat_des = torch.Tensor().cuda()
        tem_feat_batch = torch.Tensor().cuda()
        sea_feat_batch = torch.Tensor().cuda()
        
        for i in range(batch_size):
            if self.training:
                tem_image, sea_image = template[i], search[i]
                tem_mask, sea_mask = template_mask[i], search_mask[i]

                tem_mask = tem_mask.cpu().numpy().astype(np.uint8)
                sea_mask = sea_mask.cpu().numpy().astype(np.uint8)
            else:
                tem_image, sea_image = template, search
                tem_mask, sea_mask = template_mask, search_mask
            
            tem_image = image_transform(tem_image.squeeze())
            sea_image = image_transform(sea_image.squeeze())
            
            #if len(tem_mask.shape) == 2: tem_binary_mask = (tem_mask == 255)
            #if len(tem_mask.shape) == 3: tem_binary_mask = (tem_mask[:, :, 0] == 255)
            #tem_alpha = mask_transform((tem_binary_mask * 255).astype(np.uint8))
            tem_alpha = mask_transform(tem_mask * 255.0)
            tem_alpha = tem_alpha.half().cuda().unsqueeze(dim=0)
            tem_image = self.preprocess(tem_image).unsqueeze(0).half().to(self.device)
            
            #if len(sea_mask.shape) == 2: sea_binary_mask = (sea_mask == 255)
            #if len(sea_mask.shape) == 3: sea_binary_mask = (sea_mask[:, :, 0] == 255)
            #sea_alpha = mask_transform((sea_binary_mask * 255).astype(np.uint8))
            sea_alpha = mask_transform(sea_mask * 255.0)
            sea_alpha = sea_alpha.half().cuda().unsqueeze(dim=0)
            sea_image = self.preprocess(sea_image).unsqueeze(0).half().to(self.device)
                   
            with torch.no_grad():
                tem_feat = self.clip_model.visual(tem_image, tem_alpha)
                sea_feat = self.clip_model.visual(sea_image, sea_alpha)
                
                #tem_feat = self.clip_model.encode_image(tem_image, tem_alpha)
                #sea_feat = self.clip_model.encode_image(sea_image, sea_alpha)
                
                # normalize
                tem_feat = tem_feat / tem_feat.norm(dim=-1, keepdim=True)
                sea_feat = sea_feat / sea_feat.norm(dim=-1, keepdim=True)
                
                # Convolve both features to obtain the similarity
                #tem_feat_view = tem_feat.unsqueeze(-1).unsqueeze(-1)
                #sea_feat_view = sea_feat.unsqueeze(-1).unsqueeze(-1)
                #curr_feat_des = con_att(tem_feat_view, sea_feat_view)
                #curr_feat_des = sea_feat_view

            #feat_des = torch.cat([feat_des, curr_feat_des], dim=0)
            tem_feat_batch = torch.cat([tem_feat_batch, tem_feat], dim=0)
            sea_feat_batch = torch.cat([sea_feat_batch, sea_feat], dim=0)
  
        #return feat_des.resize(batch_size, 1, 768, 1, 1)
        return tem_feat_batch, sea_feat_batch


def con_att(weight ,input):
    out = F.conv2d(input, weight, stride=1, padding=0)
    out = out.repeat(1, 768, 1, 1)
    return out
