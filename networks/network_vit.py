# -*- coding: utf-8 -*-
"""
Created on Fri May 27 21:55:04 2022

@author: kankana.roy
"""

import torch.nn as nn
import torch
from torch import Tensor
import timm
import torch.nn.functional as F
from typing import Optional
from pytorch_pretrained_vit import ViT
from torch.nn.modules.transformer import _get_activation_fn

from collections import OrderedDict
class SupConTresNet(nn.Module):
    def __init__(self, name='tresnet', head='mlp', feat_dim=768,num_classes = 30,use_pretrained = True):
        super(SupConTresNet, self).__init__()
        model_tresnet = timm.create_model('tresnet_l', pretrained=use_pretrained, num_classes = num_classes)
        #print(torch.load('results and models/Tresnet_l_HOF.pth'))
        #model_tresnet.load_state_dict(torch.load('results and models/Tresnet_l_HOF.pth'))
        ########## Removing last couple of layer sto match input of Vit ###################
        model_tresnet = nn.Sequential(*list(model_tresnet.children())[:-1])
        self.encoder = model_tresnet
        self.head =  nn.Sequential(nn.AvgPool2d(12),
                                   nn.Conv2d(2432, 768, 1, stride=1))
    def forward(self, x):
        feat = self.encoder(x)
        feat = self.head(feat)
        #print(feat.size())
        feat = feat.squeeze()
        feat = F.normalize(feat, dim=1)
        #print(feat)
        return feat
            
            
model_dict = {
    'tresnet' :[SupConTresNet(), 768],
}    
class LinearViT(nn.Module):
    """Linear classifier"""
    def __init__(self, name='tresnet', num_classes = 30,use_pretrained = True):
        super(LinearViT, self).__init__()
        model_tresnet, feat_dim = model_dict[name]
        ckpt = torch.load('/content/gdrive/MyDrive/path_models/SupCon_path_tresnet_lr_0.001_decay_0.0001_bsz_80_temp_0.07_trial_1/last.pth', map_location='cpu')
        state_dict = ckpt['model']
        model_tresnet.load_state_dict(state_dict)
        model_tresnet = model_tresnet.encoder
        model_tresnet[0].layer4= nn.Identity()
        #print(state_dict.keys())
        #print(model_tresnet)
        
        vit = ViT('B_16_imagenet1k', pretrained=use_pretrained, num_classes = num_classes, image_size = 384)
        vit.patch_embedding = nn.Sequential(OrderedDict([('tresnet', model_tresnet),('reduction',nn.Conv2d(1216, 768, kernel_size=(1, 1), stride=(1, 1), bias=False))]))
        self.encoder = vit
        
        #print(self.encoder)
    def forward(self, images):
        #x =self.encoder.patch_embedding(images)
        #print(x.size())
        features = self.encoder(images)
        #print(features.size())
        return features
    
class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='tresnet', num_classes = 7, use_pretrained = True):
        super(LinearClassifier, self).__init__()
        model_tresnet, feat_dim = model_dict[name]
        ckpt = torch.load('C:\\Users\\kanroy\\Kankana\\project_hand_over_face\\results and models\\path_models\\SupCon_path_tresnet_lr_0.001_decay_0.0001_bsz_24_temp_0.07_trial_1\\last.pth', map_location='cpu')
        state_dict = ckpt['model']
        model_tresnet.load_state_dict(state_dict)
        # model_tresnet = model_tresnet.encoder
        # model_tresnet[0].layer4= nn.Identity()
        #print(state_dict.keys())
        #print(model_tresnet)
        self.encoder = model_tresnet
        encoder_layer = nn.TransformerEncoderLayer(
                d_model=2432, 
                nhead=4, 
                dim_feedforward=128, 
                dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc =  nn.Sequential(nn.AvgPool2d(12),
                                         nn.Flatten(),
                                         nn.Linear(2432, num_classes))
        
        #print(self.encoder)
    def forward(self, images):
        #x =self.encoder.patch_embedding(images)
        #print(x.size())
        features = self.encoder(images)
        features = self.transformer_encoder(features)
        #print(features.size())
        return self.fc(features)
    
class TransformerClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='tresnet', num_classes = 30,use_pretrained = True):
        super(TransformerClassifier, self).__init__()
        model_tresnet, feat_dim = model_dict[name]
        ckpt = torch.load('C:\\Users\\kanroy\\Kankana\\project_hand_over_face\\results and models\\tresnet__main_contrastive_RAFDB.pth', map_location='cpu')
        state_dict = ckpt['model']
        model_tresnet.load_state_dict(state_dict)
        # model_tresnet = model_tresnet.encoder
        # model_tresnet[0].layer4= nn.Identity()
        #print(state_dict.keys())
        #print(model_tresnet)
        self.encoder = model_tresnet
        self.fc = nn.Linear(feat_dim, num_classes)
        
        #print(self.encoder)
    def forward(self, images):
        #x =self.encoder.patch_embedding(images)
        #print(x.size())
        features = self.encoder(images)
        #print(features.size())
        return self.fc(features)
class PositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim, reduction_ratio=4):
        super().__init__()
        num_channels_reduced = dim // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.pos_embedding = nn.Sequential(OrderedDict([('fc1', nn.Linear(dim, num_channels_reduced, bias=True)),('fc2', nn.Linear(num_channels_reduced, dim, bias=True))]))
    
    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        # print(x.size())
        # print(self.pos_embedding(x).size())
        return x + self.pos_embedding(x)
    
def add_ml_decoder_head(model, num_classes=-1, num_of_groups=-1, decoder_embedding=768, zsl=0):
    if num_classes == -1:
        num_classes = model.num_classes
    num_features = model.num_features
    if hasattr(model, 'global_pool') and hasattr(model, 'fc'):  # resnet50
        model.global_pool = nn.Identity()
        del model.fc
        model.fc = MLDecoder(num_classes=num_classes, initial_num_features=num_features, num_of_groups=num_of_groups,
                             decoder_embedding=decoder_embedding, zsl=zsl)
    elif hasattr(model, 'head'):  # tresnet
        if hasattr(model, 'global_pool'):
            model.global_pool = nn.Identity()
        del model.head
        model.head = MLDecoder(num_classes=num_classes, initial_num_features=num_features, num_of_groups=num_of_groups,
                               decoder_embedding=decoder_embedding, zsl=zsl)
    else:
        print("model is not suited for ml-decoder")
        exit(-1)

    return model

class TransformerDecoderLayerOptimal(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5) -> None:
        super(TransformerDecoderLayerOptimal, self).__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerDecoderLayerOptimal, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


# @torch.jit.script
# class ExtrapClasses(object):
#     def __init__(self, num_queries: int, group_size: int):
#         self.num_queries = num_queries
#         self.group_size = group_size
#
#     def __call__(self, h: torch.Tensor, class_embed_w: torch.Tensor, class_embed_b: torch.Tensor, out_extrap:
#     torch.Tensor):
#         # h = h.unsqueeze(-1).expand(-1, -1, -1, self.group_size)
#         h = h[..., None].repeat(1, 1, 1, self.group_size) # torch.Size([bs, 5, 768, groups])
#         w = class_embed_w.view((self.num_queries, h.shape[2], self.group_size))
#         out = (h * w).sum(dim=2) + class_embed_b
#         out = out.view((h.shape[0], self.group_size * self.num_queries))
#         return out

@torch.jit.script
class GroupFC(object):
    def __init__(self, embed_len_decoder: int):
        self.embed_len_decoder = embed_len_decoder

    def __call__(self, h: torch.Tensor, duplicate_pooling: torch.Tensor, out_extrap: torch.Tensor):
        for i in range(h.shape[1]):
            h_i = h[:, i, :]
            if len(duplicate_pooling.shape)==3:
                w_i = duplicate_pooling[i, :, :]
            else:
                w_i = duplicate_pooling
            out_extrap[:, i, :] = torch.matmul(h_i, w_i)


class MLDecoder(nn.Module):
    def __init__(self, num_classes, num_of_groups=-1, decoder_embedding=768,
                 initial_num_features=2048, zsl=0):
        super(MLDecoder, self).__init__()
        embed_len_decoder = 100 if num_of_groups < 0 else num_of_groups
        if embed_len_decoder > num_classes:
            embed_len_decoder = num_classes

        # switching to 768 initial embeddings
        decoder_embedding = 768 if decoder_embedding < 0 else decoder_embedding
        embed_standart = nn.Linear(initial_num_features, decoder_embedding)

        # non-learnable queries
        if not zsl:
            query_embed = nn.Embedding(embed_len_decoder, decoder_embedding)
            query_embed.requires_grad_(False)
        else:
            query_embed = None

        # decoder
        decoder_dropout = 0.1
        num_layers_decoder = 1
        dim_feedforward = 2048
        layer_decode = TransformerDecoderLayerOptimal(d_model=decoder_embedding,
                                                      dim_feedforward=dim_feedforward, dropout=decoder_dropout)
        self.decoder = nn.TransformerDecoder(layer_decode, num_layers=num_layers_decoder)
        self.decoder.embed_standart = embed_standart
        self.decoder.query_embed = query_embed
        self.zsl = zsl

        if self.zsl:
            if decoder_embedding != 300:
                self.wordvec_proj = nn.Linear(300, decoder_embedding)
            else:
                self.wordvec_proj = nn.Identity()
            self.decoder.duplicate_pooling = torch.nn.Parameter(torch.Tensor(decoder_embedding, 1))
            self.decoder.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(1))
            self.decoder.duplicate_factor = 1
        else:
            # group fully-connected
            self.decoder.num_classes = num_classes
            self.decoder.duplicate_factor = int(num_classes / embed_len_decoder + 0.999)
            self.decoder.duplicate_pooling = torch.nn.Parameter(
                torch.Tensor(embed_len_decoder, decoder_embedding, self.decoder.duplicate_factor))
            self.decoder.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(num_classes))
        torch.nn.init.xavier_normal_(self.decoder.duplicate_pooling)
        torch.nn.init.constant_(self.decoder.duplicate_pooling_bias, 0)
        self.decoder.group_fc = GroupFC(embed_len_decoder)
        self.train_wordvecs = None
        self.test_wordvecs = None

    def forward(self, x):
        if len(x.shape) == 4:  # [bs,2048, 7,7]
            embedding_spatial = x.flatten(2).transpose(1, 2)
        else:  # [bs, 197,468]
            embedding_spatial = x
        embedding_spatial_786 = self.decoder.embed_standart(embedding_spatial)
        embedding_spatial_786 = torch.nn.functional.relu(embedding_spatial_786, inplace=True)

        bs = embedding_spatial_786.shape[0]
        if self.zsl:
            query_embed = torch.nn.functional.relu(self.wordvec_proj(self.decoder.query_embed))
        else:
            query_embed = self.decoder.query_embed.weight
        # tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = query_embed.unsqueeze(1).expand(-1, bs, -1)  # no allocation of memory with expand
        h = self.decoder(tgt, embedding_spatial_786.transpose(0, 1))  # [embed_len_decoder, batch, 768]
        h = h.transpose(0, 1)

        out_extrap = torch.zeros(h.shape[0], h.shape[1], self.decoder.duplicate_factor, device=h.device, dtype=h.dtype)
        self.decoder.group_fc(h, self.decoder.duplicate_pooling, out_extrap)
        if not self.zsl:
            h_out = out_extrap.flatten(1)[:, :self.decoder.num_classes]
        else:
            h_out = out_extrap.flatten(1)
        h_out += self.decoder.duplicate_pooling_bias
        logits = h_out
        return logits
    

