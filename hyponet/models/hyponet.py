import torch

import torch.nn as nn

import pywt

import numpy as np

import math

from .xresnet1d import xresnet1d50





def create_head1d(

    nf: int,         

    nc: int,         

    act="relu",                     

    concat_pooling=True            

):

    



                                                      

    lin_ftrs = [2 * nf , nc]

          

    actn = nn.ReLU(inplace=True) if act == "relu" else nn.ELU(inplace=True)

                    

    layers = []

                                                          

    if concat_pooling:

        layers.append(nn.AdaptiveAvgPool1d(1))

                    

    layers.append(nn.Flatten())

               

    return nn.Sequential(*layers)



class PositionalEncoding:

    @staticmethod

    def get_positional_encoding(pe_type, num_patches, d_model, learn_pe=True):

        

        if pe_type == 'zeros':

                        

            pe = torch.zeros((num_patches, d_model))

            nn.init.uniform_(pe, -0.02, 0.02)



        elif pe_type == 'sin_cos':

                                       

            pe = torch.zeros(num_patches, d_model)

            position = torch.arange(0, num_patches).unsqueeze(1)

            div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

            pe[:, 0::2] = torch.sin(position * div_term)

            pe[:, 1::2] = torch.cos(position * div_term)

            pe = pe - pe.mean()

            pe = pe / (pe.std() * 10)



        elif pe_type == 'lin1d':

                      

            pe = (2 * (torch.linspace(0, 1, num_patches).reshape(-1, 1)) - 1)

            pe = pe - pe.mean()

            pe = pe / (pe.std() * 10)



        elif pe_type == 'exp1d':

                      

            pe = (2 * (torch.linspace(0, 1, num_patches).reshape(-1, 1) ** 0.5) - 1)

            pe = pe - pe.mean()

            pe = pe / (pe.std() * 10)



        elif pe_type == 'lin2d':

                      

            pe = 2 * (torch.linspace(0, 1, num_patches).reshape(-1, 1)) * (

                torch.linspace(0, 1, d_model).reshape(1, -1)) - 1

            pe = pe - pe.mean()

            pe = pe / (pe.std() * 10)



        elif pe_type == 'exp2d':

                      

            pe = 2 * (torch.linspace(0, 1, num_patches).reshape(-1, 1) ** 0.5) * (

                        torch.linspace(0, 1, d_model).reshape(1, -1) ** 0.5) - 1

            pe = pe - pe.mean()

            pe = pe / (pe.std() * 10)



        else:

            raise ValueError(f"Unsupported positional encoding type: {pe_type}")



        return nn.Parameter(pe, requires_grad=learn_pe)





class SignalPatcherWithPE(nn.Module):

    def __init__(self, patch_len, stride, d_model, pe_type='sin_cos', padding_patch='end', learn_pe=True):

        super().__init__()

        self.patch_len = patch_len

        self.stride = stride

        self.d_model = d_model



                

        if padding_patch == 'end':

            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))

        self.padding_patch = padding_patch



                                    

        self.projection = nn.Linear(patch_len, d_model)



                   

                                

        self.num_patches = None                      



                

        self.pe_type = pe_type

        self.learn_pe = learn_pe

        self.W_pos = None                 



        self.dropout = nn.Dropout(0.1)



    def forward(self, x):

                                               



                    

        if self.padding_patch == 'end':

            x = self.padding_patch_layer(x)



                                

        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

                                                           



                

        x = x.permute(0, 1, 3, 2)

                                                           



                       

        if self.W_pos is None:

            self.num_patches = x.size(-1)

            self.W_pos = PositionalEncoding.get_positional_encoding(

                self.pe_type, self.num_patches, self.d_model, self.learn_pe

            ).to(x.device)



                         

        B, C, P, L = x.shape

        x = x.permute(0, 1, 3, 2)                

        x = self.projection(x)                      



                   

        x = x + self.W_pos



                    

        x = self.dropout(x)



        return x







class CrossAttention(nn.Module):

    def __init__(self, d_model, n_heads=8, dropout=0.1):

        super().__init__()

        self.n_heads = n_heads

        self.d_k = d_model // n_heads

        self.scaling = self.d_k ** -0.5



                         

        self.primary_q = nn.Linear(d_model, d_model)

        self.primary_k = nn.Linear(d_model, d_model)

        self.primary_v = nn.Linear(d_model, d_model)



                    

        self.auxiliary_k = nn.Linear(d_model, d_model)

        self.auxiliary_v = nn.Linear(d_model, d_model)



        self.out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)



    def forward(self, primary_signal, auxiliary_signals):

        

        B, L, D = primary_signal.shape



               

        Q = self.primary_q(primary_signal).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        K_primary = self.primary_k(primary_signal).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        V_primary = self.primary_v(primary_signal).view(B, L, self.n_heads, self.d_k).transpose(1, 2)



                

        K_aux = self.auxiliary_k(auxiliary_signals).view(B, 2, L, self.n_heads, self.d_k).transpose(2, 3)

        V_aux = self.auxiliary_v(auxiliary_signals).view(B, 2, L, self.n_heads, self.d_k).transpose(2, 3)



                        

        K = torch.cat([K_primary.unsqueeze(1), K_aux], dim=1)                           

        V = torch.cat([V_primary.unsqueeze(1), V_aux], dim=1)                           



                 

        K = K.mean(dim=1)            

        V = V.mean(dim=1)            



        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scaling

        attn = torch.softmax(attn, dim=-1)

        attn = self.dropout(attn)



              

        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, L, D)

        out = self.out(out)



        return out





import torch

import torch.nn as nn



                            

class PatchCrossBlock(nn.Module):

    def __init__(self, patcher, cross_attention):

        super().__init__()

        self.patcher = patcher

        self.cross_attention = cross_attention



    def forward(self, x):

                        

        patches = self.patcher(x)                                    

                       

        primary_signal = patches[:, 0]                                   

        auxiliary_signals = patches[:, 1:]                                    

                  

        attended = self.cross_attention(primary_signal, auxiliary_signals)

              

        attended = attended + primary_signal

        return attended





import torch

import torch.nn as nn

import pywt

import numpy as np





                                 

def wavelet_packet_transform(x, wavelet, wp_level=3):

    

    batch_size, channels, seq_len = x.shape

    device = x.device



             

    features = []



                    

    for b in range(batch_size):

        channel_features = []

        for c in range(channels):

                    

            signal = x[b, c].detach().cpu().numpy()                   



                     

            wp = pywt.WaveletPacket(signal, wavelet=wavelet, mode='symmetric', maxlevel=wp_level)



                    

            nodes = [node.path for node in wp.get_level(wp_level, 'natural')]



                       

            coeffs = np.array([wp[node].data for node in nodes])



                                 

            min_len = min(len(coef) for coef in coeffs)

            coeffs = np.array([coef[:min_len] for coef in coeffs])



                                   

            channel_features.append(torch.from_numpy(coeffs).float())



                                        

        channel_features = torch.cat(channel_features, dim=0)

        features.append(channel_features.unsqueeze(0))



                                           

    features = torch.cat(features, dim=0).to(device)

    return features





def infer_wavelet_packet_shape(seq_len, input_channels, wavelet, wp_level):

    

    dummy_signal = np.zeros(seq_len, dtype=np.float32)

    wp = pywt.WaveletPacket(dummy_signal, wavelet=wavelet, mode='symmetric', maxlevel=wp_level)

    nodes = [node.path for node in wp.get_level(wp_level, 'natural')]

    coeff_len = min(len(wp[node].data) for node in nodes)

    return input_channels * len(nodes), coeff_len





class TimeSeriesTransformer(nn.Module):

    def __init__(self,

                 input_channels=24,                        

                 seq_len=756,                 

                 d_model=64,                           

                 nhead=8,            

                 num_layers=4,                          

                 dim_feedforward=512,                     

                 dropout=0.1,             

                 patch_size=6,          

                 use_sin_pos_enc=True                             

                 ):

        super(TimeSeriesTransformer, self).__init__()

        self.patch_size = patch_size



                                                                         

                                                 

        self.patch_embed = nn.Conv1d(in_channels=input_channels,

                                     out_channels=d_model,

                                     kernel_size=patch_size,

                                     stride=patch_size)

                        

                                                             

        self.out_seq_len = math.ceil(seq_len / patch_size)                        



                                 

        if use_sin_pos_enc:

                      

            position = torch.arange(0, self.out_seq_len).unsqueeze(1).float()                    

            div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

            pos_embed = torch.zeros(self.out_seq_len, d_model)

            pos_embed[:, 0::2] = torch.sin(position * div_term)             

            pos_embed[:, 1::2] = torch.cos(position * div_term)             

            pos_embed = pos_embed.unsqueeze(0)                             

            self.register_buffer("pos_encoding", pos_embed)

        else:

                      

            self.pos_encoding = nn.Parameter(torch.zeros(1, self.out_seq_len, d_model))



                                              

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model,

                                                   nhead=nhead,

                                                   dim_feedforward=dim_feedforward,

                                                   dropout=dropout,

                                                   batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

                                      

        self.norm = nn.LayerNorm(d_model)



    def forward(self, x):

        

                                         

        L = x.shape[-1]

        if L % self.patch_size != 0:

            pad_len = self.patch_size - (L % self.patch_size)

            x = F.pad(x, (0, pad_len))                     



                                       

                                                

                                                   

        x_embed = self.patch_embed(x)



                                                                        

        x_embed = x_embed.permute(0, 2, 1)



                   

        if hasattr(self, "pos_encoding"):

                                                

            pos_enc = self.pos_encoding[:, :x_embed.size(1), :]

            x_embed = x_embed + pos_enc



                            

                                                                   

        x_embed = F.dropout(x_embed, p=0.1, training=self.training)



                                     

                                                                           

        x_transformed = self.transformer_encoder(x_embed)



                      

        x_transformed = self.norm(x_transformed)



                                                                               

        x_out = x_transformed.permute(0, 2, 1)

        return x_out





class SignalProcessor(nn.Module):

    def __init__(self, seq_len, patch_len, stride, d_model, n_heads=8, dropout=0.,

                 num_classes=1, wp_level=3, input_channels=3):

        super().__init__()

                

        self.input_norm = nn.BatchNorm1d(input_channels)              

                     

        self.patcher = SignalPatcherWithPE(

            patch_len=patch_len,

            stride=stride,

            d_model=d_model,

            pe_type='sin_cos',

            padding_patch='end'

        )



                

        self.cross_attention = CrossAttention(

            d_model=d_model,

            n_heads=n_heads,

            dropout=dropout

        )



                                    

        self.patch_cross = PatchCrossBlock(self.patcher, self.cross_attention)

        self.wp_level = wp_level           

                                       

        self.res_wavelet = 'db4'

                                    

        self.trend_wavelet = 'db2'

        res_wp_channels, res_wp_len = infer_wavelet_packet_shape(

            seq_len, input_channels, self.res_wavelet, self.wp_level

        )

        trend_wp_channels, trend_wp_len = infer_wavelet_packet_shape(

            seq_len, input_channels, self.trend_wavelet, self.wp_level

        )

                   

        self.wp_res_feature = TimeSeriesTransformer(input_channels=res_wp_channels, seq_len=res_wp_len, d_model=64, nhead=8, num_layers=4,dim_feedforward=512, dropout=0.1, patch_size=6)

                   

        self.wp_trend_feature = TimeSeriesTransformer(input_channels=trend_wp_channels, seq_len=trend_wp_len, d_model=64, nhead=8, num_layers=4,dim_feedforward=512, dropout=0.1, patch_size=6)



                   

        self.num_patches = (seq_len - patch_len) // stride + 1

        if patch_len == 'end':

            self.num_patches += 1



               

        self.resnet = xresnet1d50(input_channels=187 , num_classes=1, dropout=0.)



                                                        

        self.mlp = nn.Sequential(

            nn.Linear(81, 128),

            nn.BatchNorm1d(128),

            nn.ReLU(),

        )



        self.head = create_head1d(

            32,

            nc=num_classes,

            act="relu",

            concat_pooling=True,

        )



                                                                         

                                                

        self.classifier = nn.Linear(384, num_classes)



    def forward(self, x, ml_feature=None):

        

                

        x = self.input_norm(x)

                                                                           



                

        wavelet_features = None



                         

        res_wp_features = wavelet_packet_transform(x, self.res_wavelet, wp_level=self.wp_level)

        trend_wp_features = wavelet_packet_transform(x, self.trend_wavelet, wp_level=self.wp_level)



                                              

        res_features = self.wp_res_feature(res_wp_features)

        trend_features = self.wp_trend_feature(trend_wp_features)



              

        res_features = res_features.max(dim=-1)[0]

        trend_features = trend_features.max(dim=-1)[0]

              

        wavelet_features = torch.cat((res_features, trend_features), dim=1)

                                                           

        attended = self.patch_cross(x)

              

        net_feature= self.resnet(attended)

        output = self.head(net_feature)

        output = torch.cat((output,wavelet_features),dim=1)

        if self.classifier.in_features == output.shape[1] + 128:

            if ml_feature is None:

                raise ValueError("ml_feature is required for this SignalProcessor checkpoint.")

            ml_features = self.mlp(ml_feature)

            output = torch.cat((output, ml_features), dim=1)

        elif self.classifier.in_features != output.shape[1]:

            raise ValueError(

                f"Unexpected fusion dimension: classifier expects {self.classifier.in_features}, "

                f"but available features have {output.shape[1]} dimensions."

            )

            

        output = self.classifier(output)



        return output
