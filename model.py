from torch import nn
import torch
import torch.nn.functional as F
import timm
import math


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class ConvNeXtV2Base(nn.Module):
    def __init__(self, num_classes=200, pretrained=True, freeze=True):
        super().__init__()
        model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k', pretrained=pretrained)

        if freeze:
            model.stem.requires_grad = False
            model.stages.requires_grad = False

        model.head.fc = nn.Linear(
            in_features=1024,
            out_features=num_classes,
            bias=True
        )

        features = nn.Sequential()
        features.add_module(str(len(features)), model.stem)
        for stage in model.stages:
            features.add_module(str(len(features)), stage)
        features.add_module(str(len(features)), model.head)

        self.features = features

    def forward(self, x):
        return self.features(x)
    

class TransConvNeXtV2Base(nn.Module):
    def __init__(self, num_classes=200, pretrained=True, freeze=True):
        super().__init__()
        
        if pretrained:
            model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k', pretrained=True)
            self.features = nn.Sequential()
            self.features.add_module(str(len(self.features)), model.stem)

            for stage in model.stages[:-1]:
                self.features.add_module(str(len(self.features)), stage)                                   

            # Freeze model weights
            if freeze:
                #freeze layers
                for param in self.features.parameters():
                    param.requires_grad = False

        else:
            self.features = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k', pretrained=False)

        self.pos_encoder = PositionalEncoding(512, dropout=0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,  # This should match the feature size of ConvNeXt's last layer
            nhead=8,      # Number of attention heads
            dim_feedforward=1024,
            dropout=0.1,
            activation='gelu',
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.classifier = torch.nn.Sequential(
            nn.Linear(
                in_features=512,
                out_features=num_classes,
                bias=True
            )
        )
        
    def forward(self, x):
        x = self.features(x)

        b, c, h, w = x.size()
        x = x.view(b, c, h * w).permute(2, 0, 1)

        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=0)
        
        x = self.classifier(x)

        return x



def get_convnextv2_base(out_feat, pretrained=False, freeze=False):
    return ConvNeXtV2Base(out_feat, pretrained, freeze)

def get_transconvnextv2_base(out_feat, pretrained=False, freeze=False):
    return TransConvNeXtV2Base(out_feat, pretrained, freeze)
    

def get_model(model_name, out_feat, pretrained=False, freeze=False):
    if model_name =="ConvNeXtV2Base":
        return get_convnextv2_base(out_feat, pretrained, freeze)
    elif model_name =="TransConvNeXtV2Base":
        return get_transconvnextv2_base(out_feat, pretrained, freeze)
    else:
        raise Exception("Model not implemented")