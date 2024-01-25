from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights

from torchvision.models import densenet121
from torchvision.models import DenseNet121_Weights

from torchvision.models import convnext_tiny
from torchvision.models import ConvNeXt_Tiny_Weights

from torchvision.models import convnext_base
from torchvision.models import ConvNeXt_Base_Weights

from torch import nn
import torch
import torch.nn.functional as F
import math

from utils import Permute


def _is_contiguous(tensor):
        # jit is oh so lovely :/
        # if torch.jit.is_tracing():
        #     return True
        if torch.jit.is_scripting():
            return tensor.is_contiguous()
        else:
            return tensor.is_contiguous(memory_format=torch.contiguous_format)


class LayerNorm2d(nn.LayerNorm):
        r""" LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
        """

        def __init__(self, normalized_shape, eps=1e-6):
            super().__init__(normalized_shape, eps=eps)

        def forward(self, x):
            if _is_contiguous(x):
                return F.layer_norm(
                    x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
            else:
                s, u = torch.var_mean(x, dim=1, keepdim=True)
                x = (x - u) * torch.rsqrt(s + self.eps)
                x = x * self.weight[:, None, None] + self.bias[:, None, None]
                
                return x


class ConvNextBlock(nn.Module):
            def __init__(
                self,
                dim,
            ):
                super().__init__()

                self.block = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
                    Permute([0, 2, 3, 1]),
                    nn.LayerNorm(dim),
                    nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
                    nn.GELU(),
                    nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
                    Permute([0, 3, 1, 2]),
                )

            def forward(self, input):
                result = self.block(input)
                result += input
                return result
            

def get_resnet50(out_feat, pretrained=False, freeze=False):
    if pretrained:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

        # Freeze model weights
        if freeze:
            for param in model.parameters():
                param.requires_grad = False

        for param in model.layer4.parameters():
            param.requires_grad = True

        for i, (name, layer) in enumerate(model.layer4.named_modules()):
            if isinstance(layer, torch.nn.Conv2d):
                layer.reset_parameters()

    else:
        model = resnet50()
    

    # Add on fully connected layers for the output of our model

    # model.avgpool = torch.nn.Identity()

    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(
            in_features=2048,
            out_features=out_feat,
            bias=True
        )
    )
    
    return model


def get_densenet121(out_feat, pretrained=False, freeze=False, nextify=False):
    if pretrained:
        model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

        # Freeze all parameters
        if freeze:
            for param in model.parameters():
                param.requires_grad = False

        # Unfreeze parameters of last dense block
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True

        # Implement linear probing for last dense layers and classifier
        for param in list(model.features.denseblock4.parameters()) + list(model.classifier.parameters()):
            param.requires_grad = True

    else:
        model = densenet121()

    # Replace classifier
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(
            in_features=1024,  # densenet121 has 1024 output features
            out_features=out_feat,
            bias=True
        )
    )
    
    return model


def get_convnext_tiny(out_feat, pretrained=False, freeze=False):
    if pretrained:
        model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)

        # Freeze model weights
        if freeze:
            for i in range(6):
                for param in model.features[i].parameters():
                    param.requires_grad = False

        for i in range(6, 8):
            for name, layer in model.features[i].named_modules():
                if isinstance(layer, torch.nn.Conv2d):
                    # print(i, name, layer)
                    layer.reset_parameters()

    else:
        model = convnext_tiny()

    model.classifier = torch.nn.Sequential(
        LayerNorm2d(768, eps=1e-06),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(
            in_features=768,
            out_features=out_feat,
            bias=True
        )
    )

    return model


def get_densenext(out_feat, pretrained=False, freeze=False):
    if pretrained:
        model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Implement linear probing for last dense layers and classifier
        for param in list(model.features.denseblock4.parameters()) + list(model.classifier.parameters()):
            param.requires_grad = True

    else:
        model = densenet121()

    # Replace classifier
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2),
        torch.nn.Linear(
            in_features=1024,  # densenet121 has 1024 output features
            out_features=out_feat,
            bias=True
        )
    )

    model.features.denseblock4 = nn.Sequential(
                                            ConvNextBlock(512),
                                            nn.Conv2d(512, 1024, kernel_size=1, stride=2, bias=False),
                                            Permute([0, 2, 3, 1]),
                                            nn.LayerNorm(1024),
                                            nn.GELU(),
                                            Permute([0, 3, 1, 2]),
                                            ConvNextBlock(1024),
                                            Permute([0, 2, 3, 1]),
                                            #nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False),
                                            nn.LayerNorm(1024),
                                            nn.GELU(),
                                            Permute([0, 3, 1, 2])

            )
    
    return model


def get_convnext_base(out_feat, pretrained=False, freeze=False):
    if pretrained:
        model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)

        # Freeze model weights
        if freeze:
            for i in range(6):
                for param in model.features[i].parameters():
                    param.requires_grad = False

        for i in range(6, 8):
            for name, layer in model.features[i].named_modules():
                if isinstance(layer, torch.nn.Conv2d):
                    # print(i, name, layer)
                    layer.reset_parameters()

    else:
        model = convnext_base()

    model.classifier = torch.nn.Sequential(
        LayerNorm2d(1024, eps=1e-06),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(
            in_features=1024,
            out_features=out_feat,
            bias=True
        )
    )

    return model
    

def get_model(model_name, out_feat, pretrained=False, freeze=False):
    if model_name == "ResNet50":
        return get_resnet50(out_feat, pretrained, freeze)
    elif model_name == "DenseNet121":
        return get_densenet121(out_feat, pretrained, freeze)
    elif model_name == "DenseNeXt":
        return get_densenext(out_feat, pretrained, freeze)
    elif model_name == "ConvNeXtTiny":
        return get_convnext_tiny(out_feat, pretrained, freeze)
    elif model_name == "ConvNeXtBase":
        return get_convnext_base(out_feat, pretrained, freeze)
    else:
        raise Exception("Model not implemented")