from torch import nn
import timm
from .vit_clip import *

def vit_s_timm(num_classes=0, pretrained=True):

    net = timm.create_model("vit_small_patch16_384", pretrained=pretrained)
    if num_classes==0:
        net.head = nn.Identity()
    else:
        net.head = nn.Linear(net.head.in_features, num_classes)
    return net

def vit_b_timm(num_classes=10,pretrained=True):
    net = timm.create_model("vit_base_patch16_224_clip_laion2b", pretrained=pretrained)
    if num_classes != net.head.out_features:
        net.head = nn.Linear(net.head.in_features, num_classes)
    else:
        print("use raw header",net.head.out_features)
    return net

def vit_b_dino(num_classes=10,pretrained=True):
    net = timm.create_model("vit_base_patch16_224_dino", pretrained=pretrained)
    if num_classes==0:
        net.head = nn.Identity()
    else:
        net.head = nn.Linear(net.norm.normalized_shape[0], num_classes)
        nn.init.xavier_normal_(net.head.weight)
        nn.init.zeros_(net.head.bias)
    return net


def vit_s_timm(num_classes=0, pretrained=True):

    net = timm.create_model("vit_small_patch16_384", pretrained=pretrained)
    if num_classes==0:
        net.head = nn.Identity()
    else:
        net.head = nn.Linear(net.head.in_features, num_classes)
    return net