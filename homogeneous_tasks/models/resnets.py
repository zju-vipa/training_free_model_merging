import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import models as torchvision_models
torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))


from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = norm_layer(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            #             self.shortcut = LambdaLayer(lambda x:
            #                                         F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          planes,
                          kernel_size=3,
                          stride=stride,
                          padding=1,
                          bias=False), norm_layer(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 num_blocks,
                 w=1,
                 num_classes=10,
                 text_head=False,
                 norm_layer=nn.BatchNorm2d):
        super(ResNet, self).__init__()
        self.in_planes = int(w * 16)

        self.conv1 = nn.Conv2d(3,
                               int(w * 16),
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.norm_layer = norm_layer
        self.bn1 = norm_layer(int(w * 16))
        self.layer1 = self._make_layer(block,
                                       int(w * 16),
                                       num_blocks[0],
                                       stride=1)
        self.layer2 = self._make_layer(block,
                                       int(w * 32),
                                       num_blocks[1],
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       int(w * 64),
                                       num_blocks[2],
                                       stride=2)
        if text_head:
            num_classes = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(int(w * 64), num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(self.in_planes,
                      planes,
                      stride,
                      norm_layer=self.norm_layer))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)  # F.avg_pool2d(out, int(out.size()[3]))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(w=1, num_classes=10, text_head=False):
    return ResNet(BasicBlock, [3, 3, 3],
                  w=w,
                  num_classes=num_classes,
                  text_head=text_head)


def resnet20gn(w=1, num_classes=10, text_head=False, group_num=8):
    return ResNet(BasicBlock, [3, 3, 3],
                  w=w,
                  num_classes=num_classes,
                  text_head=text_head,
                  norm_layer=lambda x: nn.GroupNorm(group_num, x))


def resnet26timm(num_classes=10, pretrained=True):
    import timm
    net = timm.create_model("resnet26", pretrained=pretrained)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def resnet50gntimm(num_classes=10, pretrained=True):
    import timm
    net = timm.create_model("resnet50_gn", pretrained=pretrained)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return net


def resnet50dino(num_classes=10, pretrained=True):
    import timm
    model = timm.create_model("resnet50_dino", pretrained=pretrained)
    model.fc = torch.nn.Linear(2048, num_classes, bias=True)
    torch.nn.init.xavier_normal_(model.fc.weight)
    torch.nn.init.zeros_(model.fc.bias)
    return model


@register_model
def resnet50_dino(pretrained=False, image_res=224, **kwargs):
    if pretrained:
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    else:
        model = torchvision_models.__dict__["resnet50"]()
    
    return model

