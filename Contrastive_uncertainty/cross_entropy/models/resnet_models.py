import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from typing import Type, Any, Callable, Union, List, Optional
from Contrastive_uncertainty.general.models.resnet_models import BasicBlock, Bottleneck, ResNet, conv1x1, conv3x3

# Differs from other cases as it only has class forward branch, no separate branch for unsupervised learning
class CustomResNet(ResNet):
    def __init__(
        self,
        latent_size: int,
        num_channels:int,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:

        super(CustomResNet, self).__init__(block,
        layers,
        num_classes,
        zero_init_residual,
        groups,
        width_per_group,
        replace_stride_with_dilation,
        norm_layer)

        self.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.maxpool = nn.Identity()

        self.class_fc1 = nn.Linear(512 * block.expansion, latent_size)
        self.class_fc2 = nn.Linear(latent_size, num_classes)
        

    # Nawid - made new function to obtain the representation of the data
    def forward(self,x:Tensor)-> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        z = F.relu(self.class_fc1(x)) # Unnormalized currently though it will be normalised in the method    
        
        return z
    
def _custom_resnet(
    arch: str,
    latent_size:int,
    num_channels:int,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = CustomResNet(latent_size,num_channels,block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def custom_resnet18(latent_size:int = 128, num_channels:int =3,pretrained: bool = False, progress: bool = True, **kwargs: Any) -> CustomResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _custom_resnet('resnet18',latent_size,num_channels, BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def custom_resnet34(latent_size:int = 128, num_channels:int =3,pretrained: bool = False, progress: bool = True, **kwargs: Any) -> CustomResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _custom_resnet('resnet34',latent_size,num_channels, BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def custom_resnet50(latent_size:int = 128, num_channels:int =3,pretrained: bool = False, progress: bool = True, **kwargs: Any) -> CustomResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _custom_resnet('resnet50',latent_size,num_channels, Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)



class GramBasicBlock(BasicBlock):
    expansion: int = 1

    def __init__(
        self,
        model,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        
    ) -> None:
        
        super(GramBasicBlock, self).__init__(inplanes,
        planes,
        stride,
        downsample,
        groups,
        base_width,
        dilation,
        norm_layer)

        self.model = model
        
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        self.model.record(out)
        
        out = self.bn1(out)
        out = self.relu(out)
        self.model.record(out)

        out = self.conv2(out)
        self.model.record(out)

        out = self.bn2(out)
        self.model.record(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        self.model.record(out)
        return out


class GramBottleneck(Bottleneck):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4
    def __init__(
        self,
        model,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        
    ) -> None:

        super(GramBottleneck, self).__init__(inplanes,
        planes,
        stride,
        downsample,
        groups,
        base_width,
        dilation,
        norm_layer)

        self.model = model

    # Need to call the neural network model in order to access the approach
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        self.model.record(out)
        out = self.bn1(out)
        out = self.relu(out)
        self.model.record(out)

        out = self.conv2(out)
        self.model.record(out)
        out = self.bn2(out)
        out = self.relu(out)
        self.model.record(out)

        out = self.conv3(out)
        self.model.record(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        self.model.record(out)
        return out

class GramResNet(ResNet):

    def __init__(
        self,
        block: Type[Union[GramBasicBlock, GramBottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:

        super(ResNet, self).__init__(block,
        layers,
        num_classes,
        zero_init_residual,
        groups,
        width_per_group,
        replace_stride_with_dilation,
        norm_layer)
        
        self.collecting = False

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, GramBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, GramBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def record(self, t):
        if self.collecting:
            self.gram_feats.append(t)

    
    def _make_layer(self, block: Type[Union[GramBasicBlock, GramBottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,model= self))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        #import ipdb; ipdb.set_trace()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
    def gram_feature_list(self,x):
        self.collecting = True
        self.gram_feats = []
        self.forward(x)
        self.collecting = False
        temp = self.gram_feats
        self.gram_feats = []
        return temp

    def get_min_max(self, data, power):
        mins = []
        maxs = []
        
        for i in range(0,len(data),128):
            batch = data[i:i+128].cuda()
            feat_list = self.gram_feature_list(batch)
            for L,feat_L in enumerate(feat_list):
                if L==len(mins):
                    mins.append([None]*len(power))
                    maxs.append([None]*len(power))
                
                for p,P in enumerate(power):
                    g_p = self.G_p(feat_L,P)
                    
                    current_min = g_p.min(dim=0,keepdim=True)[0]
                    current_max = g_p.max(dim=0,keepdim=True)[0]
                    
                    if mins[L][p] is None:
                        mins[L][p] = current_min
                        maxs[L][p] = current_max
                    else:
                        mins[L][p] = torch.min(current_min,mins[L][p])
                        maxs[L][p] = torch.max(current_max,maxs[L][p])
        
        return mins,maxs
    
    def get_deviations(self,data,power,mins,maxs):
        deviations = []
        
        for i in range(0,len(data),128):            
            batch = data[i:i+128].cuda()
            feat_list = self.gram_feature_list(batch)
            batch_deviations = []
            for L,feat_L in enumerate(feat_list):
                dev = 0
                for p,P in enumerate(power):
                    g_p = self.G_p(feat_L,P)
                    
                    dev +=  (F.relu(mins[L][p]-g_p)/torch.abs(mins[L][p]+10**-6)).sum(dim=1,keepdim=True)
                    dev +=  (F.relu(g_p-maxs[L][p])/torch.abs(maxs[L][p]+10**-6)).sum(dim=1,keepdim=True)
                batch_deviations.append(dev.cpu().detach().numpy())
            batch_deviations = np.concatenate(batch_deviations,axis=1)
            deviations.append(batch_deviations)
        deviations = np.concatenate(deviations,axis=0)
        
        return deviations
    
    def G_p(self, ob,p):
        temp = ob.detach()

        temp = temp**p
        temp = temp.reshape(temp.shape[0],temp.shape[1],-1)
        temp = ((torch.matmul(temp,temp.transpose(dim0=2,dim1=1)))).sum(dim=2) 
        temp = (temp.sign()*torch.abs(temp)**(1/p)).reshape(temp.shape[0],-1)

        return temp




class CustomGramResNet(GramResNet):
    def __init__(
        self,
        latent_size: int,
        num_channels:int,
        block: Type[Union[GramBasicBlock, GramBottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:

        super(CustomGramResNet, self).__init__(block,
        layers,
        num_classes,
        zero_init_residual,
        groups,
        width_per_group,
        replace_stride_with_dilation,
        norm_layer)

        self.conv1 = nn.Conv2d(
            num_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.maxpool = nn.Identity()

        self.class_fc1 = nn.Linear(512 * block.expansion, latent_size)
        self.class_fc2 = nn.Linear(latent_size, num_classes)
    
    # Nawid - made new function to obtain the representation of the data
    def forward(self,x:Tensor)-> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        z = F.relu(self.class_fc1(x)) # Unnormalized currently though it will be normalised in the method    
        
        return z


def _custom_gram_resnet(
    arch: str,
    latent_size:int,
    num_channels:int,
    block: Type[Union[GramBasicBlock, GramBottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = CustomGramResNet(latent_size,num_channels,block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def custom_gram_resnet18(latent_size:int = 128, num_channels:int =3,pretrained: bool = False, progress: bool = True, **kwargs: Any) -> CustomGramResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _custom_gram_resnet('resnet18',latent_size,num_channels, GramBasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def custom_gram_resnet34(latent_size:int = 128, num_channels:int =3,pretrained: bool = False, progress: bool = True, **kwargs: Any) -> CustomGramResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _custom_gram_resnet('resnet34',latent_size,num_channels, GramBasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def custom_gram_resnet50(latent_size:int = 128, num_channels:int =3,pretrained: bool = False, progress: bool = True, **kwargs: Any) -> CustomGramResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _custom_resnet('resnet50',latent_size,num_channels, GramBottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)