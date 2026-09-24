import torch
import torch.nn as nn
import torchvision.models as tvm
from ..utils.utils import get_autocast_params


def normalize_image_tensor(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    device = x.device
    mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


class VGG19(nn.Module):
    def __init__(self, pretrained=False, amp=False, amp_dtype=torch.float16) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tvm.vgg19_bn(pretrained=pretrained).features[:40])
        self.amp = amp
        self.amp_dtype = amp_dtype

    def forward(self, x, **kwargs):
        autocast_device, autocast_enabled, autocast_dtype = get_autocast_params(x.device, self.amp, self.amp_dtype)
        with torch.autocast(device_type=autocast_device, enabled=autocast_enabled, dtype=autocast_dtype):
            x = normalize_image_tensor(x)
            feats = {}
            scale = 1
            for layer in self.layers:
                if isinstance(layer, nn.MaxPool2d):
                    feats[scale] = x
                    scale = scale * 2
                x = layer(x)
            return feats
