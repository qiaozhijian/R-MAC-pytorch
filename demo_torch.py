import torch
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import numpy as np
from rmac_torch import RMAC

def demo():
    encoder = models.vgg16(pretrained=False).cuda()
    # capture only feature part and remove last relu and maxpool
    layers = list(encoder.features.children())[:-2]
    encoder = nn.Sequential(*layers)

    x = load_image('adorable-al-reves-animal-atigrado-248280.jpg')
    y = encoder(x) # [B,K,W,H]
    print('y shape: {}'.format(y.shape))

    rmac = RMAC(y.shape, levels=5, norm_fm=True, sum_fm=True)

    rmac_f = rmac.rmac(y)
    print('rmac_f shape: {}'.format(rmac_f.shape))

def load_image(path, bacth_size=2, target_size=(640,480)):
    x = Image.open(path).resize(target_size,Image.ANTIALIAS)
    x = np.array(x)
    x = torch.from_numpy(x).float()
    x = x.unsqueeze(0).repeat(bacth_size,1,1,1).permute(0,3,1,2).cuda()
    return x

if __name__ == '__main__':
    demo()