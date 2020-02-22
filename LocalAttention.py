import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.nn.utils import spectral_norm
from torch.nn.init import xavier_uniform_

import numpy as np
import masks as sparse


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)



class LocalAttn(nn.Module):
    def __init__(self, in_channels):
        super(Attn, self).__init__()
        
        self.in_channels = in_channels
        self.conv1x1_theta = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.conv1x1_phi = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//8, kernel_size=1, stride=1, padding=0)
        self.conv1x1_g = nn.Conv2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=1, stride=1, padding=0)
        self.conv1x1_attn = nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.sigma = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        _, num_channels, h, w = x.size()
        location_num = h*w   
        downsampled_num = location_num //4 
        hidden_size = num_channels // 8 



        nH = 8
        head_size = hidden_size // nH   

        masks = self.get_grid_masks((h, w), (h // 2, w // 2))

        theta = self.conv1x1_theta(x)
        theta = theta.view(-1, nH, head_size, location_num)

        phi = self.conv1x1_phi(x)
        phi = self.maxpool(phi) 
        phi = phi.view(-1, nH, head_size, downsampled_num)

        attn = torch.einsum('abcd, abce -> abde',theta,phi)

        adder = (1.0 - masks) * (-1000.0)
        adder = torch.from_numpy(adder)
        
        attn += adder

        attn = F.softmax(attn, dim=-1)

        g = self.conv1x1_g(x)
        g = self.maxpool(g)

        g_hidden = num_channels // 2
        g_head_size = g_hidden // nH

        g = g.view(-1, nH, g_head_size, downsampled_num)    #b, nH, head_size, hw
        attn_g = torch.einsum('abcd, abed -> abec', attn, g)

        
        attn_g = attn_g.reshape(-1, num_channels//2, h, w)
        attn_g = self.conv1x1_attn(attn_g)
        out = x + self.sigma*attn_g

        return out
    
    def get_grid_masks(self, gridO, gridI):
        '''
            We organize the masks as following:
                - mask1: RTL
                - mask2: RTL
                - mask3: RTL
                - mask4: RTL
                - mask5: LTR
                - mask6: LTR
                - mask7: LTR
                - mask8: LTR
        '''
        masks = []

        # RTL
        masks.append(sparse.RightFloorMask.get_grid_mask_from_1d(gridI, nO=gridO))
        masks.append(sparse.RightRepetitiveMask.get_grid_mask_from_1d(gridI, nO=gridO))

        masks.append(sparse.RightFloorMask.get_grid_mask_from_1d(gridI, nO=gridO))
        masks.append(sparse.RightRepetitiveMask.get_grid_mask_from_1d(gridI, nO=gridO))

        # LTR
        masks.append(sparse.LeftFloorMask.get_grid_mask_from_1d(gridI, nO=gridO))
        masks.append(sparse.LeftRepetitiveMask.get_grid_mask_from_1d(gridI, nO=gridO))

        masks.append(sparse.LeftFloorMask.get_grid_mask_from_1d(gridI, nO=gridO))
        masks.append(sparse.LeftRepetitiveMask.get_grid_mask_from_1d(gridI, nO=gridO))

        return np.array(masks)


if __name__ == "__main__":
    tmp = torch.randn((16, 64*4, 32, 32))
    model = LocalAttn(64*4)

    for name, param in model.named_parameters():
        print('parameter name : ', name)