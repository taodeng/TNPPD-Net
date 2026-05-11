import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


class CBR(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.ReLU6()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output
    
class BR(nn.Module):
    def __init__(self, nOut):
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.ReLU6()

    def forward(self, input):
        output = self.bn(input)
        output = self.act(output)
        return output
    
class CB(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False,
                              groups=groups)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return output
    
class C(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False,
                              groups=groups)

    def forward(self, input):
        output = self.conv(input)
        return output
    
class CDilated(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut,kSize, stride=stride, padding=padding, bias=False,
                              dilation=d, groups=groups)

    def forward(self, input):
        output = self.conv(input)
        return output
    
class CDilatedB(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut,kSize, stride=stride, padding=padding, bias=False,
                              dilation=d, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)

    def forward(self, input):
        return self.bn(self.conv(input))

class Interpolate(nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()
        self.scale_factor = 2
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True)


class EESP(nn.Module):
    def __init__(self, nIn, nOut, stride=1, k=4, r_lim=7, down_method='esp'):
        super().__init__()
        self.stride = stride
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n
        assert down_method in ['avg', 'esp'], 'One of these is suppported (avg or esp)'
        assert n == n1, "n(={}) and n1(={}) should be equal for Depth-wise Convolution ".format(n, n1)
        self.proj_1x1 = CBR(nIn, n, 1, stride=1, groups=k)

        map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
        self.k_sizes = list()
        for i in range(k):
            ksize = int(3 + 2 * i)
            ksize = ksize if ksize <= r_lim else 3
            self.k_sizes.append(ksize)
        self.k_sizes.sort()
        self.spp_dw = nn.ModuleList()
        for i in range(k):
            d_rate = map_receptive_ksize[self.k_sizes[i]]
            self.spp_dw.append(CDilated(n, n, kSize=3, stride=stride, groups=n, d=d_rate))
        self.conv_1x1_exp = CB(nOut, nOut, 1, 1, groups=k)
        self.br_after_cat = BR(nOut)
        self.module_act = nn.ReLU6()
        self.downAvg = True if down_method == 'avg' else False

    def forward(self, input):
        output1 = self.proj_1x1(input)
        output = [self.spp_dw[0](output1)]
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            out_k = out_k + output[k - 1]
            output.append(out_k)

        expanded = self.conv_1x1_exp(
            self.br_after_cat(
                torch.cat(output, 1)
            )
        )
        del output

        if self.stride == 2 and self.downAvg:
            return expanded

        if expanded.size() == input.size():
            expanded = expanded + input

        return self.module_act(expanded)
    
class DownSampler(nn.Module):
    def __init__(self, nin, nout, k=4, r_lim=9, reinf=True):
        super().__init__()
        nout_new = nout - nin
        self.eesp = EESP(nin, nout_new, stride=2, k=k, r_lim=r_lim, down_method='avg')
        self.avg = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
        if reinf:
            self.inp_reinf = nn.Sequential(
                CBR(1, 3, 3, 1),
                CB(3, nout, 1, 1)
            )
        self.act =  nn.ReLU6(nout)

    def forward(self, input, input2=None):
        avg_out = self.avg(input)
        eesp_out = self.eesp(input)
        output = torch.cat([avg_out, eesp_out], 1)

        if input2 is not None:
            w1 = avg_out.size(2)
            while True:
                input2 = F.avg_pool2d(input2, kernel_size=3, padding=1, stride=2)
                w2 = input2.size(2)
                if w2 == w1:
                    break
            output = output + self.inp_reinf(input2)

        return self.act(output)
    
class UpSampler(nn.Module):
    def __init__(self, nin, nout, k=4, r_lim=9, reinf=True):
        super().__init__()
        n = nout//2
        self.eesp = EESP(n, n, stride=1, k=k, r_lim=r_lim, down_method='avg')
        self.up = C(nin, n, 1, 1)
        self.upsample = Interpolate()

        if reinf:
            self.inp_reinf = nn.Sequential(
                CBR(1, 3, 3, 1),
                CB(3, nout, 1, 1)
            )
        self.act =  nn.ReLU6(nout)

    def forward(self, input, input2=None):
        input = self.up(input)
        upsampled_out = self.upsample(input)

        eesp_out = self.eesp(upsampled_out)
        output = torch.cat([upsampled_out, eesp_out], 1)
        
        if input2 is not None:
            w1 = upsampled_out.size(2)
            while True:
                # input2 = self.upsample(input2)
                input2 = F.avg_pool2d(input2, kernel_size=3, padding=1, stride=2)
                w2 = input2.size(2)
                if w2 == w1:
                    break
            output = output + self.inp_reinf(input2)

        return self.act(output)



class Star(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = CB(dim, dim, 7, 1, groups=dim)
        self.f1 = C(dim, mlp_ratio * dim, 1)
        self.f2 = C(dim, mlp_ratio * dim, 1)
        self.g = CB(mlp_ratio * dim, dim, 1)
        self.dwconv2 = C(dim, dim, 7, 1, groups=dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class qkvFusion(nn.Module):
    def __init__(self):
        super(qkvFusion, self).__init__()

        self.ip = Interpolate()
        self.conv = C(512, 32, 1, 1)
        

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, seg, v):
        m_batchsize, C, height,width = v.size()

        x = self.ip(seg)
        proj_query = x.view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = x.view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = v.view(m_batchsize, -1, width*height)
        o1 = torch.bmm(proj_value, attention.permute(0, 2, 1))
        o1 = o1.view(m_batchsize, C, height, width)

        x = self.conv(seg)
        c_proj_query = x.view(m_batchsize, C, -1)
        c_proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        c_energy = torch.bmm(c_proj_query, c_proj_key)
        c_energy_new = torch.max(c_energy, -1, keepdim=True)[0].expand_as(c_energy)-c_energy
        c_attention = self.softmax(c_energy_new)
        c_proj_value = v.view(m_batchsize, C, -1)
        o2 = torch.bmm(c_attention, c_proj_value)
        o2 = o2.view(m_batchsize, C, height, width)

        output = self.gamma1*o1 + self.gamma2*o2 + v

        return output


class TNPPD(nn.Module):
    def __init__(self, in_channels = 1, mlp_ratio = 4, drop_path_rate=0.0, num_classes = 2):
        super().__init__()

        depths =[4]
        depths_s=[3, 7, 3]
        config = [32,64,128,256,512]
        r_lim = [13, 11, 9, 7, 5]
        K = [4]*len(r_lim)

#------------------------encoder----------------------------
        self.all_level1 = CBR(in_channels, config[0], 3, 2)
        for i in range(2, 5):
            setattr(self, f'seg_level{i}_0', DownSampler(config[i-2], config[i-1], K[i-2], r_lim[i-2]))
        for level in range(2, 5):
            module_list = nn.ModuleList()
            for i in range(depths_s[level - 2]):
                module_list.append(EESP(config[level-1], config[level-1], 1, K[level-1], r_lim[level-1]))
            setattr(self, f'seg_level{level}', module_list)
        self.seg_level5_1 = CBR(config[3], config[4], 1, 1, K[3])
        self.seg_level5_2 = CBR(config[4], config[4], 3, 1, config[3])

        self.cls_level2 = DownSampler(config[0], config[1], K[0], r_lim[0])
        self.cls_level3 = CB(config[1], config[0], 3, 2)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.cls_blocks = nn.ModuleList([Star(config[0], mlp_ratio, dpr[i]) for i in range(depths[0])])

        self.fu_conv = nn.Sequential(C(config[4], config[0], 1, 1),Interpolate())
        self.fu_convfusion = CBR(config[1], config[0], 1)
        self.fu_qkvfusion = qkvFusion()

        self.cls_bn = nn.BatchNorm2d(config[0])
        self.cls_avgpool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(config[0], num_classes)

#------------------------decoder----------------------------         
        self.seg_decoder5 = CBR(config[4], config[3], 1, 1, K[3])
        for i in range(2, 5):
            setattr(self, f'seg_decoder{i}', UpSampler(config[i-1], config[i-2], K[i-2], r_lim[i-2]))
        for level in range(2, 5):
            module_list = nn.ModuleList()
            for i in range(depths_s[level-2]):
                module_list.append(EESP(config[level-1], config[level-1], 1, K[level-1], r_lim[level-1]))
            setattr(self, f'seg_decoder{level}_0', module_list)
        self.seg_decoder1 = nn.Sequential(nn.ConvTranspose2d(config[0], 1, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),nn.BatchNorm2d(1))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        

    def forward(self, input):
#------------------------encoder----------------------------    # input [B, 1, 224, 224]
        level1 = self.all_level1(input)                         # level1[B, 32, 112, 112]
        
        level2_x2 = self.seg_level2_0(level1, input)            # level2[B, 64, 56, 56]    
        for i, layer in enumerate(self.seg_level2): 
            level2_x2  = layer(level2_x2)              
        level3_x2 = self.seg_level3_0(level2_x2, input)         # level3[B, 128, 28, 28]   
        for i, layer in enumerate(self.seg_level3): 
            level3_x2  = layer(level3_x2)
        level4_x2 = self.seg_level4_0(level3_x2, input)         # level4[B, 256, 14, 14]
        for i, layer in enumerate(self.seg_level4): 
            level4_x2  = layer(level4_x2)
 

        level5_x2 = self.seg_level5_1(level4_x2)   
        level5_x2 = self.seg_level5_2(level5_x2)                # level5[B, 512, 14, 14]


        x1 = self.cls_level2(level1, input)                     # [B, 64, 56, 56] 
        x1 = self.cls_level3(x1)                                # [B, 32, 28, 28] 

        level2_x1 = self.cls_blocks[0](x1)  
        level3_x1 = self.cls_blocks[1](level2_x1)  

        fu = torch.cat((level3_x1, self.fu_conv(level5_x2)), dim=1)
        fu = self.fu_convfusion(fu) 
        fu = self.fu_qkvfusion(level5_x2, fu)


        level4_x1 = self.cls_blocks[2](fu)  
        level5_x1 = self.cls_blocks[3](level4_x1 + level2_x1)  

        x1 = torch.flatten(self.cls_avgpool(self.cls_bn(level5_x1)), 1)
        cls = self.cls_head(x1)
       
#------------------------decoder----------------------------
        decoder_x = self.seg_decoder5(level5_x2) 
        decoder_layers = [
            (level4_x2, self.seg_decoder4_0, self.seg_decoder4),
            (level3_x2, self.seg_decoder3_0, self.seg_decoder3),
            (level2_x2, self.seg_decoder2_0, self.seg_decoder2)
        ]
        for fusion_level,layers,up in decoder_layers:
            decoder_x = decoder_x + fusion_level
            for layer in layers:
                decoder_x = layer(decoder_x)
            decoder_x = up(decoder_x, input)

        x2 = self.seg_decoder1(decoder_x)
        seg = (x2 - x2.min()) / (x2.max() - x2.min())

        return cls, seg
