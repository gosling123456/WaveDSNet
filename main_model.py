import os
import re
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cswin_DWT import CSWin_64_12211_tiny_224, CSWin_64_24322_small_224, CSWin_96_24322_base_384, CSWin_96_24322_base_224
from collections import OrderedDict
from util.common import ScaleInOutput

class Conv3Relu(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(Conv3Relu, self).__init__()
        self.extract = nn.Sequential(nn.Conv2d(in_ch, out_ch, (3, 3), padding=(1, 1),
                                               stride=(stride, stride), bias=False),
                                     nn.BatchNorm2d(out_ch),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.extract(x)
        return x



class EnsembleModel(nn.Module):
    def __init__(self, ckp_paths, device, method="avg2", input_size=512):
        super(EnsembleModel, self).__init__()
        self.method = method
        self.models_list = []
        assert isinstance(ckp_paths, list), "ckp_path must be a list: {}".format(ckp_paths)
        print("-"*50+"\n--Ensamble method: {}".format(method))
        for ckp_path in ckp_paths:
            if os.path.isdir(ckp_path):
                weight_file = os.listdir(ckp_path)
                ckp_path = os.path.join(ckp_path, weight_file[0])
            print("--Load model: {}".format(ckp_path))
            
            model = torch.load(ckp_path, map_location='cpu')
            model = model.to(device)
            if isinstance(model, torch.nn.parallel.DistributedDataParallel) \
                    or isinstance(model, nn.DataParallel):
                model = model.module
            self.models_list.append(model)
        self.scale = ScaleInOutput(input_size)

    def eval(self):
        for model in self.models_list:
            model.eval()

    def forward(self, xa, xb, tta=False):
        xa, xb = self.scale.scale_input((xa, xb))
        out1, out2 = 0, 0
        cd_pred1 = None

        for i, model in enumerate(self.models_list):
            _, _, outs = model(xa, xb, tta)
            if not isinstance(outs, tuple):  
                outs = (outs, outs)
            outs = self.scale.scale_output(outs)
            if "avg" in self.method:
                if self.method == "avg2":
                    outs = (F.softmax(outs[0], dim=1), F.softmax(outs[1], dim=1))  
                out1 += outs[0]
                out2 += outs[1]
                _, cd_pred1 = torch.max(out1, 1) 
            elif self.method == "vote":  
                _, out1_tmp = torch.max(outs[0], 1) 
                _, out2_tmp = torch.max(outs[1], 1)
                out1 += out1_tmp
                out2 += out2_tmp
                cd_pred1 = out1 / i >= 0.5

        return _, _, cd_pred1