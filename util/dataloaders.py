import os
import random
from PIL import Image
import torch
import torch.utils.data as data
import numpy as np
from util import transforms as tr
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader

    
def get_loaders(opt):  
    train_dataset = CDDloader(opt, 'train', aug=False)
    # train_dataset = CDDloader(opt, 'val', aug=False)
    val_dataset = CDDloader(opt, 'val', aug=False)
    test_dataset = CDDloader(opt, 'test', aug=False)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True,drop_last=False,
                                               num_workers=opt.num_workers,
                                               pin_memory=True
                                              )
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=opt.num_workers,
                                             pin_memory=True
                                            )
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=opt.num_workers,
                                             pin_memory=True
                                             )
    return train_loader, val_loader, test_loader



def get_eval_loaders(opt):   
    dataset_name = "test"
    print("using dataset: {} set".format(dataset_name))
    eval_dataset = CDDloader(opt, dataset_name, aug=False)
    eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=False,drop_last=False,
                                              num_workers=opt.num_workers)
    return eval_loader

def get_vis_loaders():
    pass

def get_train_loaders(opt):  
    train_dataset = CDDloader(opt, 'train', aug=False)
    # train_dataset = CDDloader(opt, 'val', aug=False)
    val_dataset = CDDloader(opt, 'val', aug=False)
    # test_dataset = CDDloader(opt, 'test', aug=False)

    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    train_loader = torch.utils.data.DataLoader(combined_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True,drop_last=False,
                                               num_workers=opt.num_workers,
                                               pin_memory=True
                                              )
   
    return train_loader



def get_infer_loaders(opt):
    infer_datast = CDDloadImageOnly(opt, 'test', aug=False)
    infer_loader = torch.utils.data.DataLoader(infer_datast,
                                               batch_size=opt.batch_size,
                                               shuffle=False,drop_last=False,
                                               num_workers=opt.num_workers)
    return infer_loader


class CDDloader(data.Dataset):

    def __init__(self, opt, phase, aug=False):
        self.data_dir = str(opt.dataset_dir)
        self.dual_label = opt.dual_label  
        self.phase = str(phase)
        self.aug = aug
        self.edge_label = opt.edge_label
        names = [i.strip() for i in open(f'{opt.label_file_path}/{self.phase}.txt','r',encoding='utf-8').readlines()]
        self.names = []
        for name in names:
            if is_img(name):  
                self.names.append(name)
                
        # self.names.append.append()
        random.shuffle(self.names)

    def __getitem__(self, index):

        name = str(self.names[index])
        img1 = Image.open(os.path.join(self.data_dir, 'A', name)).convert('RGB')
        img2 = Image.open(os.path.join(self.data_dir, 'B', name)).convert('RGB')
        
        if img1.mode == 'L': 
            img1 = img1.convert('RGB')
        label_name = name.replace("tif", "png") if name.endswith("tif") else name   # for shengteng
        label1 = Image.open(os.path.join(self.data_dir, 'label', label_name)).convert('P')
        if self.edge_label:
            # label2 = Image.open(os.path.join(self.data_dir, 'label2', label_name))
            label2 = Image.open(os.path.join(self.data_dir, self.edge_label, label_name)).convert('P')
        else:   
            label2 = label1
       

        if self.aug:
            img1, img2, label1, label2 = tr.with_augment_transforms([img1, img2, label1, label2])
        else:
            img1, img2, label1, label2 = tr.without_augment_transforms([img1, img2, label1, label2])
        if not self.edge_label:
            label2 = label1
        return img1, img2, label1, label2, name

    def __len__(self):
        return len(self.names)


def is_img(name):
    img_format = ["jpg", "png", "jpeg", "bmp", "tif", "tiff", "TIF", "TIFF"]
    if "." not in name:
        return False
    if name.split(".")[-1] in img_format:
        return True
    else:
        return False

class CDDloadImageOnly(data.Dataset):

    def __init__(self, opt, phase, aug=False):
        self.data_dir = str(opt.dataset_dir)
        self.phase = str(phase)
        self.aug = aug
        names = [i for i in os.listdir(os.path.join(self.data_dir, phase, 'A'))]
        self.names = []
        for name in names:
            if is_img(name):
                self.names.append(name)

    def __getitem__(self, index):

        name = str(self.names[index])
        img1 = Image.open(os.path.join(self.data_dir, self.phase, 'A', name))
        img2 = Image.open(os.path.join(self.data_dir, self.phase, 'B', name))

        img1, img2 = tr.infer_transforms([img1, img2])

        return img1, img2, name

    def __len__(self):
        return len(self.names)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train')
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--net_G", type=str, default="resnet18")
    parser.add_argument("--neck", type=str, default="fpn+drop")
    parser.add_argument("--head", type=str, default="fcn")
    parser.add_argument("--loss", type=str, default="bce+dice")

    parser.add_argument("--pretrain", type=str, default="")  
    parser.add_argument("--cuda", type=str, default="0,1")
    parser.add_argument("--dataset-dir", type=str, default="../../data/LEVIR-CD-full/")  # label_file_path
    parser.add_argument("--label_file_path", type=str, default="../../data/LEVIR-CD-full/list-难度1")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.00035)
    parser.add_argument("--dual-label", type=bool, default=False)
    parser.add_argument("--finetune", type=bool, default=True)
    parser.add_argument("--pseudo-label", type=bool, default=False)
    # 添加恢复训练参数
    parser.add_argument("--resume", type=str, default='', help="resume training from latest checkpoint")
    parser.add_argument("--use_pretrained", type=bool, default=False)
    parser.add_argument("--Ablation", type=bool, default=False)
    parser.add_argument("--use_amp", type=bool, default=False)# ignore_unchange
    parser.add_argument("--ignore_unchange", type=bool, default=False)
    parser.add_argument("--edge_label", type=str, default='edge')
    opt = parser.parse_args()

    train_loader, val_loader, test_loader = get_loaders(opt)
    for i, (batch_img1, batch_img2, batch_label1, batch_label2, _) in enumerate(val_loader):
        print(batch_label1)
        print(batch_label2)
        