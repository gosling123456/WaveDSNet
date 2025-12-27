import os
import re
import datetime # 自动停止
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from eval import eval_for_metric, test_for_metric
from losses.get_losses import SelectLoss
from util.dataloaders import get_loaders
from util.common import check_dirs, init_seed, gpu_info, SaveResult, CosOneCycle, ScaleInOutput
from torch.cuda.amp import autocast, GradScaler
from losses.about_use_loss import DepthContrastLoss
from models.DWT import WaveDSNet

def dropblock_step(model):
    """
    更新 dropblock的drop率
    """
    neck = model.module.neck if hasattr(model, "module") else model.neck
    if hasattr(neck, "drop"):
        neck.drop.step()

def train_args():
    parser = argparse.ArgumentParser('Train')
    parser.add_argument("--backbone", type=str, default="cswin_t_64")
    # parser.add_argument("--net_G", type=str, default="resnet18")
    parser.add_argument("--neck", type=str, default="fpn+drop")
    parser.add_argument("--head", type=str, default="fcn")
    parser.add_argument("--loss", type=str, default="bce+dice")

    parser.add_argument("--pretrain", type=str, default="")  
    parser.add_argument("--cuda", type=str, default="0,1")
    parser.add_argument("--dataset-dir", type=str, default="../data/LEVIR-CD-full/")
    parser.add_argument("--label_file_path", type=str, default="../data/LEVIR-CD-full/list-难度1")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.00035)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--dual-label", type=bool, default=False)
    parser.add_argument("--finetune", type=bool, default=True)
    parser.add_argument("--pseudo-label", type=bool, default=False)
    parser.add_argument("--resume", type=str, default='', help="resume training from latest checkpoint")
    parser.add_argument("--use_pretrained", type=bool, default=False)
    parser.add_argument("--edge_label", type=str, default='')
    
    opt = parser.parse_args()
    print(opt)
    return opt
def train(opt):
    opt.dual_label = False
        
    init_seed()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_info()  
    save_path, best_ckp_save_path, best_ckp_file, result_save_path, _ = check_dirs(opt.resume) if opt.resume else check_dirs(save_path=False) 
    # 创建最新检查点保存目录
    latest_ckp_save_path = os.path.join(save_path, "latest")
    os.makedirs(latest_ckp_save_path, exist_ok=True)
    latest_ckp_file = os.path.join(latest_ckp_save_path, "latest.pt")

    save_results = SaveResult(result_save_path)
    test_results = SaveResult(result_save_path.replace('result.txt','eval_test.txt'))
    save_results.prepare()
    test_results.prepare()

    train_loader, val_loader, test_loader = get_loaders(opt)
    
    scale = ScaleInOutput(opt.input_size)
    loss_Contrastive = DepthContrastLoss()

    model = WaveDSNet(opt)

    if opt.cuda!='-1':
        model = nn.DataParallel(model)
    model.to(device)

    lamda = 0.7 # 正体损失占比
    # 添加恢复训练的代码
    start_epoch = 0
    best_metric = 0
    best_metric_test = 0
    if opt.resume and os.path.exists(latest_ckp_file):
        print(f"Resuming training from {latest_ckp_file}")
        # checkpoint = torch.load(latest_ckp_file)
        checkpoint = torch.load(latest_ckp_file, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint['best_metric']
        print(f"Resumed from epoch {start_epoch}, best metric: {best_metric}")

    params = model.parameters()
    if opt.finetune:
        params = [{"params": [param for name, param in model.named_parameters()
                              if "backbone" in name], "lr": opt.learning_rate / 10},  # 微调backbone
                  {"params": [param for name, param in model.named_parameters()
                              if "backbone" not in name ], "lr": opt.learning_rate},
                 ]  # 其它层正常学习
        print("Using finetune for model")
        # params = model.parameters()
        pass
    else:
        params = model.parameters()
    
    optimizer = torch.optim.AdamW(params, lr=opt.learning_rate, weight_decay=opt.weight_decay)
    
    # 如果恢复训练，也需要加载优化器状态
    if opt.resume and os.path.exists(latest_ckp_file):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if opt.pseudo_label:
        scheduler = CosOneCycle(optimizer, max_lr=opt.learning_rate/5, epochs=opt.epochs, up_rate=0)
    else:
        scheduler = CosOneCycle(optimizer, max_lr=opt.learning_rate, epochs=opt.epochs)  # 自己定义的onecycle
    
    # 如果恢复训练，也需要加载调度器状态
    if opt.resume and os.path.exists(latest_ckp_file):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    train_avg_loss = 0
    total_bs = 8 # 多少个批次下更新一次学习率
    accumulate_iter = max(round(total_bs / opt.batch_size), 1)
    print("Accumulate_iter={} batch_size={}".format(accumulate_iter, opt.batch_size))
    
    scaler = GradScaler() # 用于梯度缩放，防止半精度下的梯度下溢
    criterion = SelectLoss(opt.loss)
    edge_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4]).to(device))
    for epoch in range(start_epoch, opt.epochs):
        model.train()
        optimizer.zero_grad()
        train_tbar = tqdm(train_loader)
        for i, (batch_img1, batch_img2, batch_label1, batch_label2, _) in enumerate(train_tbar):
            train_tbar.set_description("epoch {}, train_loss {}".format(epoch, train_avg_loss))

            mask, edge_mask = batch_label1.to(device), batch_label2.to(device)
            batch_img1 = batch_img1.float().to(device)
            batch_img2 = batch_img2.float().to(device)
            batch_label1 = batch_label1.long().to(device)
            batch_label2 = batch_label2.long().to(device)
            
            outs = model(batch_img1, batch_img2)
            block, edge = outs
            loss_block = criterion((block, ), (batch_label1,))
            loss_edge  = criterion((edge, ), (batch_label2, ))
            loss = loss_block + loss_edge

            # 反向传播
            loss = loss / accumulate_iter
            loss.backward()
        
            if ((i+1) % accumulate_iter) == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_avg_loss = (train_avg_loss * i + loss .detach().item() * accumulate_iter) / (i + 1)
            del batch_img1, batch_img2, batch_label1, batch_label2

        scheduler.step()
        # 评价
        p, r, f1, miou, oa, val_avg_loss = eval_for_metric(model, val_loader, criterion, input_size=opt.input_size, dual_label=opt.dual_label)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        refer_metric = f1
        underscore = "_"
        
        # 保存最佳模型
        if refer_metric.mean() > best_metric:
            if best_ckp_file is not None:
                os.remove(best_ckp_file)
            best_ckp_file = os.path.join(
                best_ckp_save_path,
                underscore.join([opt.backbone, opt.neck, opt.head, 'epoch',
                                    str(epoch), str(round(float(refer_metric.mean()), 5))]) + ".pth")
            torch.save(model.state_dict(), best_ckp_file)
            best_metric = refer_metric.mean()

            # 看一下测试集
            p_test, r_test, f1_test, miou_test, oa_test, val_avg_loss_test = test_for_metric(model, test_loader, criterion, input_size=opt.input_size, dual_label=opt.dual_label)
            refer_metric_test = f1_test
            if refer_metric_test.mean() > best_metric_test:
                best_metric_test = refer_metric_test.mean()
            test_results.show(p_test, r_test, f1_test, miou_test, oa_test, refer_metric_test, best_metric_test, train_avg_loss, val_avg_loss_test, lr, epoch)
         
        # 保存最新检查点（包括模型、优化器、调度器状态等）
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_metric': best_metric,
            'train_avg_loss': train_avg_loss,
        }, latest_ckp_file)

        save_results.show(p, r, f1, miou, oa, refer_metric, best_metric, train_avg_loss, val_avg_loss, lr, epoch)
        print()


        
if __name__ == '__main__':
    opt = train_args()
    train(opt)
    # python main.py --cuda 0,1 --resume 难度1/train/DDRL_with_3_loss --net_G DDRL_with_3_loss --batch_size 2 --label_file_path ../../data/LEVIR-CD-full/list-难度1