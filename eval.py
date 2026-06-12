import os
import argparse
import numpy as np
from tqdm import tqdm
import torch.utils.data
import torch.nn as nn
from main_model import EnsembleModel
from util.dataloaders import get_eval_loaders, get_loaders
from util.common import check_eval_dirs, compute_p_r_f1_miou_oa, gpu_info, SaveResult, ScaleInOutput
from util.AverageMeter import AverageMeter, RunningMetrics
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
edge_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4]).to(device))
running_metrics =  RunningMetrics(2)
def eval(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda
    
    gpu_info()  
    save_path, result_save_path = check_eval_dirs()

    save_results = SaveResult(result_save_path)
    test_results = SaveResult(result_save_path.replace('result.txt','eval_test.txt'))
    save_results.prepare()
    test_results.prepare()

    model = EnsembleModel(opt.ckp_paths, device, input_size=opt.input_size)

    opt.dual_label = False
    # eval_loader = get_eval_loaders(opt)
    train_loader, eval_loader, test_loader = get_loaders(opt)

    p, r, f1, miou, oa, avg_loss = eval_for_metric(model, eval_loader, tta=opt.tta)
    save_results.show(p, r, f1, miou, oa)
    print("F1-mean: {}".format(f1.mean()))
    print("mIOU-mean: {}".format(miou.mean()))
    
    p, r, f1, miou, oa, avg_loss = test_for_metric_for_metric(model, test_loader, tta=opt.tta)
    test_results.show(p, r, f1, miou, oa)
    print("F1-mean: {}".format(f1.mean()))
    print("mIOU-mean: {}".format(miou.mean()))

def eval_for_metric(model, eval_loader, criterion=None, tta=False, input_size=512, dual_label=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    avg_loss = 0
    val_loss = torch.tensor([0])
    scale = ScaleInOutput(input_size)

    task_level = 1 # 1 for single output, 2 for edge and block output, 3 for DDRL Original methods 
    # 改为保存所有预测和标签
    all_preds1 = []  # 保存第一个输出的所有预测
    all_preds2 = []  # 保存第二个输出的所有预测（如果适用）
    all_labels1 = []  # 保存第一个标签的所有真实值
    all_labels2 = []  # 保存第二个标签的所有真实值（如果适用）

    model.eval()
    with torch.no_grad():
        eval_tbar = tqdm(eval_loader)
        for i, (batch_img1, batch_img2, batch_label1, batch_label2, _) in enumerate(eval_tbar):
            eval_tbar.set_description("evaluating...eval_loss: {}".format(avg_loss))

            block_mask, edge_mask = batch_label1.to(device), batch_label2.to(device)
            batch_img1 = batch_img1.float().to(device)
            batch_img2 = batch_img2.float().to(device)
            batch_label1 = batch_label1.long().to(device)
            batch_label2 =  batch_label1 #batch_label2.long().to(device) # batch_label1 #
            labels = (batch_label1, batch_label2)

            outs = model(batch_img1, batch_img2)
            if not isinstance(outs, tuple): # 仅变化图
                outs = (outs, outs)
                val_loss = criterion(outs, labels)
            elif len(outs) == 2: # 变化图+边缘
                block, edge = outs
                outs = (block, block)
                # loss_block = criterion((block, ), (batch_label1,))
                # loss_edge  = edge_criterion(edge, batch_label2.unsqueeze(1).float())
                # val_loss = loss_block + loss_edge
                loss_block = criterion((block, edge), (batch_label1, batch_label2))
                val_loss = loss_block
            elif len(outs) == 3:
                _, _, final_feature = outs
                outs = (final_feature, final_feature)
                val_loss = criterion((final_feature, ), (batch_label1,))
            _, cd_pred1 = torch.max(outs[0], 1)  
            _, cd_pred2 = torch.max(outs[1], 1)
                
            # 保存这个批次的预测和标签（移动到CPU并转换为numpy）
            all_preds1.append(cd_pred1.cpu().numpy())
            all_preds2.append(cd_pred2.cpu().numpy())
            all_labels1.append(batch_label1.cpu().numpy())
            all_labels2.append(batch_label2.cpu().numpy())
            
            avg_loss = (avg_loss * i + val_loss.cpu().detach().numpy()) / (i + 1)
            
    # 合并所有批次的预测和标签
    all_preds1 = np.concatenate(all_preds1, axis=0)
    all_preds2 = np.concatenate(all_preds2, axis=0)
    all_labels1 = np.concatenate(all_labels1, axis=0)
    all_labels2 = np.concatenate(all_labels2, axis=0)
    
    # 在整个数据集上计算混淆矩阵
    tn1 = np.sum((all_preds1 == 0) & (all_labels1 == 0))
    fp1 = np.sum((all_preds1 == 1) & (all_labels1 == 0))
    fn1 = np.sum((all_preds1 == 0) & (all_labels1 == 1))
    tp1 = np.sum((all_preds1 == 1) & (all_labels1 == 1))
    
    tn2 = np.sum((all_preds2 == 0) & (all_labels2 == 0))
    fp2 = np.sum((all_preds2 == 1) & (all_labels2 == 0))
    fn2 = np.sum((all_preds2 == 0) & (all_labels2 == 1))
    tp2 = np.sum((all_preds2 == 1) & (all_labels2 == 1))
    
    tn_fp_fn_tp = [np.array([tn1, fp1, fn1, tp1]), np.array([tn2, fp2, fn2, tp2])]
    # print(tn_fp_fn_tp)
    # import sys; sys.exit(0)
    # 计算指标
    p, r, f1, miou, oa = compute_p_r_f1_miou_oa(tn_fp_fn_tp)
    
    return p, r, f1, miou, oa, avg_loss
    
def test_for_metric(model, eval_loader, criterion=None, tta=False, input_size=512, dual_label=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    avg_loss = 0
    val_loss = torch.tensor([0])
    scale = ScaleInOutput(input_size)

    task_level = 1 # 1 for single output, 2 for edge and block output, 3 for DDRL Original methods 
    # 改为保存所有预测和标签
    all_preds1 = []  # 保存第一个输出的所有预测
    all_preds2 = []  # 保存第二个输出的所有预测（如果适用）
    all_labels1 = []  # 保存第一个标签的所有真实值
    all_labels2 = []  # 保存第二个标签的所有真实值（如果适用）

    model.eval()
    with torch.no_grad():
        eval_tbar = tqdm(eval_loader)
        for i, (batch_img1, batch_img2, batch_label1, batch_label2, _) in enumerate(eval_tbar):
            eval_tbar.set_description("evaluating...test_loss: {}".format(avg_loss))

            block_mask, edge_mask = batch_label1.to(device), batch_label2.to(device)
            batch_img1 = batch_img1.float().to(device)
            batch_img2 = batch_img2.float().to(device)
            batch_label1 = batch_label1.long().to(device)
            # batch_label2 = batch_label2.long().to(device)
            batch_label2 =  batch_label1
            labels = (batch_label1, batch_label2)

            outs = model(batch_img1, batch_img2)
            if not isinstance(outs, tuple): # 仅变化图
                outs = (outs, outs)
                val_loss = criterion(outs, labels)
            elif len(outs) == 2: # 变化图+边缘
                block, edge = outs
                outs = (block, block)
                # loss_block = criterion((block, ), (batch_label1,))
                # loss_edge  = edge_criterion(edge, batch_label2.unsqueeze(1).float())
                # val_loss = loss_block + loss_edge
                loss_block = criterion((block, edge), (batch_label1, batch_label2))
                val_loss = loss_block
                pass
            elif len(outs) == 3:
                _, _, final_feature = outs
                # outs = (final_feature, final_feature)
                # val_loss = criterion((block,), (batch_label1)) + edge_criterion(edge, batch_label2.unsqueeze(1).float())
                _, _, final_feature = outs
                outs = (final_feature, final_feature)
                val_loss = criterion((final_feature, ), (batch_label1,))
            _, cd_pred1 = torch.max(outs[0], 1)  
            _, cd_pred2 = torch.max(outs[1], 1)
                
            # 保存这个批次的预测和标签（移动到CPU并转换为numpy）
            all_preds1.append(cd_pred1.cpu().numpy())
            all_preds2.append(cd_pred2.cpu().numpy())
            all_labels1.append(batch_label1.cpu().numpy())
            all_labels2.append(batch_label2.cpu().numpy())
            
            avg_loss = (avg_loss * i + val_loss.cpu().detach().numpy()) / (i + 1)
            
    # 合并所有批次的预测和标签
    all_preds1 = np.concatenate(all_preds1, axis=0)
    all_preds2 = np.concatenate(all_preds2, axis=0)
    all_labels1 = np.concatenate(all_labels1, axis=0)
    all_labels2 = np.concatenate(all_labels2, axis=0)
    
    # 在整个数据集上计算混淆矩阵
    tn1 = np.sum((all_preds1 == 0) & (all_labels1 == 0))
    fp1 = np.sum((all_preds1 == 1) & (all_labels1 == 0))
    fn1 = np.sum((all_preds1 == 0) & (all_labels1 == 1))
    tp1 = np.sum((all_preds1 == 1) & (all_labels1 == 1))
    
    tn2 = np.sum((all_preds2 == 0) & (all_labels2 == 0))
    fp2 = np.sum((all_preds2 == 1) & (all_labels2 == 0))
    fn2 = np.sum((all_preds2 == 0) & (all_labels2 == 1))
    tp2 = np.sum((all_preds2 == 1) & (all_labels2 == 1))
    
    tn_fp_fn_tp = [np.array([tn1, fp1, fn1, tp1]), np.array([tn2, fp2, fn2, tp2])]
    # print(tn_fp_fn_tp)
    # import sys; sys.exit(0)
    # 计算指标
    p, r, f1, miou, oa = compute_p_r_f1_miou_oa(tn_fp_fn_tp)
    
    return p, r, f1, miou, oa, avg_loss



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Eval')

    parser.add_argument("--ckp-paths", type=str,
                        default=[
                            "./runs/train/7/best_ckp/",
                        ])

    parser.add_argument("--cuda", type=str, default="0")  # GPU编号
    parser.add_argument("--dataset-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--tta", type=bool, default=False)

    opt = parser.parse_args()
    print("\n" + "-" * 30 + "OPT" + "-" * 30)
    print(opt)

    eval(opt)
