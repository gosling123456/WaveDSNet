import torch
import torch.nn as nn

from losses.compound_loss import BceDiceLoss


class SelectLoss(nn.Module):
    def __init__(self, loss_name):
        super(SelectLoss, self).__init__()
        if loss_name == "bce+dice":
            self.loss = BceDiceLoss()
        else:
            raise Exception('Error. This loss function hasn\'t been defined: {}'.format(self.loss_name))

    def forward(self, outs, labels, weight=(1, 1), aux_weight=0.4):
        loss = 0
        aux_loss = 0

        for i, label in enumerate(labels):
            for j, out in enumerate(outs[i::len(labels)]):
                if j == 0:
                    loss += self.loss(out, label) * weight[i]
                else:
                    aux_loss += self.loss(out, label) * weight[i]
        loss = loss / len(labels)   
        aux_loss = aux_loss / (len(outs) - len(labels)) * aux_weight if (len(outs) != len(labels)) else 0

        return loss + aux_loss

class DualTaskLoss(nn.Module):
    """变化检测+边缘检测的双任务损失"""
    def __init__(self, alpha=0.7, beta=1.0):
        """
        参数:
            alpha: 主损失中BCE与Dice的权重比
            beta: 边缘损失的权重系数
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_logits = nn.BCEWithLogitsLoss()
    
    def forward(self, pred_change, pred_edge, mask, edge_mask):
        """
        输入:
            pred_change: 变化预测logits (B, 1, H, W)
            pred_edge: 边缘预测logits (B, 1, H, W)
            mask: 真实变化标签 (B, H, W)
            edge_mask: 真实边缘标签 (B, H, W)
        输出:
            total_loss: 加权总损失
        """
        # 变化检测损失 (BCEWithLogits + Dice)
        bce_loss = self.bce_logits(pred_change.squeeze(1), mask.float())
        
        pred_probs = torch.sigmoid(pred_change)
        smooth = 1.0
        intersection = (pred_probs * mask.unsqueeze(1)).sum()
        dice = (2. * intersection + smooth) / (pred_probs.sum() + mask.sum() + smooth)
        dice_loss = 1 - dice
        
        change_loss = self.alpha * bce_loss + (1 - self.alpha) * dice_loss
        
        # 边缘检测损失 (BCEWithLogits)
        edge_loss = self.bce_logits(pred_edge.squeeze(1), edge_mask)
        
        # 总损失
        total_loss = change_loss + self.beta * edge_loss
        
        return total_loss, change_loss, edge_loss

class EATDer_loss(nn.Module):
    """变化检测+边缘检测的双任务损失"""
    def __init__(self, alpha=0.7):
        """
        参数:
            alpha: 主损失中BCE与Dice的权重比
            beta: 边缘损失的权重系数
        """
        super().__init__()
        self.alpha = alpha
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4]).to(self.device))
        self.criterion2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4]).to(self.device))
    
    def forward(self, pred, mask, alpha=0.3):
        if len(mask) == 1:
            pred = pred[0], pred[1]
            mask = mask[0], mask[1]
        edge, block, block_label, edge_label = pred[0], pred[1], mask[0], mask[1]
        loss = (1 - self.alpha) * self.criterion1(edge, edge_label.float()) + self.alpha * self.criterion2(block, block_label.float())
        return loss
# def EATDer_loss(pred, mask, alpha=0.3):
#     device = pred[0].device
#     criterion1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4]).to(device))
#     criterion2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4]).to(device))
#     if len(mask) == 1:
#         pred = pred[0], pred[1]
#         mask = mask[0], mask[1]
#     edge, block, block_label, edge_label = pred[0], pred[1], mask[0], mask[1]
#     loss = (1 - alpha) * criterion1(edge, edge_label.float()) + alpha * criterion2(block, block_label.float())
#     return loss
        