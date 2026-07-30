import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F


def mkdir(path):
    os.makedirs(path, exist_ok=True)


def weight_to_heatmap(weight):
    """
    weight: numpy array, shape [H, W], value usually in [0, 1]
    """
    weight = np.squeeze(weight)
    weight = np.clip(weight, 0, 1)
    weight_uint8 = (weight * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(weight_uint8, cv2.COLORMAP_JET)
    return heatmap


def save_W1_W2_maps(W1, W2, names, save_dir, target_size=None):
    """
    W1/W2: Tensor [B, 1, H, W]
    names: 当前 batch 的文件名
    save_dir: 保存路径
    target_size: 上采样到原图尺寸，例如 batch_img1.shape[-2:]
    """

    W1_dir = os.path.join(save_dir, "W1")
    W2_dir = os.path.join(save_dir, "W2")
    compare_dir = os.path.join(save_dir, "W1_W2_compare")
    npy_dir = os.path.join(save_dir, "npy")

    mkdir(W1_dir)
    mkdir(W2_dir)
    mkdir(compare_dir)
    mkdir(npy_dir)

    if target_size is not None:
        W1 = F.interpolate(W1, size=target_size, mode="bilinear", align_corners=False)
        W2 = F.interpolate(W2, size=target_size, mode="bilinear", align_corners=False)

    W1 = W1.detach().cpu().float().numpy()
    W2 = W2.detach().cpu().float().numpy()

    B = W1.shape[0]

    for b in range(B):
        if isinstance(names, (list, tuple)):
            img_name = names[b]
        else:
            img_name = names

        base_name = os.path.splitext(os.path.basename(img_name))[0]

        w1 = np.squeeze(W1[b])
        w2 = np.squeeze(W2[b])

        # 保存原始权重，后续可以重新画图
        np.save(os.path.join(npy_dir, base_name + "_W1.npy"), w1)
        np.save(os.path.join(npy_dir, base_name + "_W2.npy"), w2)

        # 保存热力图
        w1_heatmap = weight_to_heatmap(w1)
        w2_heatmap = weight_to_heatmap(w2)

        cv2.imwrite(os.path.join(W1_dir, base_name + "_W1.png"), w1_heatmap)
        cv2.imwrite(os.path.join(W2_dir, base_name + "_W2.png"), w2_heatmap)

        # 横向拼接：左边 W1，右边 W2
        compare = np.concatenate([w1_heatmap, w2_heatmap], axis=1)
        cv2.imwrite(os.path.join(compare_dir, base_name + "_W1_W2.png"), compare)


def save_all_gated_weights(model, names, save_root, target_size=None):
    """
    自动查找模型里所有包含 last_W1 / last_W2 的模块并保存。
    不依赖具体模块名，比如 net.sbcr / net.fusion / net.neck.xxx 都可以。
    """

    net = model.module if hasattr(model, "module") else model

    for module_name, module in net.named_modules():
        if hasattr(module, "last_W1") and hasattr(module, "last_W2"):
            W1 = module.last_W1
            W2 = module.last_W2

            if W1 is None or W2 is None:
                continue

            safe_module_name = module_name.replace(".", "_")

            save_dir = os.path.join(save_root, safe_module_name)
            print(save_dir)

            save_W1_W2_maps(
                W1=W1,
                W2=W2,
                names=names,
                save_dir=save_dir,
                target_size=target_size
            )