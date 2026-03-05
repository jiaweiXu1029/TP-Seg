import os
import argparse
import random
import numpy as np
import torch
import torch.optim as opt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import FullDataset
from SAM2UNet import Net
from tqdm import tqdm
from utils.AvgMeter import AvgMeter
import shutil
from typing import Optional, Dict


parser = argparse.ArgumentParser("TP-Seg")
parser.add_argument("--hiera_path", type=str, required=True,
                    help="path to the sam2 pretrained hiera")
parser.add_argument("--train_image_path", type=str, required=True,
                    help="path to the image that used to train the model")
parser.add_argument("--train_mask_path", type=str, required=True,
                    help="path to the mask file for training")
parser.add_argument("--val_datasets_dir", type=str, required=True,
                    help="directory containing multiple validation datasets")
parser.add_argument('--save_path', type=str, required=True,
                    help="path to store the checkpoint")
parser.add_argument("--epoch", type=int, default=30,
                    help="training epochs")
parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--weight_decay", default=5e-4, type=float)
parser.add_argument("--num_tasks", default=1, type=int, help="number of tasks")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--mode", type=str, default="joint", choices=["joint"],
                    help="training mode (kept simple: joint)")
parser.add_argument("--balance_tasks", action="store_true",
                    help="use class-balanced sampling over task_ids (if dataset exposes task_ids)")
parser.add_argument("--use_task_loss_balance", action="store_true",
                    help="dynamically weight the four heads' losses per-batch")
parser.add_argument("--use_ema", action="store_true", help="use EMA on model")
parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay")
parser.add_argument("--use_amp", action="store_true", help="enable AMP + grad clipping")
parser.add_argument("--use_warmup", action="store_true", help="2-epoch linear warmup before plateau scheduler")
parser.add_argument("--resume", type=str, default=None,
                    help="path to the checkpoint to resume training from")
args = parser.parse_args()

class cal_dice(object):
    def __init__(self):
        self.prediction = []
    def update(self, pred, gt):
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        if torch.is_tensor(gt):
            gt = gt.detach().cpu().numpy()
        score = self.cal(pred, gt)
        self.prediction.append(score)
    def cal(self, y_pred, y_true):
        smooth = 1e-5
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    def show(self):
        if len(self.prediction) == 0:
            return 0.0
        return np.mean(self.prediction)
    def reset(self):
        self.prediction = []

class cal_miou(object):
    def __init__(self):
        self.prediction = []
    def update(self, pred, gt):
        score = self.cal(pred, gt)
        self.prediction.append(score)
    def cal(self, input, target):
        smooth = 1e-5
        input = input > 0.5
        target_ = target > 0.5

        intersection = (input & target_).sum()
        union = (input | target_).sum()

        return (intersection + smooth) / (union + smooth)

    def show(self):
        # 将 Tensor 移动到 CPU 并转换为 NumPy 数组，然后计算均值
        return np.mean(torch.stack(self.prediction).cpu().numpy())  # 修改此行

def structure_loss(pred, mask, alpha=0.5, boundary_weight=2.0):
    weit = 1 + boundary_weight * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * bce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + 1e-8)
    wbce = wbce.mean()

    smooth = 1e-5
    pred_sigmoid = torch.sigmoid(pred)
    intersection = (pred_sigmoid * mask).sum(dim=(2, 3))
    union = pred_sigmoid.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice.mean()

    return alpha * dice_loss + (1 - alpha) * wbce

def ensure_mask_shape_dtype(mask: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.float32:
        mask = mask.float()
    if mask.max() > 1.0:
        mask = (mask > 127.5).float()
    else:
        mask = (mask > 0.5).float()
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return mask

def validate_dataset(model, image_path, mask_path, device, dataset_name="", task_id=0, batch_size=1):
    try:
        temp_mask_base = os.path.join(os.path.dirname(mask_path), f"temp_mask_validation_{dataset_name}_{task_id}")
        temp_mask_task_path = os.path.join(temp_mask_base, f"gt_task{task_id}")
        os.makedirs(temp_mask_task_path, exist_ok=True)

        mask_files = [f for f in os.listdir(mask_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))]

        for mask_file in mask_files:
            source_file = os.path.join(mask_path, mask_file)
            link_path = os.path.join(temp_mask_task_path, mask_file)
            if not os.path.exists(link_path):
                try:
                    os.symlink(source_file, link_path)
                except:
                    shutil.copy2(source_file, link_path)

        val_dataset = FullDataset(image_path, temp_mask_base, 512, mode='val', num_tasks=args.num_tasks)
        for i in range(len(val_dataset.task_ids)):
            val_dataset.task_ids[i] = task_id

        if len(val_dataset) == 0:
            if os.path.exists(temp_mask_base):
                shutil.rmtree(temp_mask_base)
            return 0.0

        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    except Exception as e:
        print(f"创建验证数据集失败 {dataset_name}: {e}")
        return 0.0

    model.eval()
    dice_calculator = cal_dice()
    miou_calculator = cal_miou()  # 添加mIoU计算器

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc=f"验证 {dataset_name} (Task {task_id})"):
            x = batch['image'].to(device)
            target = ensure_mask_shape_dtype(batch['label'].to(device))
            task_ids = torch.full((x.size(0),), task_id, dtype=torch.long, device=device)
            pred0, pred1, pred2, pred3 = model(x, task_ids)
            pred = torch.sigmoid(pred0)
            pred_binary = (pred > 0.5).float()
            dice_calculator.update(pred_binary, target)
            miou_calculator.update(pred_binary, target)  # 更新mIoU

    if os.path.exists(temp_mask_base):
        shutil.rmtree(temp_mask_base)

    dice_score = dice_calculator.show()
    miou_score = miou_calculator.show()  # 获取mIoU得分
    print(f"验证完成 {dataset_name} (Task {task_id})。Dice: {dice_score:.4f}, mIoU: {miou_score:.4f}")
    return dice_score, miou_score  # 返回Dice和mIoU得分

def validate(model, val_datasets_dir, device, batch_size=1, num_tasks=8):
    model.eval()
    dataset_folders = [f for f in os.listdir(val_datasets_dir)
                       if os.path.isdir(os.path.join(val_datasets_dir, f))]
    dataset_folders.sort()

    if not dataset_folders:

        return 0.0

    all_dices = []
    all_mious = []
    dataset_results: Dict[str, Dict] = {}

    for task_id, dataset_folder in enumerate(dataset_folders[:num_tasks]):
        dataset_path = os.path.join(val_datasets_dir, dataset_folder)
        image_path = os.path.join(dataset_path, "image")
        mask_path = os.path.join(dataset_path, "mask")

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            continue


        dice, miou = validate_dataset(model, image_path, mask_path, device, dataset_folder, task_id, batch_size)

        if dice > 0.0:
            all_dices.append(dice)
            all_mious.append(miou)  # 存储mIoU得分
            dataset_results[dataset_folder] = {'dice': dice, 'miou': miou, 'task_id': task_id}

    if all_dices:
        avg_dice = sum(all_dices) / len(all_dices)
        avg_miou = sum(all_mious) / len(all_mious)  # 计算mIoU的平均值
        print("\n所有数据集的验证结果 (使用对应task_id):")
        for dataset, results in dataset_results.items():
            print(f"- {dataset} (Task {results['task_id']}): Dice = {results['dice']:.4f}, mIoU = {results['miou']:.4f}")
        print(f"平均Dice (所有数据集): {avg_dice:.4f}, 平均mIoU (所有数据集): {avg_miou:.4f}")
        return avg_dice, avg_miou
    else:
        print("没有成功验证任何数据集")
        return 0.0, 0.0

class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.detach().clone()
    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            assert name in self.shadow
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
    def apply_shadow(self, model: torch.nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.detach().clone()
            param.data.copy_(self.shadow[name])
    def restore(self, model: torch.nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.data.copy_(self.backup[name])
        self.backup = {}

def build_balanced_sampler_if_needed(dataset: FullDataset) -> Optional[WeightedRandomSampler]:
    if not hasattr(dataset, "task_ids"):
        return None
    task_ids = np.array(dataset.task_ids, dtype=np.int64)
    num = len(task_ids)
    if num == 0:
        return None
    counts = np.bincount(task_ids, minlength=max(task_ids) + 1)
    counts[counts == 0] = 1
    weights = 1.0 / counts[task_ids]
    weights = torch.DoubleTensor(weights)
    sampler = WeightedRandomSampler(weights, num_samples=num, replacement=True)
    return sampler

def run_one_epoch(model, loader, optimizer, device, *,
                  use_task_loss_balance=False,
                  ema_obj: Optional[EMA] = None,
                  amp_enable: bool = False,
                  scaler: Optional[torch.cuda.amp.GradScaler] = None):
    model.train()
    loss_meter = AvgMeter()

    for batch in tqdm(loader, desc="训练中"):
        x = batch['image'].to(device, non_blocking=True)
        target = ensure_mask_shape_dtype(batch['label'].to(device, non_blocking=True))
        task_ids = batch['task_id'].to(device, non_blocking=True).long()

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp_enable):
            p0, p1, p2, p3 = model(x, task_ids)   # [B,1,512,512] x 4

            # 原始多头损失
            L0 = structure_loss(p0, target)
            L1 = structure_loss(p1, target)
            L2 = structure_loss(p2, target)
            L3 = structure_loss(p3, target)

            if use_task_loss_balance:
                with torch.no_grad():
                    ls = torch.tensor([L0.item(), L1.item(), L2.item(), L3.item()],
                                      device=device, dtype=torch.float32)
                    w = (1.0 - ls / (ls.sum() + 1e-8))
                    w = w / (w.sum() + 1e-8)
                loss = w[0]*L0 + w[1]*L1 + w[2]*L2 + w[3]*L3
            else:
                loss = L0 + L1 + L2 + L3

        if amp_enable and scaler is not None:
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if ema_obj is not None:
            ema_obj.update(model)

        loss_meter.update(loss.item(), x.size(0))

    return loss_meter.avg

def main(args):
    torch.manual_seed(args.seed);
    np.random.seed(args.seed);
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    train_dataset = FullDataset(args.train_image_path, args.train_mask_path, 512, mode='train',
                                num_tasks=args.num_tasks)

    sampler = None
    if args.balance_tasks:
        sampler = build_balanced_sampler_if_needed(train_dataset)
        if sampler is not None:
            print("WeightedRandomSample")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=8,
        drop_last=False,
        pin_memory=True
    )
    print(f"训练集大小: {len(train_dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model = Net(num_tasks=args.num_tasks, checkpoint_path=args.hiera_path).to(device)

    optimizer = opt.AdamW([{"params": model.parameters(), "initial_lr": args.lr}],
                          lr=args.lr, weight_decay=args.weight_decay)

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.1,
        patience=6,
        threshold=1e-4,
        threshold_mode='rel',
        cooldown=0,
        min_lr=1e-7
    )

    ema_obj = EMA(model, decay=args.ema_decay) if args.use_ema else None
    scaler = torch.cuda.amp.GradScaler(enabled=(args.use_amp and torch.cuda.is_available()))
    os.makedirs(args.save_path, exist_ok=True)

    start_epoch = 0
    best_val_dice = 0.0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_dice = checkpoint.get('best_val_dice', 0.0)
        print(f"从epoch {checkpoint['epoch']} 恢复训练")
        print(f"   最佳Dice: {best_val_dice:.4f}")
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("已恢复 ReduceLROnPlateau 调度器状态。")

    print(f"\n训练配置：")
    print(f"- 模式: 联合训练（{args.mode}）")
    print(f"- balance_tasks(采样均衡): {args.balance_tasks}")
    print(f"- use_task_loss_balance: {args.use_task_loss_balance}")
    print(f"- use_ema: {args.use_ema} (decay={args.ema_decay})")
    print(f"- use_amp: {args.use_amp}")
    print(f"- use_warmup: {args.use_warmup}")
    print(f"- 学习率自适应策略: ReduceLROnPlateau(patience=6, factor=0.1)")

    def get_lr(optim_):
        return optim_.param_groups[0]['lr']

    for epoch in range(start_epoch, args.epoch):
        if args.use_warmup and epoch < 2:
            warm = 0.5 + 0.5 * (epoch + 1) / 2.0  # 0.75 -> 1.0
            for g in optimizer.param_groups:
                base_lr = g.get('initial_lr', args.lr)
                g['lr'] = base_lr * warm

        print(f"\nEpoch {epoch + 1}/{args.epoch}")
        print("-" * 20)
        print(f"当前学习率: {get_lr(optimizer):.6g}")

        loss_avg = run_one_epoch(
            model, train_dataloader, optimizer, device,
            use_task_loss_balance=args.use_task_loss_balance,
            ema_obj=ema_obj,
            amp_enable=args.use_amp and torch.cuda.is_available(),
            scaler=scaler
        )
        if ema_obj is not None:
            ema_obj.apply_shadow(model)
        val_dice, val_miou = validate(model, args.val_datasets_dir, device, args.batch_size, args.num_tasks)
        if ema_obj is not None:
            ema_obj.restore(model)
        print(f"当前Dice评分: {val_dice:.4f}, 当前mIoU评分: {val_miou:.4f}")

        prev_lr = get_lr(optimizer)
        scheduler.step(val_dice)
        new_lr = get_lr(optimizer)
        if new_lr < prev_lr:
            print(f"[{prev_lr:.6g} -> {new_lr:.6g}")

        checkpoint_path_latest = os.path.join(args.save_path, 'SAM2-UNet-latest.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_dice': best_val_dice,
        }, checkpoint_path_latest)

        if val_dice > best_val_dice + 1e-6:
            best_val_dice = val_dice
            checkpoint_path_best = os.path.join(args.save_path, 'SAM2-UNet-best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_dice': best_val_dice,
            }, checkpoint_path_best)
            print(f'   Dice: {best_val_dice:.4f}')
        else:
            print(f'   Dice: {val_dice:.4f} (最佳: {best_val_dice:.4f})')

    print("=" * 80)
    print("=" * 80)

if __name__ == "__main__":
    main(args)
