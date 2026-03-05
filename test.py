import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import FullDataset
from SAM2UNet import Net
from tqdm import tqdm
import cv2
import shutil
import requests
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

parser = argparse.ArgumentParser("SAM2-UNet Testing with Prototype Diagnostics")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--test_datasets_dir", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="./test_results")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--num_tasks", type=int, default=8)
parser.add_argument("--save_predictions", action="store_true")
parser.add_argument("--visualize_prototypes", action="store_true",
                    help="是否可视化原型(仅当模型有PGTD时)")
parser.add_argument("--ablation_mode", type=str, default="none",
                    choices=["none", "compare", "disable_proto"],
                    help="消融实验模式: none=正常测试, compare=对比有无原型, disable_proto=禁用原型测试")
args = parser.parse_args()

class MetricsCalculator:
    def __init__(self, num_classes=2):
        """
        num_classes: 类别数量（默认2：背景和前景）
        """
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.dice_scores = []
        self.iou_scores = []
        # 添加混淆矩阵累计器用于计算mIOU
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred, gt):
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()
        if torch.is_tensor(gt):
            gt = gt.detach().cpu().numpy()

        pred_flat = pred.flatten()
        gt_flat = gt.flatten()

        # 原有的Dice计算
        intersection = np.sum(pred_flat * gt_flat)
        dice = (2.0 * intersection + 1e-5) / (np.sum(pred_flat) + np.sum(gt_flat) + 1e-5)
        self.dice_scores.append(dice)

        # 原有的IoU计算
        union = np.sum(pred_flat) + np.sum(gt_flat) - intersection
        iou = (intersection + 1e-5) / (union + 1e-5)
        self.iou_scores.append(iou)

        # 更新混淆矩阵用于mIOU计算
        # 确保pred和gt都是二值的（0或1）
        pred_binary = (pred_flat > 0.5).astype(np.int64)
        gt_binary = gt_flat.astype(np.int64)

        # 计算混淆矩阵
        for pred_class in range(self.num_classes):
            for gt_class in range(self.num_classes):
                self.confusion_matrix[pred_class, gt_class] += np.sum(
                    (pred_binary == pred_class) & (gt_binary == gt_class)
                )

    def compute_miou(self):
        """
        从混淆矩阵计算mIOU (mean Intersection over Union)
        mIOU = mean(TP / (TP + FP + FN)) for each class
        """
        iou_per_class = []

        for i in range(self.num_classes):
            # TP: 对角线元素
            tp = self.confusion_matrix[i, i]

            # FP: 该类预测的总数 - TP
            fp = np.sum(self.confusion_matrix[i, :]) - tp

            # FN: 该类真值的总数 - TP
            fn = np.sum(self.confusion_matrix[:, i]) - tp

            # IoU for this class
            denom = tp + fp + fn
            if denom > 0:
                iou = tp / denom
                iou_per_class.append(iou)
            else:
                # 如果该类别不存在，可以选择跳过或计为0
                iou_per_class.append(np.nan)

        # 计算平均IoU（忽略NaN值）
        iou_per_class = np.array(iou_per_class)
        valid_ious = iou_per_class[~np.isnan(iou_per_class)]

        if len(valid_ious) > 0:
            miou = np.mean(valid_ious)
        else:
            miou = 0.0

        return miou, iou_per_class

    def show(self):
        if len(self.dice_scores) == 0:
            return 0.0
        return np.mean(self.dice_scores)

    def get_metrics(self):
        miou, iou_per_class = self.compute_miou()

        return {
            'dice': np.mean(self.dice_scores) if self.dice_scores else 0.0,
            'iou': np.mean(self.iou_scores) if self.iou_scores else 0.0,
            'miou': miou,
            'iou_per_class': iou_per_class.tolist(),  # 转换为列表以便序列化
            'num_samples': len(self.dice_scores)
        }

@torch.no_grad()
def diagnose_prototypes(model, save_dir):
    task_names = ['WA', 'BT', 'ADC', 'TN', 'CP', 'LI', 'BL', 'SL']
    pgtd_modules = []
    for name in ['pgtd1', 'pgtd2', 'pgtd3']:
        if hasattr(model, name):
            pgtd_modules.append((name, getattr(model, name)))
        elif hasattr(model, 'module') and hasattr(model.module, name):
            pgtd_modules.append((name, getattr(model.module, name)))

    if not pgtd_modules:
        print("WARNING: Model has no pgtd modules, skipping prototype diagnosis")
        return

    print("\n" + "=" * 80)
    print("Prototype Value Diagnosis")
    print("=" * 80)

    for pgtd_name, pgtd in pgtd_modules:
        print(f"\n{'=' * 40}")
        print(f" {pgtd_name.upper()}")
        print(f"{'=' * 40}")

        protos = pgtd.multi_scale_protos.cpu()  # [T,S,2,C]

        if protos.abs().sum() < 1e-6:
            print("CRITICAL: Prototypes are all zeros! Not initialized or updated")
            continue
        norms = protos.norm(dim=-1)  # [T,S,2]
        print(f"\nPrototype L2 Norm Distribution:")
        print(f"  Mean: {norms.mean():.4f} +/- {norms.std():.4f}")
        print(f"  Range: [{norms.min():.4f}, {norms.max():.4f}]")

        if norms.mean() < 0.1:
            print("  WARNING: Norms too small (<0.1), may not be properly normalized")
        elif norms.std() > 0.5:
            print("  WARNING: Large norm variance (std>0.5), uneven prototype updates")

        init_status = pgtd.proto_initialized.cpu().numpy()
        print(f"\nInitialization Status: {init_status.sum()}/{len(init_status)} tasks initialized")
        if not init_status.all():
            uninit_tasks = np.where(~init_status)[0]
            uninit_names = [task_names[i] for i in uninit_tasks if i < len(task_names)]
            print(f"  WARNING: Uninitialized tasks: {uninit_names} {uninit_tasks}")

        # 4. Check fg-bg similarity (using scale 0)
        print(f"\nFG-BG Separation Analysis:")
        fg_bg_sims = []
        for t in range(protos.shape[0]):
            fg = F.normalize(protos[t, 0, 0].unsqueeze(0), dim=-1)
            bg = F.normalize(protos[t, 0, 1].unsqueeze(0), dim=-1)
            cos_sim = F.cosine_similarity(fg, bg, dim=-1).item()
            fg_bg_sims.append(cos_sim)

        avg_cos = np.mean(fg_bg_sims)
        separation = 1 - avg_cos
        print(f"  FG-BG Cosine Similarity: {avg_cos:.4f}")
        print(f"  FG-BG Separation (1-cos): {separation:.4f}")

        if avg_cos > 0.9:
            print("  CRITICAL: FG and BG prototypes almost identical (cos>0.9), no discriminative info learned")
        elif avg_cos < -0.5:
            print("  WARNING: FG and BG prototypes opposite directions (cos<-0.5, abnormal)")
        elif separation > 0.3:
            print("  GOOD: FG-BG well separated")

    # 5. Visualization (only visualize pgtd1 as representative)
    if pgtd_modules:
        _, pgtd1 = pgtd_modules[0]
        protos_flat = pgtd1.multi_scale_protos.cpu().reshape(-1, pgtd1.multi_scale_protos.shape[-1]).numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Heatmap: prototype values
        sample_size = min(20, protos_flat.shape[0])
        im1 = ax1.imshow(protos_flat[:sample_size], aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax1.set_title(f'Prototype Value Heatmap (pgtd1 first {sample_size})')
        ax1.set_xlabel('Feature Dimension')
        ax1.set_ylabel('Prototype Index')
        plt.colorbar(im1, ax=ax1)

        # Histogram: value distribution
        ax2.hist(protos_flat.flatten(), bins=50, alpha=0.7, color='blue')
        ax2.set_title('All Prototype Value Distribution (pgtd1)')
        ax2.set_xlabel('Value')
        ax2.set_ylabel('Frequency')
        ax2.axvline(0, color='r', linestyle='--', label='Zero')
        ax2.legend()

        plt.tight_layout()
        save_path = f'{save_dir}/proto_value_distribution.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"\nPrototype value distribution plot saved: {save_path}")

    print("=" * 80 + "\n")

class TestPrototypeVisualizer:
    def __init__(self, model, save_dir):
        self.model = model
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Task name mapping
        self.task_names = ['WA', 'BT', 'ADC', 'TN', 'CP', 'LI', 'BL', 'SL']

        # Collect all PGTD modules
        self.pgtd_modules = []
        for name in ['pgtd1', 'pgtd2', 'pgtd3']:
            if hasattr(model, name):
                self.pgtd_modules.append((name, getattr(model, name)))
            elif hasattr(model, 'module') and hasattr(model.module, name):
                self.pgtd_modules.append((name, getattr(model.module, name)))

        if not self.pgtd_modules:
            raise ValueError("Model does not have pgtd1/pgtd2/pgtd3 modules")

        print(f"Found {len(self.pgtd_modules)} PGTD modules: {[name for name, _ in self.pgtd_modules]}")

    @torch.no_grad()
    @torch.no_grad()
    def visualize_quick(self):
        self._plot_task_similarity_combined()  # FG相似度
        self._plot_bg_task_similarity_combined()  # 新增: BG相似度
        self._plot_fg_bg_separation_combined()
        self._print_all_statistics()

        print(f"\nVisualization saved in: {self.save_dir}")
        print("=" * 60 + "\n")

    @torch.no_grad()
    def _plot_task_similarity_combined(self):
        """Combined task similarity heatmaps for all PGTD modules"""
        num_modules = len(self.pgtd_modules)
        fig, axes = plt.subplots(1, num_modules, figsize=(8 * num_modules, 6))

        # Handle single module case
        if num_modules == 1:
            axes = [axes]

        for idx, (pgtd_name, pgtd) in enumerate(self.pgtd_modules):
            num_tasks = pgtd.num_tasks
            protos = pgtd.multi_scale_protos.cpu()

            # scale0的fg原型
            protos_fg = protos[:, 0, 0, :]
            protos_norm = F.normalize(protos_fg, dim=-1)
            sim_matrix = torch.mm(protos_norm, protos_norm.t()).numpy()

            ax = axes[idx]
            sns.heatmap(sim_matrix, annot=True, fmt='.3f',
                        cmap='YlGnBu',
                        vmin=-0.2, vmax=1,
                        xticklabels=self.task_names[:num_tasks],
                        yticklabels=self.task_names[:num_tasks],
                        cbar_kws={'label': 'Cosine Similarity'},
                        linewidths=0.5, linecolor='white',
                        square=True, ax=ax)

            ax.set_title(f'{pgtd_name.upper()}', fontsize=13, fontweight='bold', pad=10)
            ax.set_xlabel('Task', fontsize=12, fontweight='bold')
            if idx == 0:
                ax.set_ylabel('Task', fontsize=12, fontweight='bold')

        plt.suptitle('Task-wise Prototype Similarity (FG)',
                     fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'{self.save_dir}/task_similarity_combined.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Combined task similarity plot saved")

    @torch.no_grad()
    def _plot_bg_task_similarity_combined(self):
        """Combined task similarity heatmaps for background prototypes across all PGTD modules"""
        num_modules = len(self.pgtd_modules)
        fig, axes = plt.subplots(1, num_modules, figsize=(8 * num_modules, 6))

        # Handle single module case
        if num_modules == 1:
            axes = [axes]

        for idx, (pgtd_name, pgtd) in enumerate(self.pgtd_modules):
            num_tasks = pgtd.num_tasks
            protos = pgtd.multi_scale_protos.cpu()

            # scale0的BG原型 (索引为1)
            protos_bg = protos[:, 0, 1, :]
            protos_norm = F.normalize(protos_bg, dim=-1)
            sim_matrix = torch.mm(protos_norm, protos_norm.t()).numpy()

            ax = axes[idx]
            sns.heatmap(sim_matrix, annot=True, fmt='.3f',
                        cmap='YlOrRd',  # 使用不同的配色方案区分BG
                        vmin=-0.2, vmax=1,
                        xticklabels=self.task_names[:num_tasks],
                        yticklabels=self.task_names[:num_tasks],
                        cbar_kws={'label': 'Cosine Similarity'},
                        linewidths=0.5, linecolor='white',
                        square=True, ax=ax)

            ax.set_title(f'{pgtd_name.upper()}', fontsize=13, fontweight='bold', pad=10)
            ax.set_xlabel('Task', fontsize=12, fontweight='bold')
            if idx == 0:
                ax.set_ylabel('Task', fontsize=12, fontweight='bold')

        plt.suptitle('Task-wise Prototype Similarity (BG)',
                     fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'{self.save_dir}/bg_task_similarity_combined.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Combined BG task similarity plot saved")

    @torch.no_grad()
    def _plot_fg_bg_separation_combined(self):
        """Combined FG-BG separation bar charts for all PGTD modules"""
        num_modules = len(self.pgtd_modules)
        fig, axes = plt.subplots(1, num_modules, figsize=(10 * num_modules, 6))

        # Handle single module case
        if num_modules == 1:
            axes = [axes]

        for idx, (pgtd_name, pgtd) in enumerate(self.pgtd_modules):
            protos = pgtd.multi_scale_protos.cpu()
            num_tasks = protos.shape[0]

            fg_bg_cosines = []
            for t in range(num_tasks):
                fg = F.normalize(protos[t, 0, 0].unsqueeze(0), dim=-1)
                bg = F.normalize(protos[t, 0, 1].unsqueeze(0), dim=-1)
                cos_sim = F.cosine_similarity(fg, bg).item()
                fg_bg_cosines.append(cos_sim)

            fg_bg_cosines = np.array(fg_bg_cosines)
            separations = 1 - fg_bg_cosines

            ax = axes[idx]

            # Color gradient
            norm = plt.Normalize(vmin=0, vmax=1.5)
            cmap = plt.cm.YlGnBu
            colors_sep = [cmap(norm(val)) for val in separations]

            bars = ax.bar(range(num_tasks), separations, color=colors_sep,
                          edgecolor='white', linewidth=1.2, alpha=0.9)

            # Reference lines
            ax.axhline(y=1.2, color='#2E7D32', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
            ax.axhline(y=1.0, color='#66BB6A', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)
            ax.axhline(y=0.5, color='#FFA726', linestyle='--', linewidth=1.5, alpha=0.5, zorder=1)

            # Annotate values
            for i, (bar, val) in enumerate(zip(bars, separations)):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05,
                        f'{val:.3f}', ha='center', va='bottom',
                        fontsize=9, fontweight='bold', color='#424242')

            ax.set_xticks(range(num_tasks))
            ax.set_xticklabels(self.task_names[:num_tasks], fontsize=11, fontweight='bold')
            ax.set_xlabel('Task', fontsize=12, fontweight='bold')
            if idx == 0:
                ax.set_ylabel('FG-BG Separation (1-cosine)', fontsize=12, fontweight='bold')
            ax.set_title(f'{pgtd_name.upper()}', fontsize=13, fontweight='bold', pad=10)
            ax.set_ylim(-0.1, max(separations.max() + 0.25, 1.5))

            # Clean grid
            ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=0.8, color='gray')
            ax.set_axisbelow(True)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Statistics
            avg_sep = separations.mean()
            std_sep = separations.std()
            stats_text = f"Avg: {avg_sep:.3f}±{std_sep:.3f}"
            ax.text(0.5, 0.95, stats_text, transform=ax.transAxes,
                    fontsize=9, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='#E3F2FD',
                              alpha=0.8, edgecolor='#90CAF9', linewidth=1))

        plt.suptitle('FG-BG Separation Score',
                     fontsize=15, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'{self.save_dir}/fg_bg_separation_combined.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Combined FG-BG separation plot saved")

    def _print_all_statistics(self):
        """Print statistics for all PGTD modules"""
        for pgtd_name, pgtd in self.pgtd_modules:
            protos = pgtd.multi_scale_protos.cpu()
            num_tasks = protos.shape[0]

            print(f"\n{pgtd_name.upper()} Statistics:")

            # FG任务间相似度
            fg_sims = []
            for t1 in range(num_tasks):
                for t2 in range(t1 + 1, num_tasks):
                    p1 = F.normalize(protos[t1, 0, 0].unsqueeze(0), dim=-1)
                    p2 = F.normalize(protos[t2, 0, 0].unsqueeze(0), dim=-1)
                    sim = F.cosine_similarity(p1, p2).item()
                    fg_sims.append(sim)

            avg_fg_sim = np.mean(fg_sims)
            print(f"  Avg inter-task similarity (FG): {avg_fg_sim:.4f}", end="")
            if avg_fg_sim > 0.95:
                print(f" WARNING: Too high (>0.95)!")
            else:
                print()

            # 新增: BG任务间相似度
            bg_sims = []
            for t1 in range(num_tasks):
                for t2 in range(t1 + 1, num_tasks):
                    p1 = F.normalize(protos[t1, 0, 1].unsqueeze(0), dim=-1)
                    p2 = F.normalize(protos[t2, 0, 1].unsqueeze(0), dim=-1)
                    sim = F.cosine_similarity(p1, p2).item()
                    bg_sims.append(sim)

            avg_bg_sim = np.mean(bg_sims)
            print(f"  Avg inter-task similarity (BG): {avg_bg_sim:.4f}", end="")
            if avg_bg_sim > 0.95:
                print(f" WARNING: Too high (>0.95)!")
            else:
                print()

            # FG-BG separation (保持原有代码)
            fg_bg_sims = []
            for t in range(num_tasks):
                fg = F.normalize(protos[t, 0, 0].unsqueeze(0), dim=-1)
                bg = F.normalize(protos[t, 0, 1].unsqueeze(0), dim=-1)
                sim = F.cosine_similarity(fg, bg).item()
                fg_bg_sims.append(sim)

            avg_cos = np.mean(fg_bg_sims)
            avg_sep = 1 - avg_cos
            print(f"  Avg FG-BG cosine similarity: {avg_cos:.4f}")
            print(f"  Avg FG-BG separation: {avg_sep:.4f}", end="")
            if avg_sep < 0.1:
                print(f" WARNING: Too low (<0.1)!")
            elif avg_sep > 0.5:
                print(f" GOOD: Excellent (>0.5)!")
            elif avg_sep > 0.3:
                print(f" GOOD: Good (>0.3)")
            else:
                print()


class PrototypeSwitch:
    def __init__(self, model, mode='full'):
        self.model = model
        self.mode = mode
        self.original_state = {}

        self.pgtd_modules = []
        for name in ['pgtd1', 'pgtd2', 'pgtd3']:
            if hasattr(model, name):
                self.pgtd_modules.append(getattr(model, name))
            elif hasattr(model, 'module') and hasattr(model.module, name):
                self.pgtd_modules.append(getattr(model.module, name))

        if not self.pgtd_modules:
            print("⚠️ 模型中未找到pgtd1/pgtd2/pgtd3模块")

    def __enter__(self):
        if self.mode == 'disabled':
            for pgtd in self.pgtd_modules:
                self.original_state[id(pgtd)] = {
                    'proto_lambda': pgtd.proto_lambda,
                    'forward': pgtd.forward
                }
                pgtd.proto_lambda = 0.0
                pgtd.forward = self._make_proto_free_forward(pgtd)

        elif self.mode == 'signal_only':
            for pgtd in self.pgtd_modules:
                self.original_state[id(pgtd)] = {'proto_lambda': pgtd.proto_lambda}
                pgtd.proto_lambda = 0.0

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for pgtd in self.pgtd_modules:
            if id(pgtd) in self.original_state:
                state = self.original_state[id(pgtd)]
                pgtd.proto_lambda = state['proto_lambda']
                if 'forward' in state:
                    pgtd.forward = state['forward']

    @staticmethod
    def _make_proto_free_forward(pgtd):
        original_forward = pgtd.forward.__func__
        def proto_free_forward(self, x_low, x_high, task_ids, task_emb, gt_masks=None):
            B = x_low.size(0)
            if x_high.shape[-2:] != x_low.shape[-2:]:
                x_high = F.interpolate(x_high, size=x_low.shape[-2:],
                                       mode="bilinear", align_corners=True)
            xl = self.proj_low(x_low)
            xh = self.proj_high(x_high)
            x = self.fuse(torch.cat([xh, xl], dim=1))
            B, C, H, W = x.shape
            y = self.scale_encoders[0](x)
            k = 3
            x_unf = F.unfold(x, kernel_size=k, padding=1)
            HW = H * W
            x_unf = x_unf.view(B, C, 9, HW)

            identity9 = self.identity_kernel.view(self.c_mid, 9).unsqueeze(0)
            y = torch.einsum('cr,bcrw->bcw', identity9[0], x_unf)
            y = y.view(B, C, H, W)

            # 纯 logits
            logits_all = self.head(y)
            idx = task_ids.view(-1, 1, 1, 1).long()
            logit_task = logits_all.gather(1, idx.expand(-1, 1, H, W))
            pred = F.interpolate(logit_task, size=512, mode="bilinear", align_corners=True)
            dummy_scale_weights = torch.ones(B, len(self.scales), device=x.device) / len(self.scales)
            return pred, y, dummy_scale_weights

        return proto_free_forward.__get__(pgtd, type(pgtd))

def ensure_mask_shape_dtype(mask):
    if mask.dtype != torch.float32:
        mask = mask.float()
    if mask.max() > 1.0:
        mask = (mask > 127.5).float()
    else:
        mask = (mask > 0.5).float()
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return mask

def test_single_dataset(model, image_path, mask_path, device, dataset_name, task_id,
                        save_dir=None, enable_proto=True, mode_suffix=""):
    """
    enable_proto: 是否启用原型机制
    mode_suffix: 结果文件名后缀（用于区分有/无原型）
    """
    temp_mask_base = os.path.join(
        os.path.dirname(mask_path),
        f"temp_mask_test_{dataset_name}_{task_id}"
    )
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

    test_dataset = FullDataset(
        image_path, temp_mask_base, 512,
        mode='val',
        num_tasks=args.num_tasks
    )

    for i in range(len(test_dataset.task_ids)):
        test_dataset.task_ids[i] = task_id

    if len(test_dataset) == 0:
        if os.path.exists(temp_mask_base):
            shutil.rmtree(temp_mask_base)
        return None

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    model.eval()
    dice_calculator = MetricsCalculator()

    # 使用完整的原型开关（mode='disabled'）
    switch_mode = 'full' if enable_proto else 'disabled'
    with PrototypeSwitch(model, mode=switch_mode):
        with torch.no_grad():
            desc = f"测试 {dataset_name} (Task {task_id}){mode_suffix}"
            for batch in tqdm(test_loader, desc=desc):
                x = batch['image'].to(device)
                target = ensure_mask_shape_dtype(batch['label'].to(device))
                task_ids = torch.full((x.size(0),), task_id, dtype=torch.long, device=device)

                pred0, pred1, pred2, pred3 = model(x, task_ids)
                pred = torch.sigmoid(pred0)
                pred_binary = (pred > 0.5).float()

                dice_calculator.update(pred_binary, target)

                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    for b in range(x.size(0)):
                        pred_img = (pred_binary[b, 0].cpu().numpy() * 255).astype(np.uint8)
                        # 直接用数据集返回的文件名
                        base = batch["filename"][b]  # 获取原始文件名
                        save_path = os.path.join(save_dir, f"{base}{mode_suffix}")
                        cv2.imwrite(save_path, pred_img)

    if os.path.exists(temp_mask_base):
        shutil.rmtree(temp_mask_base)

    metrics = dice_calculator.get_metrics()
    dice_score = metrics['dice']
    miou_score = metrics['miou']

    # 修改打印信息，添加mIOU
    print(f"测试完成 {dataset_name} (Task {task_id}){mode_suffix}。"
          f"Dice: {dice_score:.4f}, mIOU: {miou_score:.4f}")

    return metrics

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(num_tasks=args.num_tasks).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型来自epoch {checkpoint['epoch']}, 最佳验证Dice: {checkpoint.get('best_val_dice', 'N/A'):.4f}")

    # ===== 显示消融模式 =====
    print(f"\n消融实验模式: {args.ablation_mode}")
    if args.ablation_mode == "compare":
        print("  将对比【有原型】vs【无原型】的性能差异")
    elif args.ablation_mode == "disable_proto":
        print("  将禁用原型机制进行测试")

    if args.visualize_prototypes:
        try:
            proto_vis_dir = os.path.join(args.output_dir, "prototype_diagnostics")
            diagnose_prototypes(model, proto_vis_dir)
            visualizer = TestPrototypeVisualizer(model, proto_vis_dir)
            visualizer.visualize_quick()

        except Exception as e:
            import traceback
            traceback.print_exc()

    dataset_folders = sorted([
        f for f in os.listdir(args.test_datasets_dir)
        if os.path.isdir(os.path.join(args.test_datasets_dir, f))
    ])

    if not dataset_folders:
        return

    print(f"\n开始测试所有数据集...")
    print("=" * 80)

    all_results = {}
    all_dices = []

    # ===== 新增：对比模式的结果存储 =====
    if args.ablation_mode == "compare":
        comparison_results = {}

    for task_id, dataset_folder in enumerate(dataset_folders[:args.num_tasks]):
        dataset_path = os.path.join(args.test_datasets_dir, dataset_folder)
        image_path = os.path.join(dataset_path, "image")
        mask_path = os.path.join(dataset_path, "mask")

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"警告: {dataset_folder} 中缺少image或mask文件夹,跳过")
            continue

        print(f"\n{'=' * 80}")
        print(f"测试数据集 '{dataset_folder}' (Task {task_id})")
        print(f"{'=' * 80}")

        save_dir = None
        if args.save_predictions:
            save_dir = os.path.join(args.output_dir, dataset_folder)

        if args.ablation_mode == "compare":
            print("\n【1/2】测试：启用原型")
            save_dir_with = os.path.join(args.output_dir, "有原型", dataset_folder)
            results_with_proto = test_single_dataset(
                model, image_path, mask_path, device,
                dataset_folder, task_id,
                save_dir=save_dir_with,
                enable_proto=True,
                mode_suffix=""  # 空后缀 -> 保持原始文件名不变
            )

            print("\n【2/2】测试：禁用原型")
            save_dir_without = os.path.join(args.output_dir, "无原型", dataset_folder)
            results_without_proto = test_single_dataset(
                model, image_path, mask_path, device,
                dataset_folder, task_id,
                save_dir=save_dir_without,
                enable_proto=False,
                mode_suffix=""  # 空后缀 -> 保持原始文件名不变
            )

            if results_with_proto and results_without_proto:
                diff = results_with_proto['dice'] - results_without_proto['dice']
                miou_diff = results_with_proto['miou'] - results_without_proto['miou']

                comparison_results[dataset_folder] = {
                    'task_id': task_id,
                    'with_proto': results_with_proto['dice'],
                    'without_proto': results_without_proto['dice'],
                    'diff': diff,
                    'miou_with_proto': results_with_proto['miou'],
                    'miou_without_proto': results_without_proto['miou'],
                    'miou_diff': miou_diff
                }

                print(f"\n原型贡献分析:")
                print(f"  有原型:   Dice = {results_with_proto['dice']:.4f}, mIOU = {results_with_proto['miou']:.4f}")
                print(
                    f"  无原型:   Dice = {results_without_proto['dice']:.4f}, mIOU = {results_without_proto['miou']:.4f}")
                print(f"  Dice差异: {diff:+.4f} ({'提升✅' if diff > 0 else '下降❌'})")
                print(f"  mIOU差异: {miou_diff:+.4f} ({'提升✅' if miou_diff > 0 else '下降❌'})")

                if abs(diff) < 0.01 and abs(miou_diff) < 0.01:
                    print(f"  ⚠️ 原型影响微弱（<1%），可能冗余")
                elif diff < -0.02 or miou_diff < -0.02:
                    print(f"  ❌ 原型降低了性能（>2%），建议禁用")
                elif diff > 0.02 or miou_diff > 0.02:
                    print(f"  ✅ 原型显著提升性能（>2%）")

                all_dices.append(results_with_proto['dice'])
                all_results[dataset_folder] = results_with_proto
                all_results[dataset_folder]['task_id'] = task_id

        elif args.ablation_mode == "disable_proto":
            # 禁用原型模式
            results = test_single_dataset(
                model, image_path, mask_path, device,
                dataset_folder, task_id, save_dir,
                enable_proto=False, mode_suffix=" [原型已禁用]"
            )
            if results and results['dice'] > 0.0:
                all_dices.append(results['dice'])
                all_results[dataset_folder] = results
                all_results[dataset_folder]['task_id'] = task_id

        else:
            # 正常模式
            results = test_single_dataset(
                model, image_path, mask_path, device,
                dataset_folder, task_id, save_dir,
                enable_proto=True
            )
            if results and results['dice'] > 0.0:
                all_dices.append(results['dice'])
                all_results[dataset_folder] = results
                all_results[dataset_folder]['task_id'] = task_id

    # ===== 汇总结果 =====
    # ===== 汇总结果 =====
    if all_dices:
        avg_dice = sum(all_dices) / len(all_dices)
        avg_miou = sum(r['miou'] for r in all_results.values()) / len(all_results)

        print("\n" + "=" * 80)
        print("所有数据集的测试结果:")
        print("=" * 80)
        for dataset, results in all_results.items():
            print(f"- {dataset} (Task {results['task_id']}): "
                  f"Dice = {results['dice']:.4f}, "
                  f"IoU = {results['iou']:.4f}, "
                  f"mIOU = {results['miou']:.4f}")
        print(f"\n平均Dice: {avg_dice:.4f}")
        print(f"平均mIOU: {avg_miou:.4f}")

        os.makedirs(args.output_dir, exist_ok=True)
        results_file = os.path.join(args.output_dir, "test_results.txt")
        with open(results_file, 'w') as f:
            f.write("所有数据集的测试结果:\n")
            f.write("=" * 80 + "\n")
            for dataset, results in all_results.items():
                f.write(f"{dataset} (Task {results['task_id']}): ")
                f.write(f"Dice = {results['dice']:.4f}, "
                        f"IoU = {results['iou']:.4f}, "
                        f"mIOU = {results['miou']:.4f}\n")
            f.write(f"\n平均Dice: {avg_dice:.4f}\n")
            f.write(f"平均mIOU: {avg_miou:.4f}\n")

            # ===== 保存对比结果 =====
        if args.ablation_mode == "compare" and comparison_results:
          comparison_file = os.path.join(args.output_dir, "ablation_comparison.txt")
        with open(comparison_file, 'w') as f:
            f.write("原型消融实验对比结果\n")
            f.write("=" * 100 + "\n")
            f.write(f"{'数据集':<30} {'Task':<6} {'Dice(有)':<10} {'Dice(无)':<10} "
                    f"{'mIOU(有)':<10} {'mIOU(无)':<10} {'Dice差异':<10} {'mIOU差异':<10} {'结论'}\n")
            f.write("-" * 100 + "\n")

            total_with = 0
            total_without = 0
            total_miou_with = 0
            total_miou_without = 0

            for dataset, comp in comparison_results.items():
                status = ""
                if comp['diff'] > 0.02 or comp['miou_diff'] > 0.02:
                    status = "✅显著提升"
                elif comp['diff'] < -0.02 or comp['miou_diff'] < -0.02:
                    status = "❌降低性能"
                elif abs(comp['diff']) < 0.01 and abs(comp['miou_diff']) < 0.01:
                    status = "⚠️影响微弱"
                else:
                    status = "轻微差异"

                f.write(f"{dataset:<30} {comp['task_id']:<6} "
                        f"{comp['with_proto']:.4f}    {comp['without_proto']:.4f}    "
                        f"{comp['miou_with_proto']:.4f}    {comp['miou_without_proto']:.4f}    "
                        f"{comp['diff']:+.4f}    {comp['miou_diff']:+.4f}    {status}\n")

                total_with += comp['with_proto']
                total_without += comp['without_proto']
                total_miou_with += comp['miou_with_proto']
                total_miou_without += comp['miou_without_proto']

            avg_with = total_with / len(comparison_results)
            avg_without = total_without / len(comparison_results)
            avg_diff = avg_with - avg_without
            avg_miou_with = total_miou_with / len(comparison_results)
            avg_miou_without = total_miou_without / len(comparison_results)
            avg_miou_diff = avg_miou_with - avg_miou_without

            f.write("-" * 100 + "\n")
            f.write(f"{'平均':<30} {'':<6} "
                    f"{avg_with:.4f}    {avg_without:.4f}    "
                    f"{avg_miou_with:.4f}    {avg_miou_without:.4f}    "
                    f"{avg_diff:+.4f}    {avg_miou_diff:+.4f}\n")

            f.write("\n总结:\n")
            if avg_diff > 0.02 or avg_miou_diff > 0.02:
                f.write("  ✅ 原型机制整体上显著提升了性能（>2%），建议保留\n")
            elif avg_diff < -0.02 or avg_miou_diff < -0.02:
                f.write("  ❌ 原型机制整体上降低了性能（>2%），建议移除\n")
            elif abs(avg_diff) < 0.01 and abs(avg_miou_diff) < 0.01:
                f.write("  ⚠️ 原型机制影响微弱（<1%），可能冗余，建议简化或移除\n")
            else:
                f.write("  原型机制有轻微影响，需要根据具体任务权衡\n")

        print(f"\n消融对比结果已保存到: {comparison_file}")

        # 终端输出总结
        print("\n" + "=" * 80)
        print("消融实验总结")
        print("=" * 80)
        print(f"平均Dice (有原型):   {avg_with:.4f}")
        print(f"平均Dice (无原型):   {avg_without:.4f}")
        print(f"平均Dice差异:        {avg_diff:+.4f}")
        print(f"平均mIOU (有原型):   {avg_miou_with:.4f}")
        print(f"平均mIOU (无原型):   {avg_miou_without:.4f}")
        print(f"平均mIOU差异:        {avg_miou_diff:+.4f}")

        if avg_diff > 0.02 or avg_miou_diff > 0.02:
            print("\n✅ 结论: 原型机制显著提升性能，建议保留")
        elif avg_diff < -0.02 or avg_miou_diff < -0.02:
            print("\n❌ 结论: 原型机制降低性能，建议移除")
        elif abs(avg_diff) < 0.01 and abs(avg_miou_diff) < 0.01:
            print("\n⚠️ 结论: 原型影响微弱，可能冗余")
        print("=" * 80)

if __name__ == "__main__":
    main(args)

