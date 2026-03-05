from torch.utils import data
import os
from PIL import Image


class EvalDataset(data.Dataset):
    def __init__(self, pred_root, label_root):
        pred_names = os.listdir(pred_root)

        # 获取标签文件的扩展名映射
        label_files = os.listdir(label_root)
        label_map = {os.path.splitext(name)[0]: name for name in label_files}

        self.image_path = []
        self.label_path = []

        for pred_name in pred_names:
            pred_path = os.path.join(pred_root, pred_name)
            base_name = os.path.splitext(pred_name)[0]

            # 查找对应的标签文件
            if base_name in label_map:
                label_name = label_map[base_name]
                label_path = os.path.join(label_root, label_name)

                self.image_path.append(pred_path)
                self.label_path.append(label_path)
            else:
                print(f"Warning: No matching label found for {pred_name}")

    def __getitem__(self, item):
        pred = Image.open(self.image_path[item]).convert('L')
        gt = Image.open(self.label_path[item]).convert('L')
        if pred.size != gt.size:
            pred = pred.resize(gt.size, Image.BILINEAR)
        return pred, gt

    def __len__(self):
        return len(self.image_path)
