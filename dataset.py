import torchvision.transforms.functional as F
import numpy as np
import random
import os
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class ToTensor(object):
    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': F.to_tensor(image), 'label': F.to_tensor(label)}


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        image, label = data['image'], data['label']
        return {'image': F.resize(image, self.size),
                'label': F.resize(label, self.size, interpolation=InterpolationMode.BICUBIC)}


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.hflip(image), 'label': F.hflip(label)}

        return {'image': image, 'label': label}


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        image, label = data['image'], data['label']

        if random.random() < self.p:
            return {'image': F.vflip(image), 'label': F.vflip(label)}

        return {'image': image, 'label': label}


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = F.normalize(image, self.mean, self.std)
        return {'image': image, 'label': label}


class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, size, mode, num_tasks=8):
        if not image_root.endswith('/'):
            image_root = image_root + '/'
        if not gt_root.endswith('/'):
            gt_root = gt_root + '/'

        self.num_tasks = num_tasks

        # 获取文件名（不含路径）
        image_files = [f for f in os.listdir(image_root) if
                       f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp') or f.endswith('.tif')]

        # 每个任务对应的gt文件夹
        gt_folders = [f'gt_task{task_id}' for task_id in range(num_tasks)]

        # 获取图像文件名（不含扩展名）以便匹配
        image_basenames = {os.path.splitext(f)[0]: f for f in image_files}

        # 构建完整的文件路径
        self.images = []
        self.gts = []
        self.task_ids = []
        self.filenames = []  # ← 新增：存储原始文件名

        # 为每个任务创建数据条目
        for task_id in range(num_tasks):
            gt_folder_path = os.path.join(gt_root, gt_folders[task_id])
            if os.path.exists(gt_folder_path):
                gt_files = [f for f in os.listdir(gt_folder_path) if
                            f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp') or f.endswith('.tif')]

                for gt_file in gt_files:
                    gt_basename = os.path.splitext(gt_file)[0]
                    if gt_basename in image_basenames:
                        self.images.append(os.path.join(image_root, image_basenames[gt_basename]))
                        self.gts.append(os.path.join(gt_folder_path, gt_file))
                        self.task_ids.append(task_id)
                        self.filenames.append(gt_file)  # ← 保存GT文件名（与预测mask对应）

        print(f"找到匹配的图像-掩码对: {len(self.images)} (来自 {num_tasks} 个任务)")
        for task_id in range(num_tasks):
            count = self.task_ids.count(task_id)
            print(f"  任务 {task_id}: {count} 对")

        if mode == 'train':
            self.transform = transforms.Compose([
                Resize((size, size)),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ToTensor(),
                Normalize()
            ])
        else:
            self.transform = transforms.Compose([
                Resize((size, size)),
                ToTensor(),
                Normalize()
            ])

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])
        task_id = self.task_ids[idx]

        data = {'image': image, 'label': label}
        data = self.transform(data)

        # 返回task_id和文件名
        data['task_id'] = task_id
        data['filename'] = self.filenames[idx]  # ← 新增
        return data

    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class TestDataset:
    def __init__(self, image_root, gt_root, size, task_id=0, num_tasks=8):
        if not image_root.endswith('/'):
            image_root = image_root + '/'
        if not gt_root.endswith('/'):
            gt_root = gt_root + '/'

        self.task_id = task_id
        self.num_tasks = num_tasks

        # 获取图像文件
        self.images = [image_root + f for f in os.listdir(image_root)
                       if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp') or f.endswith('.tif')]

        # 获取对应任务的GT文件
        gt_folder = f'gt_task{task_id}'
        gt_path = os.path.join(gt_root, gt_folder)

        if os.path.exists(gt_path):
            self.gts = [f'{gt_path}/{f}' for f in os.listdir(gt_path)
                        if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp') or f.endswith('.tif')]
        else:
            # 如果没有任务特定的文件夹，使用原始mask路径
            self.gts = [gt_root + f for f in os.listdir(gt_root)
                        if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp') or f.endswith('.tif')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)

        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index])
        gt = np.array(gt)

        name = self.images[self.index].split('/')[-1]

        self.index += 1
        return image, gt, name, self.task_id

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')