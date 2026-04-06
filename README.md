<div align="center">

<h1>TP-Seg: Task-Prototype Framework for Unified Medical Lesion Segmentation</h1>
  

**CVPR,2026**<br>
Jiawei Xu, Qiangqiang Zhou, Dandan Zhu, Yong Chen, Yugen Yi, [Xiaoqi Zhao](https://github.com/Xiaoqi-Zhao-DLUT)

</div>

## Overview
Building a unified model with a single set of parameters to efficiently handle diverse types of medical lesion segmentation has become a crucial objective for AI-assisted diagnosis. Existing unified segmentation approaches typically rely on shared encoders across heterogeneous tasks and modalities, which often leads to feature entanglement, gradient interference, and suboptimal lesion discrimination. In this work, we propose TP-Seg, a task-prototype framework for unified medical lesion segmentation. On one hand, the task-conditioned adapter effectively balances shared and task-specific representations through a dual-path expert structure, enabling adaptive feature extraction across diverse medical imaging modalities and lesion types. On the other hand, the prototype-guided task decoder introduces learnable task prototypes as semantic anchors and employs a cross-attention mechanism to achieve fine-grained modeling of task-specific foreground and background semantics. Without bells and whistles, TP-Seg consistently outperforms specialized, general and unified segmentation methods across 8 different medical lesion segmentation tasks covering multiple imaging modalities, demonstrating strong generalization, scalability and clinical applicability.

## Framework

<p align="center">
  <img src="https://raw.githubusercontent.com/jiaweiXu1029/TP-Seg/main/image/Framework.png" width="800"><br>
  <em>Figure 1. Overall architecture of the proposed TP-Seg framework for unified medical lesion segmentation. Each input image, together with its task embedding, is processed by the task-conditioned routing block (TCRB) for feature extraction, followed by the prototype-guided task decoder (PGTD) for task-aware decoding and final lesion prediction.</em>
</p>

---

## Visual Comparison

<p align="center">
  <img src="https://raw.githubusercontent.com/jiaweiXu1029/TP-Seg/main/image/Quantitative.png" width="800"><br>
  <em>Figure 2. Visual comparison of TP-Seg with other unified models, <div align="center">
including <a href="https://github.com/Xiaoqi-Zhao-DLUT/Spider-UniCDSeg">Spider</a>, SAM2-UNet, SegGPT and <a href="https://github.com/DUT-CSJ/SR-ICL">SR-ICL</a>, across the 8 medical lesion segmentation tasks.
</div></em>
</p>

---

## Performance

<p align="center">
  <img src="https://raw.githubusercontent.com/jiaweiXu1029/TP-Seg/main/image/Performance.png" width="800"><br>
</p>

---

## Datasets
<p align="center">
  <img src="https://raw.githubusercontent.com/jiaweiXu1029/TP-Seg/main/image/datasets.png" width="800"><br>
</p>

---
## Prediction Maps
[link](https://drive.google.com/file/d/1F6sVp6_Vf3J8a93sWq1eEoAWFZoRi_-c/view?usp=drive_link).

---
## Usage
1.First, download the corresponding [datasets](https://github.com/DUT-CSJ/SR-ICL)（  

> **⚠️ Note** In practice, we obtained the permissions and download rights for each dataset individually.  
> The link provided here is only for reference.）.
> and the [pretrained weights](https://github.com/facebookresearch/sam2).

2.Then run train.py to train the model and obtain the trained weights.
python train.py \
  --hiera_path path/sam2_hiera_large.pt \
  --train_image_path ～/train/image \
  --train_mask_path ～/train/mask \
  --val_datasets_dir ～ \
  --save_path ～ \
  --num_tasks 8 \
  --epoch 30 \
  --batch_size 8 \
  --lr 0.0005 \

3.Finally, run test.py to perform testing using the trained weights.

## Citing TP-Seg

If you find **TP-Seg** useful in your research or work, please consider citing our paper:

```bibtex
@article{xu2026tp,
  title={TP-Seg: Task-Prototype Framework for Unified Medical Lesion Segmentation},
  author={Xu, Jiawei and Zhou, Qiangqiang and Zhu, Dandan and Chen, Yong and Yi, Yugen and Zhao, Xiaoqi},
  journal={arXiv preprint arXiv:2604.00684},
  year={2026}
}
