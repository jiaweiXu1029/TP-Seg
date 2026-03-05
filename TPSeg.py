import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile, clever_format

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, k, s=1, p=0, d=1, use_gn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, k, s, p, d, bias=False)
        self.norm = nn.GroupNorm(1, out_planes) if use_gn else nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, k=(1, 3), p=(0, 1)),
            BasicConv2d(out_channel, out_channel, k=(3, 1), p=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, p=3, d=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, k=(1, 5), p=(0, 2)),
            BasicConv2d(out_channel, out_channel, k=(5, 1), p=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, p=5, d=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, k=(1, 7), p=(0, 3)),
            BasicConv2d(out_channel, out_channel, k=(7, 1), p=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, p=7, d=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, p=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        return self.relu(x_cat + self.conv_res(x))

class TaskLogitFuser(nn.Module):
    def __init__(self, num_tasks):
        super().__init__()
        self.weight = nn.Embedding(num_tasks, 4)
        with torch.no_grad():
            init = torch.tensor([2.0, 0.0, 0.0, 0.0])
            self.weight.weight.copy_(init.unsqueeze(0).repeat(num_tasks, 1))

    def forward(self, outs, task_ids):
        assert len(outs) == 4
        B = outs[0].size(0)
        stack = torch.stack(outs, dim=1)  # [B,4,1,H,W]
        w = torch.softmax(self.weight(task_ids), dim=-1).view(B, 4, 1, 1, 1)
        return (stack * w).sum(dim=1)

class PrototypeGuidedTaskDecoder(nn.Module):
    def __init__(self, c_low=64, c_high=128, c_mid=128, tdim=64,
                 num_tasks=1, M=6, scales=[1],
                 proto_momentum=0.9, proto_lambda=5.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.M = M
        self.scales = scales
        self.c_mid = c_mid
        self.proto_lambda = proto_lambda

        self.proj_low = nn.Sequential(
            nn.Conv2d(c_low, c_mid // 2, 1, bias=False),
            nn.GroupNorm(1, c_mid // 2), nn.GELU()
        )
        self.proj_high = nn.Sequential(
            nn.Conv2d(c_high, c_mid, 1, bias=False),
            nn.GroupNorm(1, c_mid), nn.GELU()
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(c_mid + c_mid // 2, c_mid, 1, bias=False),
            nn.GroupNorm(1, c_mid), nn.GELU()
        )

        self.register_buffer("multi_scale_protos",
                             torch.zeros(num_tasks, len(scales), 2, c_mid))
        self.register_buffer("proto_initialized",
                             torch.zeros(num_tasks, dtype=torch.bool))
        self.register_buffer("proto_momentum", torch.tensor(proto_momentum))

        self.proto_initializer = nn.Sequential(
            nn.Linear(tdim, c_mid * 2 * len(scales)),
            nn.GELU(),
            nn.Linear(c_mid * 2 * len(scales), c_mid * 2 * len(scales))
        )

        self.scale_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c_mid, c_mid, 3, padding=1, groups=max(c_mid // 4, 1)),
                nn.GroupNorm(min(4, c_mid), c_mid),
                nn.GELU(),
                nn.Conv2d(c_mid, c_mid, 1)
            )
        ])
        self.proto_cross_attn = nn.ModuleList([
            nn.MultiheadAttention(c_mid, num_heads=4, batch_first=True)
        ])

        self.proto_fuser = nn.Linear(c_mid * 3, c_mid)
        self.scale_router = nn.Sequential(
            nn.Linear(tdim, len(scales) * 2),
            nn.GELU(),
            nn.Linear(len(scales) * 2, len(scales)),
            nn.Softmax(dim=-1)
        )
        self.adaptive_scale_weights = nn.Parameter(
            torch.ones(len(scales)) / len(scales)
        )
        self.expert_generator = nn.Sequential(
            nn.Linear(c_mid * len(scales), c_mid * 2),
            nn.GELU(),
            nn.Linear(c_mid * 2, M * c_mid * 9)
        )
        self.expert_weight_gen = nn.Sequential(
            nn.Linear(c_mid * len(scales), M),
            nn.Softmax(dim=-1)
        )

        self.register_buffer("identity_kernel", self._make_identity(c_mid))
        self.expert_init_scale = nn.Parameter(torch.tensor(0.3))

        self.sim_temp = nn.Parameter(torch.tensor(0.1))
        self.proto_amplifier = nn.Parameter(torch.tensor(1.0))

        self.head = nn.Sequential(
            nn.Conv2d(c_mid, c_mid, 3, padding=1, groups=c_mid),
            nn.GroupNorm(1, c_mid), nn.GELU(),
            nn.Conv2d(c_mid, num_tasks, 1)
        )

    @staticmethod
    def _make_identity(c):
        w = torch.zeros(c, 1, 3, 3)
        w[:, 0, 1, 1] = 1.0
        return w

    def initialize_prototypes(self, task_ids, task_emb):
        proto_init = self.proto_initializer(task_emb)
        proto_init = proto_init.view(-1, len(self.scales), 2, self.c_mid)

        for b, tid in enumerate(task_ids):
            tid_int = int(tid.item())
            if not self.proto_initialized[tid_int]:
                with torch.no_grad():
                    self.multi_scale_protos[tid_int] = F.normalize(
                        proto_init[b], dim=-1
                    )
                    self.proto_initialized[tid_int] = True

    @torch.no_grad()
    def update_prototypes(self, multi_scale_features, masks, task_ids):
        for scale_idx, scale_feat in enumerate(multi_scale_features):
            B, C, H, W = scale_feat.shape
            feat_norm = F.normalize(scale_feat, dim=1)

            if masks.shape[-2:] != (H, W):
                masks_resized = F.interpolate(masks, size=(H, W),
                                              mode='bilinear', align_corners=True)
            else:
                masks_resized = masks

            for b in range(B):
                tid = int(task_ids[b].item())
                mask = masks_resized[b].squeeze()

                fg_mask = (mask > 0.5).float()
                bg_mask = (mask < 0.1).float()

                if fg_mask.sum() > 10:
                    fg_feat = (feat_norm[b] * fg_mask.unsqueeze(0)).sum(
                        dim=(1, 2)) / (fg_mask.sum() + 1e-6)
                    old_fg = self.multi_scale_protos[tid, scale_idx, 0]
                    new_fg = self.proto_momentum * old_fg + \
                             (1 - self.proto_momentum) * fg_feat
                    self.multi_scale_protos[tid, scale_idx, 0] = \
                        F.normalize(new_fg, dim=0)

                if bg_mask.sum() > 10:
                    bg_feat = (feat_norm[b] * bg_mask.unsqueeze(0)).sum(
                        dim=(1, 2)) / (bg_mask.sum() + 1e-6)
                    old_bg = self.multi_scale_protos[tid, scale_idx, 1]
                    new_bg = self.proto_momentum * old_bg + \
                             (1 - self.proto_momentum) * bg_feat
                    self.multi_scale_protos[tid, scale_idx, 1] = \
                        F.normalize(new_bg, dim=0)

    def proto_contrastive_loss(self, task_ids, margin=0.0):
        fg_list, bg_list = [], []
        for tid in task_ids:
            t = int(tid.item())
            protos_t = self.multi_scale_protos[t]
            fg_list.append(protos_t[0, 0, :])
            bg_list.append(protos_t[0, 1, :])
        fg = torch.stack(fg_list, dim=0)
        bg = torch.stack(bg_list, dim=0)
        cos = F.cosine_similarity(fg, bg, dim=-1)
        return cos.mean()

    def forward(self, x_low, x_high, task_ids, task_emb, gt_masks=None):
        B = x_low.size(0)

        if x_high.shape[-2:] != x_low.shape[-2:]:
            x_high = F.interpolate(x_high, size=x_low.shape[-2:],
                                   mode="bilinear", align_corners=True)
        xl = self.proj_low(x_low)
        xh = self.proj_high(x_high)
        x = self.fuse(torch.cat([xh, xl], dim=1))
        B, C, H, W = x.shape

        self.initialize_prototypes(task_ids, task_emb)

        multi_scale_features = []
        proto_similarities = []
        proto_attended_features = []
        for scale_idx, scale_enc in enumerate(self.scale_encoders):
            scale_feat = scale_enc(x)
            multi_scale_features.append(scale_feat)
            scale_feat_norm = F.normalize(scale_feat, dim=1)

            fg_protos = torch.stack([
                self.multi_scale_protos[int(tid.item()), scale_idx, 0]
                for tid in task_ids
            ])
            bg_protos = torch.stack([
                self.multi_scale_protos[int(tid.item()), scale_idx, 1]
                for tid in task_ids
            ])

            fg_protos = F.normalize(fg_protos, dim=1)
            bg_protos = F.normalize(bg_protos, dim=1)

            fg_sim = torch.einsum('bchw,bc->bhw', scale_feat_norm, fg_protos)
            bg_sim = torch.einsum('bchw,bc->bhw', scale_feat_norm, bg_protos)
            scale_sim = (fg_sim - bg_sim).unsqueeze(1)
            proto_similarities.append(scale_sim)

            feat_seq = scale_feat.flatten(2).transpose(1, 2)
            proto_query = torch.stack([fg_protos, bg_protos], dim=1)

            attended, attn_weights = self.proto_cross_attn[scale_idx](
                proto_query, feat_seq, feat_seq
            )

            fg_att = attended[:, 0, :]
            bg_att = attended[:, 1, :]
            concat = torch.cat([fg_att, bg_att, fg_att - bg_att], dim=-1)
            proto_feat = self.proto_fuser(concat)
            proto_attended_features.append(proto_feat)

        if self.training and gt_masks is not None:
            self.update_prototypes(multi_scale_features, gt_masks, task_ids)

        task_scale_weights = self.scale_router(task_emb)
        final_scale_weights = task_scale_weights * self.adaptive_scale_weights.unsqueeze(0)
        final_scale_weights = final_scale_weights / final_scale_weights.sum(dim=1, keepdim=True)

        fused_proto = proto_attended_features[0]
        fused_sim = proto_similarities[0]

        proto_features_flat = fused_proto
        expert_kernels = self.expert_generator(proto_features_flat)
        expert_kernels = expert_kernels.view(B, self.M, C, 9)

        identity9 = self.identity_kernel.view(self.c_mid, 9).unsqueeze(0).unsqueeze(0)
        expert_kernels = expert_kernels * self.expert_init_scale + \
                         identity9 * (1 - self.expert_init_scale)

        expert_weights = self.expert_weight_gen(proto_features_flat)

        sim_weight = torch.sigmoid(fused_sim)
        x_enhanced = x * (1 + 0.5 * sim_weight)

        k = 3
        x_unf = F.unfold(x_enhanced, kernel_size=k, padding=1)
        HW = H * W
        x_unf = x_unf.view(B, C, 9, HW)

        expert_conv = torch.einsum('bmcr,bcrw->bmcw', expert_kernels, x_unf)
        y = (expert_conv * expert_weights.view(B, self.M, 1, 1)).sum(dim=1)
        y = y.view(B, C, H, W)

        logits_all = self.head(y)
        idx = task_ids.view(-1, 1, 1, 1).long()
        logit_task = logits_all.gather(1, idx.expand(-1, 1, H, W))

        sim_norm = fused_sim / (fused_sim.abs().mean() + 1e-6)
        sim_scaled = torch.tanh(sim_norm / self.sim_temp.clamp(min=1e-3))
        proto_signal = self.proto_lambda * self.proto_amplifier * sim_scaled

        logit_task = logit_task + proto_signal

        pred = F.interpolate(logit_task, size=512, mode="bilinear", align_corners=True)

        return pred, y, final_scale_weights

class LearnableSplitGate(nn.Module):
    def __init__(self, num_tasks, num_blocks, temperature=1.0, learnable_range=(8, 48)):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_blocks = num_blocks
        self.learnable_start, self.learnable_end = learnable_range
        learnable_len = self.learnable_end - self.learnable_start

 
        self.register_buffer('base_temperature', torch.tensor(temperature))
        self.register_buffer('min_temperature', torch.tensor(0.3))
        self.register_buffer('epoch_counter', torch.tensor(0, dtype=torch.long))

 
        init_logits = []
        for i in range(num_tasks):
            center_ratio = torch.rand(1).item()
            center_block = int(center_ratio * learnable_len)
            logits = torch.linspace(-5, 5, learnable_len)
            shift = (learnable_len // 2) - center_block
            logits = logits + shift * 0.3
            logits = logits + torch.randn(learnable_len) * 0.8
            init_logits.append(logits)

        self.split_logits = nn.Parameter(torch.stack(init_logits))

        self.register_buffer('best_task_dice', torch.zeros(num_tasks))
        self.register_buffer('recent_task_dice', torch.zeros(num_tasks))
        self.register_buffer('epochs_since_improvement', torch.zeros(num_tasks, dtype=torch.long))

    @property
    def current_temperature(self):
        """动态温度：50 epoch内从1.0降到0.3"""
        progress = min(self.epoch_counter.float() / 50.0, 1.0)
        return self.base_temperature * (1 - progress) + self.min_temperature * progress

    def update_epoch(self):
        self.epoch_counter += 1

    def update_task_performance(self, task_id, dice_score):
        """

        Args:
            task_id: 
            dice_score:
        """
        self.recent_task_dice[task_id] = dice_score

  
        if dice_score > self.best_task_dice[task_id]:
            self.best_task_dice[task_id] = dice_score
            self.epochs_since_improvement[task_id] = 0
        else:
            self.epochs_since_improvement[task_id] += 1

    def get_split_weight(self, task_id, block_id):
        if block_id < self.learnable_start:
            return torch.tensor(0.0, device=self.split_logits.device)
        elif block_id >= self.learnable_end:
            return torch.tensor(1.0, device=self.split_logits.device)
        else:
            idx = block_id - self.learnable_start
            logit = self.split_logits[task_id, idx]
            return torch.sigmoid(logit / self.current_temperature)

    def get_all_probs(self):
        return torch.sigmoid(self.split_logits / self.current_temperature)

    def compute_regularization(self, task_losses=None):
        probs = self.get_all_probs()

        if probs.size(1) > 1:
            violations = F.relu(probs[:, :-1] - probs[:, 1:] - 0.15)
            monotonic_loss = violations.mean()
        else:
            monotonic_loss = torch.tensor(0.0, device=probs.device)

        sparsity_loss = (4 * probs * (1 - probs)).pow(2).mean()

        temp_reg = F.relu(0.5 - self.current_temperature)

        if self.num_tasks > 1:
            centers = []
            for task_id in range(self.num_tasks):
                center_idx = (probs[task_id] - 0.5).abs().argmin()
                centers.append(center_idx.float())
            centers = torch.stack(centers)
            diversity_loss = -torch.std(centers)
        else:
            diversity_loss = torch.tensor(0.0, device=probs.device)

        exploration_penalty = torch.tensor(0.0, device=probs.device)

        for task_id in range(self.num_tasks):

            is_early_stage = self.epoch_counter < 10
            is_stagnant = self.epochs_since_improvement[task_id] > 5
            current_perf = self.recent_task_dice[task_id]
            best_perf = self.best_task_dice[task_id]

            if is_early_stage:
                exploration_penalty += 0.0 
            elif is_stagnant and current_perf < best_perf - 0.02:
                mid_range_prob = probs[task_id, 15:25].mean()
                stuck_penalty = 1.0 - (mid_range_prob - 0.5).abs() * 2.0
                exploration_penalty += stuck_penalty * 0.5
            else:
                pass

        loss_guided_penalty = torch.tensor(0.0, device=probs.device)
        if task_losses is not None:
            task_losses = task_losses.detach()
            mean_loss = task_losses.mean()
            std_loss = task_losses.std() + 1e-6

            for task_id in range(self.num_tasks):
                normalized_loss = (task_losses[task_id] - mean_loss) / std_loss

                if normalized_loss > 0.5:
                    if self.epoch_counter < 10:
                        mid_range_prob = probs[task_id, 15:25].mean()
                        stuck_penalty = (mid_range_prob - 0.5).abs()
                        loss_guided_penalty += (1.0 - stuck_penalty) * 0.3

        return {
            'monotonic': monotonic_loss,
            'sparsity': sparsity_loss,
            'temperature': temp_reg,
            'diversity': diversity_loss,
            'exploration': exploration_penalty,
            'loss_guided': loss_guided_penalty
        }

class TaskConditionedAdapter(nn.Module):
    def __init__(self, hidden_dim: int, num_tasks: int, task_r: int = 64):
        super().__init__()
        self.num_tasks = num_tasks
        self.norm = nn.GroupNorm(1, hidden_dim)
        self.task_routers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, task_r, 1, bias=False),
                nn.GELU(),
                nn.Conv2d(task_r, hidden_dim, 1, bias=False)
            ) for _ in range(num_tasks)
        ])
        self.task_scales = nn.Parameter(torch.ones(num_tasks) * 1)
        for router in self.task_routers:
            nn.init.kaiming_uniform_(router[0].weight, a=math.sqrt(5))
            nn.init.zeros_(router[2].weight)

    def forward(self, x, task_ids):
        x = x.permute(0, 3, 1, 2)
        x_norm = self.norm(x)
        B = x.size(0)
        y_task_list = []
        for b in range(B):
            tid = int(task_ids[b].item())
            task_delta = self.task_routers[tid](x_norm[b:b + 1])
            y_task = x[b:b + 1] + self.task_scales[tid] * task_delta
            y_task_list.append(y_task)
        y_out = torch.cat(y_task_list, dim=0)
        y_out = y_out.permute(0, 2, 3, 1)
        return y_out

class TaskConditionedBlockWrapper(nn.Module):
    def __init__(self, original_block, adapter):
        super().__init__()
        self.block = original_block
        self.adapter = adapter
        self._task_ids = None

    def forward(self, x):
        if self._task_ids is not None:
            x = self.adapter(x, self._task_ids)
        x = self.block(x)
        return x

    def set_task_ids(self, task_ids):
        self._task_ids = task_ids

class Tpseg(nn.Module):
    def __init__(self, num_tasks=8, checkpoint_path=None, expert_M=6):
        super().__init__()
        self.num_tasks = num_tasks

        from sam2.build_sam import build_sam2
        model_cfg = "sam2_hiera_l.yaml"
        model = build_sam2(model_cfg, checkpoint_path) if checkpoint_path else build_sam2(model_cfg)

        del model.sam_mask_decoder, model.sam_prompt_encoder
        del model.memory_encoder, model.memory_attention
        del model.mask_downsample, model.obj_ptr_tpos_proj, model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for p in self.encoder.parameters():
            p.requires_grad = False

        wrapped_blocks = []
        for blk in self.encoder.blocks:
            hidden_dim = blk.attn.qkv.in_features
            adapter = TaskConditionedAdapter(
                hidden_dim=hidden_dim,
                num_tasks=num_tasks,
                task_r=64
            )
            wrapped_block = TaskConditionedBlockWrapper(blk, adapter)
            wrapped_blocks.append(wrapped_block)

        self.encoder.blocks = nn.ModuleList(wrapped_blocks)

        self.task_embedding = nn.Embedding(num_tasks, 64)

        self.rfb1 = RFB_modified(144, 64)
        self.rfb2 = RFB_modified(288, 128)
        self.rfb3 = RFB_modified(576, 256)
        self.rfb4 = RFB_modified(1152, 512)
        self.rfb2_1 = RFB_modified(128, 32)
        self.rfb3_1 = RFB_modified(256, 32)
        self.rfb4_1 = RFB_modified(512, 32)
        self.pgtd1 = PrototypeGuidedTaskDecoder(
            c_low=64, c_high=128, c_mid=128, tdim=64,
            num_tasks=num_tasks, M=expert_M, scales=[1],
            proto_momentum=0.995, proto_lambda=5.0
        )
        self.pgtd2 = PrototypeGuidedTaskDecoder(
            c_low=128, c_high=256, c_mid=128, tdim=64,
            num_tasks=num_tasks, M=expert_M, scales=[1],
            proto_momentum=0.995, proto_lambda=5.0
        )
        self.pgtd3 = PrototypeGuidedTaskDecoder(
            c_low=256, c_high=512, c_mid=256, tdim=64,
            num_tasks=num_tasks, M=expert_M, scales=[1],
            proto_momentum=0.995, proto_lambda=5.0
        )

        self.predtrans4 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, groups=512), nn.GroupNorm(1, 512), nn.GELU(),
            nn.Conv2d(512, num_tasks, 1)
        )
        self.predtrans3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, groups=256), nn.GroupNorm(1, 256), nn.GELU(),
            nn.Conv2d(256, num_tasks, 1)
        )
        self.predtrans2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1, groups=128), nn.GroupNorm(1, 128), nn.GELU(),
            nn.Conv2d(128, num_tasks, 1)
        )

        self.fuser = TaskLogitFuser(num_tasks)

    def forward(self, x, task_ids, gt_masks=None):
        task_ids = task_ids.long()
        task_emb = self.task_embedding(task_ids)

        for block_wrapper in self.encoder.blocks:
            block_wrapper.set_task_ids(task_ids)

        x1, x2, x3, x4 = self.encoder(x)
        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        logits4 = self.predtrans4(x4)
        idx = task_ids.view(-1, 1, 1, 1)
        out4 = logits4.gather(1, idx.expand(-1, 1, logits4.size(2), logits4.size(3)))
        out4 = F.interpolate(out4, 512, mode="bilinear", align_corners=True)

        out3, x3, _ = self.pgtd3(x3, x4, task_ids, task_emb, gt_masks)
        out2, x2, _ = self.pgtd2(x2, x3, task_ids, task_emb, gt_masks)
        out1, enhanced_features, scale_weights = self.pgtd1(x1, x2, task_ids, task_emb, gt_masks)

        out = self.fuser([out1, out2, out3, out4], task_ids)

        if self.training and gt_masks is not None:
            return out, out1, out2, out3, scale_weights
        else:
            return out, out1, out2, out3
    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)




















