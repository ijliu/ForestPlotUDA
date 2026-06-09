"""
Trainer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import weakref
import torch
import torch.nn as nn
import torch.utils.data
from packaging import version
from functools import partial

if sys.version_info >= (3, 10):
    from collections.abc import Iterator
else:
    from collections import Iterator
from tensorboardX import SummaryWriter

from .defaults import create_ddp_model, worker_init_fn
from .hooks import HookBase, build_hooks
import pointcept.utils.comm as comm
from pointcept.datasets import build_dataset, point_collate_fn, collate_fn
from pointcept.models import build_model
from pointcept.utils.logger import get_root_logger
from pointcept.utils.optimizer import build_optimizer
from pointcept.utils.scheduler import build_scheduler
from pointcept.utils.events import EventStorage, ExceptionWriter
from pointcept.utils.registry import Registry

from pointcept.utils.scheduler import CosineScheduler

from pointcept.utils.pc_views import PCViews
import numpy as np
import cv2

import torch.nn.functional as F


import matplotlib.pyplot as plt

# SAM2
from image_segment.sam2.build_sam import build_sam2
from image_segment.sam2.sam2_image_predictor import SAM2ImagePredictor

RESOLUTION = 512
checkpoint = "./image_segment/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))



TRAINERS = Registry("trainers")
AMP_DTYPE = dict(
    float16=torch.float16,
    bfloat16=torch.bfloat16,
)


class TrainerBase:
    def __init__(self) -> None:
        self.hooks = []
        self.model = None
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = 0
        self.max_iter = 0
        self.comm_info = dict()
        self.data_iterator: Iterator = enumerate([])
        self.storage: EventStorage
        self.writer: SummaryWriter

    def register_hooks(self, hooks) -> None:
        hooks = build_hooks(hooks)
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self.hooks.extend(hooks)

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def before_train(self):
        for h in self.hooks:
            h.before_train()

    def before_epoch(self):
        for h in self.hooks:
            h.before_epoch()

    def before_step(self):
        for h in self.hooks:
            h.before_step()

    def run_step(self):
        raise NotImplementedError

    def after_step(self):
        for h in self.hooks:
            h.after_step()

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()

    def after_train(self):
        # Sync GPU before running train hooks
        comm.synchronize()
        for h in self.hooks:
            h.after_train()
        if comm.is_main_process():
            self.writer.close()


@TRAINERS.register_module("DefaultTrainer")
class Trainer(TrainerBase):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def run_step(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(torch.amp.autocast, device_type="cuda")
        else:
            # deprecated warning
            auto_cast = torch.cuda.amp.autocast

        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

                
        ##################
        # conditions = str(self.epoch % 5)
        # input_dict['condition'] = [conditions for _ in range(len(input_dict["name"]))]
        # input_dict['context'] = torch.zeros((256)).to('cuda')
        ###################

        with auto_cast(
            enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]
        ):
            output_dict = self.model(input_dict)
            loss = output_dict["loss"]
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            # ############
            # print("=== 裁剪前的梯度 ===")
            # total_norm_before = 0.0  # 所有参数的梯度总范数
            # max_grad_before = -float('inf')  # 所有参数的最大梯度值

            # for name, param in self.model.named_parameters():
            #     if param.grad is not None:  # 确保参数有梯度（排除冻结层）
            #         # 计算当前参数的梯度范数（L2范数）
            #         param_norm = param.grad.data.norm(2)
            #         total_norm_before += param_norm.item() **2  # 累加平方和（用于总范数）
            #         # 记录当前参数的最大梯度值
            #         current_max = param.grad.data.max().item()
            #         if current_max > max_grad_before:
            #             max_grad_before = current_max
            #         # 打印单个参数的梯度信息（可选，参数多的话会刷屏）
            #         # print(f"{name} 梯度范数: {param_norm.item():.4f}, 最大梯度值: {current_max:.4f}")

            # total_norm_before = total_norm_before** 0.5  # 总范数 = 平方和的平方根
            # print(f"所有参数的梯度总范数: {total_norm_before:.4f}")
            # print(f"所有参数的最大梯度值: {max_grad_before:.4f}\n")
            # ############
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.scaler.step(self.optimizer)

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.optimizer.step()
            self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            grad_scaler = partial(torch.amp.GradScaler, device="cuda")
        else:
            # deprecated warning
            grad_scaler = torch.cuda.amp.GradScaler
        scaler = grad_scaler() if self.cfg.enable_amp else None
        return scaler


# import cv2
# import numpy as np
# import torch

def preprocess_depth_for_sam2(depth_map, max_dilation=3):
    """
    预处理深度图，使其更适合SAM2分割
    Args:
        depth_map: [H, W] 深度图张量
        max_dilation: 最大膨胀半径（像素）
    Returns:
        processed_depth: 处理后的深度图
    """
    # 转换为numpy数组
    depth_np = depth_map.cpu().numpy()

    # 创建有效区域掩码（深度值有效的区域）
    valid_mask = (depth_np > 0) & (depth_np < 1e5)
    
    if np.any(valid_mask):
        # 1. 填充小空洞 - 使用形态学闭运算
        # 首先创建二值掩码
        binary_mask = valid_mask.astype(np.uint8)
        
        # 设置膨胀核大小（根据图像分辨率调整）
        kernel_size = min(5, max(2, int(RESOLUTION * 0.01)))  # 大约1%的图像尺寸
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 闭运算：先膨胀后腐蚀，填充小空洞但保持大致形状
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 2. 轻微膨胀以连接近邻区域
        dilated_mask = cv2.dilate(closed_mask, kernel, iterations=1)
        
        # 3. 创建处理后的深度图
        processed_depth = np.copy(depth_np)
        
        # 仅对原来无效但现在有效的区域进行插值
        to_fill_mask = (dilated_mask == 1) & (binary_mask == 0)
        
        if np.any(to_fill_mask):
            # 使用最近邻插值填充空洞
            coords = np.array(np.nonzero(valid_mask)).T
            values = depth_np[valid_mask]
            fill_coords = np.array(np.nonzero(to_fill_mask)).T
            
            if len(fill_coords) > 0 and len(coords) > 0:
                # 使用简单最近邻（实际应用中可使用KDTree加速）
                for i in range(len(fill_coords)):
                    y, x = fill_coords[i]
                    # 计算到所有有效点的距离
                    dists = np.sum((coords - [y, x])**2, axis=1)
                    nearest_idx = np.argmin(dists)
                    processed_depth[y, x] = values[nearest_idx]
        
        # 4. 高斯模糊轻微平滑边界（可选）
        processed_depth = cv2.GaussianBlur(processed_depth, (3, 3), 0.5)
        
        # 转回PyTorch张量
        processed_depth = torch.from_numpy(processed_depth).to(depth_map.device)
        return processed_depth
    else:
        # 没有有效深度，直接返回原图
        return depth_map



def create_semantic_mask(index_maps, segments):
    """
    根据索引图和点云语义标签生成语义分割mask
    
    Args:
        index_maps: [B*4, 512, 512] - 索引图
        segments: [B, N] - 每个点的语义标签
    
    Returns:
        semantic_mask: [B*4, 512, 512] - 语义分割mask
    """
    B, N = segments.shape
    Bv, H, W = index_maps.shape
    v = Bv // B
    
    # 复制segments以匹配Bv (B*4)
    segments_expanded = segments.repeat_interleave(v, dim=0)  # [Bv, N]
    
    # 创建输出mask，初始化为-1
    semantic_mask = torch.full_like(index_maps, -1)
    
    # 获取所有有效像素的坐标
    b_coords, h_coords, w_coords = torch.where(index_maps != -1)
    
    # 获取对应的点索引
    point_indices = index_maps[b_coords, h_coords, w_coords].long()
    
    # 获取对应的语义标签
    semantic_values = segments_expanded[b_coords, point_indices]
    
    # 填充有效区域
    semantic_mask[b_coords, h_coords, w_coords] = semantic_values
    
    return semantic_mask

def semantic_mask_to_color(semantic_mask, num_classes=None):
    """
    将语义分割mask转换为彩色图像
    
    Args:
        semantic_mask: [B*4, 512, 512] - 语义分割mask
        num_classes: 可选，类别总数
    
    Returns:
        color_mask: [B*4, 3, 512, 512] - 彩色语义分割mask
    """
    # 确定类别数量
    if num_classes is None:
        valid_mask = (semantic_mask != -1)
        if valid_mask.any():
            num_classes = semantic_mask[valid_mask].max().item() + 1
        else:
            num_classes = 1
    
    # 生成颜色映射表
    cmap = plt.cm.get_cmap('tab20', num_classes)
    
    # 将colormap转换为[0, 255]范围的RGB值
    colors = torch.tensor([cmap(i)[:3] for i in range(num_classes)], dtype=torch.float32) * 255
    colors = torch.cat([colors, torch.tensor([[0, 0, 0]], dtype=torch.float32)], dim=0)  # 添加黑色表示无效区域
    colors = colors.to(semantic_mask.device)
    
    # 将-1转换为num_classes，以便索引colors
    semantic_mask_with_invalid = semantic_mask.clone()
    semantic_mask_with_invalid[semantic_mask == -1] = num_classes
    
    # 直接索引获取颜色
    color_mask = colors[semantic_mask_with_invalid.long()]
    
    # 转换维度为[B*4, 3, 512, 512]
    color_mask = color_mask.permute(0, 3, 1, 2)
    
    return color_mask

class SoftWeightBank:
    def __init__(self, C, feat_dim, device, momentum=0.8, temp=0.07):
        self.C = C
        self.feat_dim = feat_dim
        self.momentum = momentum
        self.temp = temp  # 用于相似度锐化
        # 初始化：单位随机向量
        self.protos = F.normalize(torch.randn(C, feat_dim), dim=1).to(device)
        self.counts = torch.zeros(C, dtype=torch.long, device=device)

    @torch.no_grad()
    def soft_weight(self, feat, pseudo):
        """
        feat:   (M, feat_dim)  教师特征，已 L2 归一化
        pseudo: (M,)           伪标签（5*C 条）
        return: (M,)           软权重 w，均值≈1
        """
        M = feat.size(0)
        w = torch.zeros(M, device=feat.device)
        for c in range(self.C):
            mask = (pseudo == c)
            if mask.sum() == 0:
                continue
            feat_c = feat[mask]                # (m, feat_dim)
            proto_c = self.protos[c]           # (feat_dim,)
            # 余弦相似度 + 温度锐化
            sim = torch.mv(feat_c, proto_c) / self.temp  # (m,)
            score = torch.sigmoid(sim)         # 压到 0-1
            # 与教师置信度几何平均（教师 prob 已在外部算好）
            # 若外部没给 prob，可只用 score
            w[mask] = score
        # 归一化：让平均值为 1，方便直接乘 CE
        w = w / (w.mean() + 1e-8)
        return w

    @torch.no_grad()
    def update(self, feat, pseudo, epoch):
        """
        用本次 5*C 条特征（feat）更新原型
        建议：只在学生 loss 回传后再调一次
        """
        for c in range(self.C):
            mask = (pseudo == c)
            if mask.sum() == 0:
                continue
            feat_c = feat[mask]                  # (m, feat_dim)
            new_proto = feat_c.median(0)[0]
            new_proto = F.normalize(new_proto, dim=0)
            # 动量更新
            self.protos[c] = (1 - self.momentum) * new_proto + \
                             self.momentum * self.protos[c]
            self.protos[c] = F.normalize(self.protos[c], dim=0)
            self.counts[c] += mask.sum().item()



def fps(x, k, start_idx=None):
    """
    x: (N,3)  tensor
    k: int
    return: (k,)  LongTensor  全局索引（这里实际是类内索引）
    """
    N, _ = x.size()
    if k >= N:
        return torch.arange(N, device=x.device, dtype=torch.long)
    idx = torch.zeros(k, dtype=torch.long, device=x.device)
    if start_idx is None:
        start_idx = torch.randint(0, N, (1,)).item()
    idx[0] = start_idx
    dist = torch.full((N,), float('inf'), device=x.device)
    for i in range(1, k):
        last = x[idx[i-1]]
        dist = torch.minimum(dist, (x - last).pow(2).sum(1))
        idx[i] = torch.argmax(dist)
    return idx


@TRAINERS.register_module("DefaultUDATrainer")
class UDATrainer(TrainerBase):
    def __init__(self, cfg):
        super(UDATrainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")

        self.select_score = cfg.score
        
        #########
        self.momentum_base = 0.996
        self.momentum_final = 1.0

        self.logger.info("=> Building Teacher model ...")
        self.teacher = self.build_teacher_model()
        self.logger.info("=> Building Student model ...")
        self.model = self.build_student_model()
        #########

        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)


        self.bank = SoftWeightBank(4, feat_dim=64, device='cuda', temp=0.07)

        # self.embedding_table = nn.Embedding(5,  256)
        # context = self.embedding_table(
        #     torch.tensor(
        #         [self.conditions.index(condition)], device=data_dict["coord"].device
        #     )
        # )

    def before_train(self):
        for h in self.hooks:
            h.before_train()

        total_steps = self.cfg.scheduler.total_steps
        curr_step = self.start_epoch * len(self.train_loader)

        # momentum scheduler
        self.momentum_scheduler = CosineScheduler(
            base_value=self.momentum_base,
            final_value=self.momentum_final,
            total_iters=total_steps,
        )
        self.momentum_scheduler.iter = curr_step

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                self.teacher.eval()
                self.data_iterator = enumerate(self.train_loader)
                self.before_epoch()

                noise_error_num = 0
                noise_total_num = 0
                reliable_num = 0
                total_points = 0
                mask_sum = 0

                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    ret_dict = self.run_step()

                    noise_error_num += ret_dict["noise_error_num"]
                    noise_total_num += ret_dict["noise_total_num"]
                    reliable_num += ret_dict["reliable_num"]
                    total_points += ret_dict["total_points"]
                    mask_sum += ret_dict["mask_sum"]
                    # => after_step
                    self.after_step()

                noise_rate = noise_error_num / noise_total_num if noise_total_num > 0 else 0.0
                reliable_ratio = reliable_num / total_points * 100
                used_points = mask_sum.item()

                print(f'noise_rate {noise_rate.item():.4f} '
                f'reliable_ratio {reliable_ratio:.2f}% '
                f'used_points {used_points}')
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()


    def before_step(self):
        for h in self.hooks:
            h.before_step()

        self.momentum = self.momentum_scheduler.step()

    def run_step(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(torch.amp.autocast, device_type="cuda")
        else:
            # deprecated warning
            auto_cast = torch.cuda.amp.autocast

        # import copy
        # input_dict = copy.deepcopy(self.comm_info["input_dict"])
        input_dict = self.comm_info["input_dict"]
        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

        ###################### 生成伪标签 ################################
        # print(input_dict["name"])

        ####################
        conditions = None
        if "CULS" in input_dict["name"][0]:
            conditions = "CULS"
        elif "NIBIO" in input_dict["name"][0]:
            conditions = "NIBIO"
        elif "RMIT" in input_dict["name"][0]:
            conditions = "RMIT"
        elif "SCION" in input_dict["name"][0]:
            conditions = "SCION"
        elif "TUWIEN" in input_dict["name"][0]:
            conditions = "TUWIEN"

        input_dict['condition'] = conditions
        ###################

        random_k = True
        
        IGNORE_INDEX = -1
        PROB_MAX = self.select_score
        MAX_PER_CLASS = 5
        MIN_PER_CLASS = 5

        with torch.no_grad():
            output = self.teacher(input_dict)
            logits = output['seg_logits']
            feat = output['feat']
            prob = torch.softmax(logits, dim=1)
            max_prob, pseudo_cls = torch.max(prob, dim=1)

            base_thresh = torch.full_like(max_prob, PROB_MAX)
            confident = max_prob >= base_thresh                  

            N = pseudo_cls.shape[0]
            num_classes = prob.shape[1]
            selected_mask = torch.zeros_like(confident, dtype=torch.bool)

            # 分别对每个类别进行处理
            for cls in range(num_classes):
                cls_mask = pseudo_cls == cls
                if not cls_mask.any():
                    continue

                if "CULS" in input_dict["name"][0] and cls == 1:
                    # print(f"Skipping class 1 for CULS dataset: {input_dict["name"]}")
                    continue

                # 该类所有高置信点的置信度 + 索引
                cls_conf = max_prob[cls_mask & confident]
                cls_idx = torch.nonzero(cls_mask & confident, as_tuple=False).squeeze(1)

                if cls_conf.shape[0] == 0:  # 这个类连一个高置信点都没有 → 强制取 top-k
                    cls_conf_all = max_prob[cls_mask]
                    cls_idx_all = torch.nonzero(cls_mask, as_tuple=False).squeeze(1)
                    topk = torch.topk(cls_conf_all, k=min(MIN_PER_CLASS, cls_conf_all.shape[0]), largest=True)
                    selected_mask[cls_idx_all[topk.indices]] = True
                else:
                    if cls_conf.shape[0] > MAX_PER_CLASS:
                        # 超过 MAX ，随机下采样（比 topk 更快且方差小）
                        # 随机采样
                        if random_k:
                            perm = torch.randperm(cls_conf.shape[0], device=cls_conf.device)
                            selected_idx = cls_idx[perm[:MAX_PER_CLASS]]
                            selected_mask[selected_idx] = True
                        else:  # FPS
                            cls_coord = input_dict["coord"][cls_mask & confident]   # (M,3)
                            # 2. FPS 选 50 个最远点 → 返回类内索引
                            fps_idx_intra = fps(cls_coord, k=MAX_PER_CLASS)        # (50,)
                            # 3. 映射回全局索引
                            cls_idx_global = torch.nonzero(cls_mask & confident, as_tuple=False).squeeze(1)
                            selected_idx = cls_idx_global[fps_idx_intra]           # (50,)
                            selected_mask[selected_idx] = True
                    else:
                        selected_mask[cls_idx] = True
                        if cls_conf.shape[0] < MIN_PER_CLASS:
                            need = MIN_PER_CLASS - cls_conf.shape[0]
                            # 同类但低于阈值的点
                            lower_mask = cls_mask & ~confident
                            if lower_mask.any():
                                lower_conf = max_prob[lower_mask]
                                lower_idx = torch.nonzero(lower_mask, as_tuple=False).squeeze(1)
                                topk = torch.topk(lower_conf, k=min(need, lower_conf.shape[0]), largest=True)
                                selected_mask[lower_idx[topk.indices]] = True

            reliable = selected_mask

            pseudo = torch.where(reliable,
                                 pseudo_cls.long(),
                                 torch.tensor(IGNORE_INDEX, device=max_prob.device))
            
            feat5_all = feat[selected_mask]
            y5_all = pseudo_cls[selected_mask]
            weight = torch.zeros_like(pseudo, dtype=torch.float32)

            ############ bank 
            w = self.bank.soft_weight(feat5_all, y5_all)
            self.bank.update(feat5_all, y5_all, self.epoch)
            weight[selected_mask] = w
            #############
            # cnum = np.array([0,0,0,0])
            # for cid in torch.unique(pseudo):
            #     if cid >= 0:
            #         idx = torch.where(pseudo == cid)
            #         cnum[cid] = cnum[cid] + idx[0].shape[0]
            # cw = cnum.sum() / (cnum)

            # for cid in torch.unique(pseudo):
            #     if cid >= 0:
            #         idx = torch.where(pseudo == cid)
            #         weight[idx] = cw[cid]
            # exit()
            # weight[selected_mask] = 1.0

            input_dict["weight"] = weight

            final_pseudo = pseudo
            final_reliable = reliable.clone()
            
            # ============== 5. 噪声率统计和结果更新 ==============
            with torch.no_grad():
                gt = input_dict['segment']
                valid_gt = gt != -1
                mask = final_reliable & valid_gt
                mask_sum = mask.sum()

                noise_error_num = (final_pseudo[mask] != gt[mask]).sum().float()
                noise_total_num = mask_sum.float()
                reliable_num = final_reliable.sum().float()
                total_points = torch.numel(final_reliable)

                ret_dict = {"mask_sum": mask_sum, "noise_error_num": noise_error_num, "noise_total_num": noise_total_num, "reliable_num": reliable_num, "total_points": total_points}

                # noise_rate = noise_error_num / noise_total_num if noise_total_num > 0 else 0.0
                # reliable_ratio = reliable_num / total_points * 100
                # used_points = mask_sum.item()

                # if mask.sum() > 0:
                #     noise_rate = (final_pseudo[mask] != gt[mask]).float().mean()
                # else:
                #     noise_rate = torch.tensor(0.0, device=gt.device)
                
                # reliable_ratio = final_reliable.float().mean() * 100
                # used_points = final_reliable.sum().item()
                # print(f'noise_rate {noise_rate.item():.4f} '
                #     f'reliable_ratio {reliable_ratio:.2f}% '
                #     f'used_points {used_points} (max{MAX_PER_CLASS}/min{MIN_PER_CLASS} per class)')
                
                segment = input_dict["segment"]
            # 更新输入字典中的分割标签
            input_dict['segment'] = final_pseudo
            
            # import numpy as np
            # from datetime import datetime

            # # ============== 新增的可视化代码 ==============
            # # 获取点云坐标 (假设坐标存储在 input_dict["coord"]，形状为 [N, 3])
            # # 若您的坐标键不同 (如 "points"), 请修改此处
            # coords = input_dict["coord"]  # [N, 3] tensor
            # # segment = input_dict["segment"]

            # # 将坐标和伪标签移至CPU并转为numpy
            # coords_np = coords.cpu().numpy()  # [N, 3]
            # pseudo_np = final_pseudo.cpu().numpy()  # [N]
            # segment_np = segment.cpu().numpy()

            # # 定义颜色映射 (BGR顺序，Open3D常用；若需RGB请交换R/B通道)
            # CLASS_COLORS = {
            #     0: [255, 0, 0],    # 红色 (类0)
            #     1: [0, 255, 0],    # 绿色 (类1)
            #     2: [0, 0, 255],    # 蓝色 (类2)
            #     3: [255, 255, 0],  # 黄色 (类3)
            #     -1: [128, 128, 128] # 灰色 (无效点)
            # }

            # # 为每个点分配颜色
            # colors = np.zeros((pseudo_np.shape[0], 3), dtype=np.uint8)
            # for i, cls_id in enumerate(pseudo_np):
            #     # 处理超出颜色映射范围的类别 (安全保护)
            #     if cls_id not in CLASS_COLORS and cls_id != -1:
            #         cls_id = -1  # 未知类别视为无效点
            #     colors[i] = CLASS_COLORS[cls_id]

            # # 准备输出数据 (x,y,z,r,g,b,cls)
            # output_data = np.hstack([
            #     coords_np,          # x,y,z
            #     segment_np[:, None],
            #     # colors,             # r,g,b (0-255)
            #     # pseudo_np[:, None]  # cls (保留原始类别ID)
            # ])

            # # 生成安全的文件名 (使用时间戳避免冲突)
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # scene_name = input_dict["name"][0] if isinstance(input_dict["name"], list) else str(input_dict["name"])
            # safe_name = "".join(c for c in scene_name if c.isalnum() or c in ('_', '-')).rstrip()
            # output_dir = "pseudo_label_vis"
            # os.makedirs(output_dir, exist_ok=True)
            # output_path = os.path.join(output_dir, f"{safe_name}_{timestamp}.txt")

            # # 保存为txt (逗号分隔)
            # np.savetxt(
            #     output_path,
            #     output_data,
            
            #     fmt="%.4f,%.4f,%.4f,%d",
            #     comments=''
            # )
            # print(f"伪标签可视化已保存至: {output_path}")


        with auto_cast(
            enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]
        ):
            output_dict = self.model(input_dict)
            loss = output_dict["loss"]
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.scaler.step(self.optimizer)

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.optimizer.step()
            self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

        return ret_dict

    def after_step(self):
        for h in self.hooks:
            h.after_step()
        # print("EMA UPDATE")
        # EMA update teacher
        with torch.no_grad():
            m = self.momentum
            student_param_list = list(self.model.parameters())
            teacher_param_list = list(self.teacher.parameters())
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_student_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model
    
    def build_teacher_model(self):
        model = build_model(self.cfg.teacher)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        # model = create_ddp_model(
        #     model.cuda(),
        #     broadcast_buffers=False,
        #     find_unused_parameters=self.cfg.find_unused_parameters,
        # )
        return model.cuda()

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            grad_scaler = partial(torch.amp.GradScaler, device="cuda")
        else:
            # deprecated warning
            grad_scaler = torch.cuda.amp.GradScaler
        scaler = grad_scaler() if self.cfg.enable_amp else None
        return scaler

@TRAINERS.register_module("DefaultTrainerV1")
class UDATrainerV1(TrainerBase):
    def __init__(self, cfg):
        super(UDATrainer, self).__init__()
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        self.best_metric_value = -torch.inf
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "train.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        # self.train_loader = self.build_train_loader()

        #########
        self.aux_loader = self.build_aux_loader()
        self.train_loader = self.build_train_loader()
        self.pseudo_loader = self.build_pseudo_loader()
        ##########
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building optimize, scheduler, scaler(amp) ...")
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Building hooks ...")
        self.register_hooks(self.cfg.hooks)

    def train(self):
        with EventStorage() as self.storage, ExceptionWriter():
            # => before train
            self.before_train()
            self.logger.info(">>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>")
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                if comm.get_world_size() > 1:
                    self.train_loader.sampler.set_epoch(self.epoch)
                self.model.train()
                self.data_iterator = enumerate(self.train_loader)
                self.aux_iter = iter(self.aux_loader)
                self.pseudo_iter = iter(self.pseudo_loader)
                self.before_epoch()
                # => run_epoch
                for (
                    self.comm_info["iter"],
                    self.comm_info["input_dict"],
                ) in self.data_iterator:
                    # => before_step
                    self.before_step()
                    # => run_step
                    self.run_step()
                    # => after_step
                    self.after_step()
                # => after epoch
                self.after_epoch()
            # => after train
            self.after_train()

    def run_step(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            auto_cast = partial(torch.amp.autocast, device_type="cuda")
        else:
            # deprecated warning
            auto_cast = torch.cuda.amp.autocast

        try:
            # 尝试从迭代器取数据（未耗尽时正常返回）
            aux_input_dict = next(self.aux_iter)
        except StopIteration:
            # 迭代器耗尽（一轮结束），重新创建迭代器（从头开始）
            self.aux_iter = iter(self.aux_loader)
            aux_input_dict = next(self.aux_iter)  # 取新一轮的第一个批次

        pseudo_input_dict = None
        if self.epoch > 100:
            try:
                # 尝试从迭代器取数据（未耗尽时正常返回）
                pseudo_input_dict = next(self.pseudo_iter)
            except StopIteration:
                # 迭代器耗尽（一轮结束），重新创建迭代器（从头开始）
                self.pseudo_iter = iter(self.pseudo_loader)
                pseudo_input_dict = next(self.pseudo_iter)  # 取新一轮的第一个批次

            for key in pseudo_input_dict.keys():
                if isinstance(pseudo_input_dict[key], torch.Tensor):
                    pseudo_input_dict[key] = pseudo_input_dict[key].cuda(non_blocking=True)
        # 

    
        train_input_dict = self.comm_info["input_dict"]
        for key in train_input_dict.keys():
            if isinstance(train_input_dict[key], torch.Tensor):
                train_input_dict[key] = train_input_dict[key].cuda(non_blocking=True)
        
        for key in aux_input_dict.keys():
            if isinstance(aux_input_dict[key], torch.Tensor):
                aux_input_dict[key] = aux_input_dict[key].cuda(non_blocking=True)

        with auto_cast(
            enabled=self.cfg.enable_amp, dtype=AMP_DTYPE[self.cfg.amp_dtype]
        ):
            output_dict = self.model(self.epoch, train_input_dict, aux_input_dict, pseudo_input_dict)
            loss = output_dict["loss"]
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.scaler.step(self.optimizer)

            # When enable amp, optimizer.step call are skipped if the loss scaling factor is too large.
            # Fix torch warning scheduler step before optimizer step.
            scaler = self.scaler.get_scale()
            self.scaler.update()
            if scaler <= self.scaler.get_scale():
                self.scheduler.step()
        else:
            loss.backward()
            if self.cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.clip_grad
                )
            self.optimizer.step()
            self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        self.comm_info["model_output_dict"] = output_dict

    def after_epoch(self):
        for h in self.hooks:
            h.after_epoch()
        self.storage.reset_histories()
        if self.cfg.empty_cache_per_epoch:
            torch.cuda.empty_cache()

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        self.logger.info(f"Tensorboard writer logging dir: {self.cfg.save_path}")
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=True,
        )
        return train_loader
    
    def build_pseudo_loader(self):
        train_data = build_dataset(self.cfg.data.pseudo)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=True,
        )
        return train_loader
    
    def build_aux_loader(self):
        train_data = build_dataset(self.cfg.data.aux)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=True,
        )
        return train_loader

    def build_pesudo_loader(self):
        train_data = build_dataset(self.cfg.data.pesudo)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = (
            partial(
                worker_init_fn,
                num_workers=self.cfg.num_worker_per_gpu,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=self.cfg.batch_size_per_gpu,
            shuffle=(train_sampler is None),
            num_workers=self.cfg.num_worker_per_gpu,
            sampler=train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=init_fn,
            drop_last=len(train_data) > self.cfg.batch_size,
            persistent_workers=True,
        )
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size=self.cfg.batch_size_val_per_gpu,
                shuffle=False,
                num_workers=self.cfg.num_worker_per_gpu,
                pin_memory=True,
                sampler=val_sampler,
                collate_fn=collate_fn,
            )
        return val_loader

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        if version.parse(torch.__version__) >= version.parse("2.4"):
            grad_scaler = partial(torch.amp.GradScaler, device="cuda")
        else:
            # deprecated warning
            grad_scaler = torch.cuda.amp.GradScaler
        scaler = grad_scaler() if self.cfg.enable_amp else None
        return scaler



@TRAINERS.register_module("MultiDatasetTrainer")
class MultiDatasetTrainer(Trainer):
    def build_train_loader(self):
        from pointcept.datasets import MultiDatasetDataloader

        train_data = build_dataset(self.cfg.data.train)
        train_loader = MultiDatasetDataloader(
            train_data,
            self.cfg.batch_size_per_gpu,
            self.cfg.num_worker_per_gpu,
            self.cfg.mix_prob,
            self.cfg.seed,
        )
        self.comm_info["iter_per_epoch"] = len(train_loader)
        return train_loader
