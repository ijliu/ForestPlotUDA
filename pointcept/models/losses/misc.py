"""
Misc Losses

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import LOSSES, build_criteria


@LOSSES.register_module()
class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
        loss_weight=1.0,
        ignore_index=-1,
    ):
        super(CrossEntropyLoss, self).__init__()
        weight = torch.tensor(weight).cuda() if weight is not None else None
        self.loss_weight = loss_weight

        self.reduction = reduction
        # self.loss = nn.CrossEntropyLoss(
        #     weight=weight,
        #     size_average=size_average,
        #     ignore_index=ignore_index,
        #     reduce=reduce,
        #     reduction=reduction,
        #     label_smoothing=label_smoothing,
        # )
        self.loss = nn.CrossEntropyLoss(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=False,               # 关键：先不求和
            label_smoothing=label_smoothing,
        )

    def forward(self, pred, target, pixel_weight=1.0):
        # return self.loss(pred, target) * self.loss_weight
        loss = self.loss(pred, target)
        loss = loss * pixel_weight

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        
        return loss * self.loss_weight

@LOSSES.register_module()
class SmoothCELoss(nn.Module):
    def __init__(self, smoothing_ratio=0.1):
        super(SmoothCELoss, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    def forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)
        loss = -(one_hot * log_prb).total(dim=1)
        loss = loss[torch.isfinite(loss)].mean()
        return loss


@LOSSES.register_module()
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.5, logits=True, reduce=True, loss_weight=1.0):
        """Binary Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(BinaryFocalLoss, self).__init__()
        assert 0 < alpha < 1
        self.gamma = gamma
        self.alpha = alpha
        self.logits = logits
        self.reduce = reduce
        self.loss_weight = loss_weight

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N)
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        if self.logits:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        else:
            bce = F.binary_cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-bce)
        alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_loss = alpha * (1 - pt) ** self.gamma * bce

        if self.reduce:
            focal_loss = torch.mean(focal_loss)
        return focal_loss * self.loss_weight


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(
        self, gamma=2.0, alpha=0.5, reduction="mean", loss_weight=1.0, ignore_index=-1
    ):
        """Focal Loss
        <https://arxiv.org/abs/1708.02002>`
        """
        super(FocalLoss, self).__init__()
        assert reduction in (
            "mean",
            "sum",
        ), "AssertionError: reduction should be 'mean' or 'sum'"
        assert isinstance(
            alpha, (float, list)
        ), "AssertionError: alpha should be of type float"
        assert isinstance(gamma, float), "AssertionError: gamma should be of type float"
        assert isinstance(
            loss_weight, float
        ), "AssertionError: loss_weight should be of type float"
        assert isinstance(ignore_index, int), "ignore_index must be of type int"
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        """Forward function.
        Args:
            pred (torch.Tensor): The prediction with shape (N, C) where C = number of classes.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1, If containing class probabilities,
                same shape as the input.
        Returns:
            torch.Tensor: The calculated loss
        """
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        if len(target) == 0:
            return 0.0

        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes)

        alpha = self.alpha
        if isinstance(alpha, list):
            alpha = pred.new_tensor(alpha)
        pred_sigmoid = pred.sigmoid()
        target = target.type_as(pred)
        one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (alpha * target + (1 - alpha) * (1 - target)) * one_minus_pt.pow(
            self.gamma
        )

        loss = (
            F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            * focal_weight
        )
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.total()
        return self.loss_weight * loss



@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self, smooth=1, exponent=2, loss_weight=1.0, ignore_index=-1):
        """DiceLoss.
        This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
        Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self, pred, target, **kwargs):
        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()
        # (B, d_1, d_2, ..., d_k) --> (B * d_1 * d_2 * ... * d_k,)
        target = target.view(-1).contiguous()
        assert pred.size(0) == target.size(
            0
        ), "The shape of pred doesn't match the shape of target"
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        pred = pred[valid_mask]

        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        target = F.one_hot(
            torch.clamp(target.long(), 0, num_classes - 1), num_classes=num_classes
        )

        total_loss = 0
        for i in range(num_classes):
            if i != self.ignore_index:
                num = torch.sum(torch.mul(pred[:, i], target[:, i])) * 2 + self.smooth
                den = (
                    torch.sum(
                        pred[:, i].pow(self.exponent) + target[:, i].pow(self.exponent)
                    )
                    + self.smooth
                )
                dice_loss = 1 - num / den
                total_loss += dice_loss
        loss = total_loss / num_classes
        return self.loss_weight * loss


@LOSSES.register_module()
class ForAINetv2UnifiedCriterion_XAwarequery(nn.Module):
    """统一语义和实例损失（nn.Module形式）"""
    def __init__(self, num_semantic_classes, sem_criterion, inst_criterion):
        super().__init__()  # 初始化nn.Module
        self.num_semantic_classes = num_semantic_classes
        self.sem_criterion = build_criteria(sem_criterion)  # 语义损失（如DiceLoss）
        self.inst_criterion = build_criteria(inst_criterion)  # 实例损失
        self.fp16_enabled = False  # 支持混合精度训练

    def forward(self, pred, insts):
        """前向传播计算损失（替代原__call__方法）"""
        pred_masks = pred['masks']
        pred_scores = pred['scores']
        
        sem_preds = []
        sem_gts = []  # 用字典存储语义GT，替代InstanceData_
        inst_gts = []  # 用字典存储实例GT，替代InstanceData_
        n = self.num_semantic_classes

        for i in range(len(pred_masks)):
            # 分离语义预测（后n个为语义分支）
            sem_preds.append(pred_masks[i][-n:, :])
            pred_masks[i] = pred_masks[i][:-n, :]
            pred_scores[i] = pred_scores[i][:-n, :]
            
            # 语义GT：用字典存储sp_masks，无额外数据结构
            sem_gt = {
                'sp_masks': insts[i]['sp_sem_masks']  # 对应GT的语义掩码
            }
            sem_gts.append(sem_gt)
            
            # 实例GT：用字典存储所有必要键，替代InstanceData_
            inst_gt = {
                'sp_masks': insts[i]['sp_inst_masks'],  # 实例掩码
                'labels_3d': insts[i]['labels_3d'],      # 实例标签
                'ratio_inspoint': insts[i]['ratio_inspoint']  # 点占比
            }
            
            # 生成query_masks（与原逻辑一致）
            n_gts = inst_gt['sp_masks'].shape[0]
            n_queries = pred_masks[i].shape[0]
            query_inslabel = insts[i]['query_inslabel']
            query_masks = torch.zeros(
                (n_queries, n_gts), dtype=torch.bool, device=inst_gt['sp_masks'].device
            )
            valid_queries = query_inslabel != -1
            query_masks[valid_queries, query_inslabel[valid_queries]] = True
            inst_gt['query_masks'] = query_masks  # 加入字典
            
            inst_gts.append(inst_gt)

        # 处理辅助输出（若有）
        sem_aux_outputs = None
        if 'aux_outputs' in pred:
            sem_aux_outputs = [self.prepare_aux_outputs(aux) for aux in pred['aux_outputs']]
        
        # 计算实例损失和语义损失
        loss = self.inst_criterion(pred, inst_gts)
        sem_loss = self.sem_criterion(
            {'masks': sem_preds, 'aux_outputs': sem_aux_outputs}, sem_gts
        )
        loss.update(sem_loss)  # 合并损失字典

        return loss

    def prepare_aux_outputs(self, aux_outputs):
        """处理辅助输出（与原逻辑一致）"""
        pred_masks = aux_outputs['masks']
        pred_scores = aux_outputs['scores']
        sem_preds = []
        n = self.num_semantic_classes
        
        for i in range(len(pred_masks)):
            sem_preds.append(pred_masks[i][-n:, :])
            pred_masks[i] = pred_masks[i][:-n, :]
            pred_scores[i] = pred_scores[i][:-n, :]
        
        return {'masks': sem_preds}
    

@LOSSES.register_module()
class S3DISSemanticCriterion(nn.Module):
    """S3DIS语义损失（适配现有所有基础损失，无额外数据结构）"""
    def __init__(self,
                 loss_weight,
                 seg_loss=[dict(
                     type='CrossEntropyLoss',  # 直接使用你注册的LOSSES名称
                     loss_weight=1.0)]):
        super().__init__()
        self.loss_weight = loss_weight
        # 用LOSSES.build加载你现有的损失函数（CrossEntropyLoss/FocalLoss等）
        self.seg_loss = build_criteria(seg_loss)
        self.fp16_enabled = False  # 支持混合精度训练

    def get_layer_loss(self, layer, aux_outputs, insts):
        """计算中间层损失（逻辑不变，仅适配字典化GT）"""
        pred_masks = aux_outputs['masks']
        seg_losses = []
        for pred_mask, gt_dict in zip(pred_masks, insts):
            # GT改为字典访问：gt_dict['sp_masks']（替代原gt_mask.sp_masks）
            gt_sem_mask = gt_dict['sp_masks']  # 形状：(n_classes+1, n_points_i)
            # 转换为类别索引（与现有损失函数的target格式匹配）
            gt_label = gt_sem_mask.float().argmax(0)  # 形状：(n_points_i,)
            # 预测调整为(N, C)格式，适配损失函数输入
            pred = pred_mask.T  # 形状：(n_points_i, n_classes)
            # 计算单样本损失（兼容所有现有损失函数）
            seg_loss = self.seg_loss(pred, gt_label)
            seg_losses.append(seg_loss)

        # 批次平均并乘权重
        seg_loss = self.loss_weight * torch.mean(torch.stack(seg_losses))
        return {f'layer_{layer}_seg_loss': seg_loss}

    def __call__(self, pred, insts):
        """计算最终损失（主逻辑）"""
        pred_masks = pred['masks']
        seg_losses = []
        for pred_mask, gt_dict in zip(pred_masks, insts):
            # 核心适配：GT从字典取sp_masks，无InstanceData_依赖
            gt_sem_mask = gt_dict['sp_masks']
            gt_label = gt_sem_mask.float().argmax(0)  # 类别索引（0~n_classes）
            pred = pred_mask.T  # 适配损失函数的(N, C)输入格式

            # 直接调用现有损失函数（自动兼容CrossEntropy/Focal/Dice等）
            seg_loss = self.seg_loss(pred, gt_label)
            seg_losses.append(seg_loss)

        # 计算主损失
        seg_loss = self.loss_weight * torch.mean(torch.stack(seg_losses))
        loss = {'last_layer_seg_loss': seg_loss}

        # 处理辅助输出（中间层损失）
        if 'aux_outputs' in pred:
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                layer_loss = self.get_layer_loss(i, aux_outputs, insts)
                loss.update(layer_loss)

        return loss
    

###########

# 导入之前修改好的无注册器版本HungarianMatcher（若需），此处用One2ManyMatcher
# 保留原有工具函数（dice_loss、get_iou_with_crop需确保已定义）
# 假设以下函数已在代码中存在（与原逻辑一致）：
# from your_utils import dice_loss, get_iou_with_crop



def get_iou_with_crop(inputs, targets, ratio):
    """IoU for to equal shape masks.

    Args:
        inputs (Tensor): of shape (n_gts, n_points).
        targets (Tensor): of shape (n_gts, n_points).
    
    Returns:
        Tensor: IoU of shape (n_gts,).
    """
    inputs = inputs.sigmoid()
    binarized_inputs = (inputs >= 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) / ratio + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


def dice_loss(inputs, targets):
    """Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs.
            Stores the binary classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
    
    Returns:
        Tensor: loss value.
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()




# ---------------------- 第一步：修改One2ManyMatcher（移除注册器+适配字典）----------------------
class One2ManyMatcher:
    """一对多匹配器（移除TASK_UTILS注册器，适配字典输入）"""
    def __init__(self):
        self.inf = 1e8

    @torch.no_grad()
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """
        适配字典输入：pred_instances/gt_instances为字典
        Args:
            pred_instances (dict): 预测实例，含'masks'键（shape: (n_queries, n_points)）
            gt_instances (dict): GT实例，含'labels'（shape: (n_gts,)）、'masks'（shape: (n_gts, n_points)）、
                                'query_masks'（shape: (n_queries, n_gts)）
        Returns:
            Tuple[Tensor, Tensor]: query_ids（所有查询ID）和gt_ids（匹配的GT ID，-1表示未匹配）
        """
        # 字典访问替代属性访问（原：gt_instances.query_masks → 现：gt_instances['query_masks']）
        query_masks = gt_instances['query_masks']
        n_queries, n_gts = query_masks.shape

        # 无GT时返回所有查询未匹配
        if n_gts == 0:
            query_ids = torch.arange(n_queries).to(query_masks.device)
            gt_ids = torch.full((n_queries,), -1, dtype=torch.long).to(query_masks.device)
            return query_ids, gt_ids

        # 初始化查询ID和GT ID（一对多匹配核心逻辑不变）
        query_ids = torch.arange(n_queries).to(query_masks.device)
        gt_ids = torch.full((n_queries,), -1, dtype=torch.long).to(query_masks.device)

        # 对每个查询，匹配query_masks中标记的GT（argmax取置信度最高的GT）
        matched_queries = torch.any(query_masks, dim=1)
        gt_ids[matched_queries] = torch.argmax(query_masks[matched_queries].float(), dim=1)

        return query_ids, gt_ids


# ---------------------- 第二步：修改InstanceCriterionForAI_OneToManyMatch（继承nn.Module+无注册器）----------------------
@LOSSES.register_module()
class InstanceCriterionForAI_OneToManyMatch(nn.Module):
    """一对多实例损失（移除注册器、InstanceData_，继承nn.Module）"""
    def __init__(self, 
                 matcher,  # 占位配置，因One2ManyMatcher无参数，格式为dict(type='One2ManyMatcher')
                 loss_weight, 
                 fix_dice_loss_weight, 
                 iter_matcher, 
                 fix_mean_loss=False):
        super().__init__()
        # 移除TASK_UTILS.build，直接实例化One2ManyMatcher（无参数）
        self.matcher = One2ManyMatcher()
        self.loss_weight = loss_weight  # [mask_bce, mask_dice, score]权重
        self.fix_dice_loss_weight = fix_dice_loss_weight
        self.iter_matcher = iter_matcher
        self.fix_mean_loss = fix_mean_loss
        self.fp16_enabled = False  # 支持混合精度训练

    def _get_src_permutation_idx(self, indices):
        """保持原索引排列逻辑"""
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_layer_loss(self, aux_outputs, insts, indices=None):
        """中间层损失计算（适配字典输入）"""
        pred_scores = aux_outputs['scores']
        pred_masks = aux_outputs['masks']

        # 未提供匹配结果时，重新计算匹配
        if indices is None:
            indices = []
            for i in range(len(insts)):
                # 字典包装预测实例（替代InstanceData_）
                pred_instances = {'masks': pred_masks[i]}
                # 字典包装GT实例（替代InstanceData_）
                gt_instances = {
                    'labels': insts[i]['labels_3d'],
                    'masks': insts[i]['sp_masks']
                }
                # 若有query_masks，加入GT字典
                if 'query_masks' in insts[i]:
                    gt_instances['query_masks'] = insts[i]['query_masks']
                # 调用一对多匹配器
                indices.append(self.matcher(pred_instances, gt_instances))

        # 计算损失（正负样本均参与，保持原逻辑）
        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for mask, score, inst_dict, (idx_q, idx_gt) in zip(
            pred_masks, pred_scores, insts, indices
        ):
            # 无GT实例时跳过
            if inst_dict['sp_masks'].shape[0] == 0:
                continue

            pred_mask = mask[idx_q]  # 匹配后的预测掩码（含正负样本）

            # 初始化GT掩码（负样本为全0）
            tgt_mask = torch.zeros_like(pred_mask)
            # 正样本（idx_gt != -1）分配真实掩码
            if (idx_gt != -1).any():
                positive_idx = idx_gt != -1
                # 字典访问GT掩码（原：inst.sp_masks → 现：inst_dict['sp_masks']）
                tgt_mask[positive_idx] = inst_dict['sp_masks'][idx_gt[positive_idx]].float()

            # 计算mask BCE和Dice损失（正负样本均计算）
            mask_bce_losses.append(F.binary_cross_entropy_with_logits(pred_mask, tgt_mask))
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask))

            # 计算score损失（正负样本均计算）
            if score is not None:
                pred_score = score[idx_q]
                tgt_score = torch.zeros_like(pred_score)  # 负样本score为0

                # 正样本score为IOU
                if (idx_gt != -1).any():
                    positive_idx = idx_gt != -1
                    with torch.no_grad():
                        # 字典访问ratio_inspoint（原：inst.ratio_inspoint → 现：inst_dict['ratio_inspoint']）
                        tgt_score[positive_idx] = get_iou_with_crop(
                            pred_mask[positive_idx],
                            tgt_mask[positive_idx],
                            inst_dict['ratio_inspoint'][idx_gt[positive_idx]]
                        ).unsqueeze(1)

                score_losses.append(F.mse_loss(pred_score, tgt_score))

        # 聚合损失（保持原逻辑）
        score_loss = torch.stack(score_losses).sum() / len(pred_masks) if len(score_losses) else 0.0

        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum() / len(pred_masks)

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4
            
            if self.fix_mean_loss:
                mask_bce_loss = mask_bce_loss * len(pred_masks) / len(mask_bce_losses)
                mask_dice_loss = mask_dice_loss * len(pred_masks) / len(mask_dice_losses)
        else:
            mask_bce_loss = 0.0
            mask_dice_loss = 0.0

        # 加权求和损失
        loss = (
            self.loss_weight[0] * mask_bce_loss +
            self.loss_weight[1] * mask_dice_loss +
            self.loss_weight[2] * score_loss
        )
        return loss

    def __call__(self, pred, insts):
        """主损失计算（适配字典输入）"""
        pred_scores = pred['scores']
        pred_masks = pred['masks']

        # 一对多匹配（字典输入）
        indices = []
        for i in range(len(insts)):
            pred_instances = {'masks': pred_masks[i]}
            gt_instances = {
                'labels': insts[i]['labels_3d'],
                'masks': insts[i]['sp_masks']
            }
            if 'query_masks' in insts[i]:
                gt_instances['query_masks'] = insts[i]['query_masks']
            indices.append(self.matcher(pred_instances, gt_instances))

        # 计算正负样本损失（保持原一对多逻辑）
        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for mask, score, inst_dict, (idx_q, idx_gt) in zip(
            pred_masks, pred_scores, insts, indices
        ):
            if inst_dict['sp_masks'].shape[0] == 0:
                continue

            pred_mask = mask[idx_q]
            tgt_mask = torch.zeros_like(pred_mask)

            if (idx_gt != -1).any():
                positive_idx = idx_gt != -1
                tgt_mask[positive_idx] = inst_dict['sp_masks'][idx_gt[positive_idx]].float()

            mask_bce_losses.append(F.binary_cross_entropy_with_logits(pred_mask, tgt_mask))
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask))

            if score is not None:
                pred_score = score[idx_q]
                tgt_score = torch.zeros_like(pred_score)

                if (idx_gt != -1).any():
                    positive_idx = idx_gt != -1
                    with torch.no_grad():
                        tgt_score[positive_idx] = get_iou_with_crop(
                            pred_mask[positive_idx],
                            tgt_mask[positive_idx],
                            inst_dict['ratio_inspoint'][idx_gt[positive_idx]]
                        ).unsqueeze(1)

                score_losses.append(F.mse_loss(pred_score, tgt_score))

        # 聚合损失（原逻辑不变）
        score_loss = torch.stack(score_losses).sum() / len(pred_masks) if len(score_losses) else 0.0

        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum()

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4
            
            if self.fix_mean_loss:
                mask_bce_loss = mask_bce_loss * len(pred_masks) / len(mask_bce_losses)
                mask_dice_loss = mask_dice_loss * len(pred_masks) / len(mask_dice_losses)
        else:
            mask_bce_loss = 0.0
            mask_dice_loss = 0.0

        # 主层损失
        loss = (
            self.loss_weight[0] * mask_bce_loss +
            self.loss_weight[1] * mask_dice_loss +
            self.loss_weight[2] * score_loss
        )

        # 累加辅助层损失
        if 'aux_outputs' in pred:
            if self.iter_matcher:
                indices = None
            for i, aux_outputs in enumerate(pred['aux_outputs']):
                loss += self.get_layer_loss(aux_outputs, insts, indices)

        return {'inst_loss': loss}