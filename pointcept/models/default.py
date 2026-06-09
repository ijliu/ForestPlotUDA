import torch
import clip
import torch.nn as nn
import torch_scatter
import torch_cluster
from torch_cluster import fps

import torch.nn.functional as F

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2batch
from .builder import MODELS, build_model

import numpy as np


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"].long(), input_dict['weight'])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"].long())
            return_dict["feat"] = feat
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict

# CLIP-style
@MODELS.register_module()
class DefaultSegmentorV3(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        # self.seg_head = (
        #     nn.Linear(backbone_out_channels, num_classes)
        #     if num_classes > 0
        #     else nn.Identity()
        # )
        self.proj_head = nn.Linear(
                backbone_out_channels, 512
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        with torch.no_grad():
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-B/32", device=device)
            prompts = clip.tokenize(["terrain points in a forest plot",
                                    "low-vegetation points in a forest plot",
                                    "wood points in a forest plot",
                                    "leaf points in a forest plot"]).cuda()
            text_prompts = model.encode_text(prompts)          # 4×d
            self.text_prompts = F.normalize(text_prompts, dim=1)

            self.logit_scale = model.logit_scale

            # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))  # 标量
            # self.logit_bias  = nn.Parameter(torch.zeros(num_classes))           # 向量，每类一个偏置

    def forward(self, input_dict, return_point=False):

        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        # seg_logits = self.seg_head(feat)

        feat = self.proj_head(feat)
        feat = feat / feat.norm(dim=-1, keepdim=True)

        self.text_prompts = self.text_prompts.to(feat.dtype)

        sim = (
            feat
            @ self.text_prompts.T
        )
        logit_scale = self.logit_scale.exp()
        seg_logits = logit_scale * sim

        # with torch.no_grad():
        #     confidences = torch.softmax(seg_logits, dim=1)
        #     sparse_feat = point['sparse_conv_feat']
        #     point_features = sparse_feat.features
  
        #     conf_mask = (confidences > 0.7).float()
        #     class_counts = torch.sum(conf_mask, dim=0)
        #     feature_sums = torch.matmul(point_features.t(), conf_mask).t()
        #     epsilon = 1e-8
        #     class_avg_features = feature_sums / (class_counts.unsqueeze(1) + epsilon)

        # combined_tensor = torch.cat([class_avg_features, text_prompts], dim=1)

        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"].long())
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"].long())
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict

# V4 中使用的一些类和函数
def MLP(channels, activation=nn.LeakyReLU(0.2), bn_momentum=0.1, bias=True):
    return nn.Sequential(
        *[
            nn.Sequential(
                nn.Linear(channels[i - 1], channels[i], bias=bias),
                FastBatchNorm1d(channels[i], momentum=bn_momentum),
                activation,
            )
            for i in range(1, len(channels))
        ]
    )

class FastBatchNorm1d(nn.Module):
    def __init__(self, num_features, momentum=0.1, **kwargs):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features, momentum=momentum, **kwargs)

    def _forward_dense(self, x):
        return self.batch_norm(x.permute(0, 2, 1)).permute(0, 2, 1)

    def _forward_sparse(self, x):
        """ Batch norm 1D is not optimised for 2D tensors. The first dimension is supposed to be
        the batch and therefore not very large. So we introduce a custom version that leverages BatchNorm1D
        in a more optimised way
        """
        x = x.unsqueeze(2)
        x = x.transpose(0, 2)
        x = self.batch_norm(x)
        x = x.transpose(0, 2)
        return x.squeeze(dim=2)

    def forward(self, x):
        if x.dim() == 2:
            return self._forward_sparse(x)
        elif x.dim() == 3:
            return self._forward_dense(x)
        else:
            raise ValueError("Non supported number of dimensions {}".format(x.dim()))


class Seq(nn.Sequential):
    def __init__(self):
        super().__init__()
        self._num_modules = 0

    def append(self, module):
        self.add_module(str(self._num_modules), module)
        self._num_modules += 1
        return self


from torch_scatter import scatter

def discriminative_loss(
    embedding_logits: torch.Tensor,
    instance_labels: torch.Tensor,
    batch: torch.Tensor,
    feature_dim,
):
    loss = []
    loss_var = []
    loss_dist = []
    loss_reg = []
    batch_size = torch.unique(batch) #batch[-1] + 1
    for s in batch_size: #range(batch_size):
        batch_mask = batch == s
        sample_gt_instances = instance_labels[batch_mask]
        sample_embed_logits = embedding_logits[batch_mask]
        sample_loss, sample_loss_var, sample_loss_dist, sample_loss_reg = discriminative_loss_single(sample_embed_logits, sample_gt_instances, feature_dim)
        loss.append(sample_loss)
        loss_var.append(sample_loss_var)
        loss_dist.append(sample_loss_dist)
        loss_reg.append(sample_loss_reg)
    loss = torch.stack(loss)
    loss_var = torch.stack(loss_var)
    loss_dist = torch.stack(loss_dist)
    loss_reg = torch.stack(loss_reg)
    return {"ins_loss": torch.mean(loss), "ins_var_loss": torch.mean(loss_var), "ins_dist_loss": torch.mean(loss_dist), "ins_reg_loss": torch.mean(loss_reg)}
    #return torch.mean(loss), torch.mean(loss_var), torch.mean(loss_dist), torch.mean(loss_reg)
    
def discriminative_loss_single(
    prediction,
    correct_label,
    feature_dim,
    delta_v = 0.5,
    delta_d = 1.5,
    param_var = 1.,
    param_dist = 1.,
    param_reg = 0.001,
):

    ''' Discriminative loss for a single prediction/label pair.
    :param prediction: inference of network
    :param correct_label: instance label
    :feature_dim: feature dimension of prediction
    :param delta_v: cutoff variance distance
    :param delta_d: curoff cluster distance
    :param param_var: weight for intra cluster variance
    :param param_dist: weight for inter cluster distances
    :param param_reg: weight regularization
    '''
    ### Reshape so pixels are aligned along a vector
    reshaped_pred = torch.reshape(prediction, (-1, feature_dim))
    ### Count instances
    unique_labels, unique_id, counts = torch.unique(correct_label, return_inverse=True, return_counts=True)
    #counts = tf.cast(counts, tf.float32)
    
    num_instances = unique_labels.size()
    #segmented_sum = tf.unsorted_segment_sum(reshaped_pred, unique_id, num_instances)
    #not sure
    segmented_sum = scatter(reshaped_pred, unique_id, dim=0, reduce="sum")

    mu = torch.div(segmented_sum, (torch.reshape(counts, (-1, 1)) + 1e-8 ))
    unique_id_t = unique_id.unsqueeze(1)
    
    unique_id_t = unique_id_t.expand(unique_id_t.size()[0], mu.size()[-1])
    mu_expand = torch.gather(mu, 0, unique_id_t)

    ### Calculate l_var
    #distance = tf.norm(tf.subtract(mu_expand, reshaped_pred), axis=1)
    #tmp_distance = tf.subtract(reshaped_pred, mu_expand)
    tmp_distance = reshaped_pred - mu_expand
    distance = torch.norm(tmp_distance, p=1, dim=1)
    distance = torch.subtract(distance, delta_v)
    distance = torch.clip(distance, min=0.)
    distance = torch.square(distance)
    l_var = scatter(distance, unique_id, dim=0, reduce="sum")
    l_var = torch.div(l_var, counts + 1e-8)
    l_var = torch.sum(l_var)
    l_var = torch.div(l_var, float(num_instances[0]))

    ### Calculate l_dist

    # Get distance for each pair of clusters like this:
    #   mu_1 - mu_1
    #   mu_2 - mu_1
    #   mu_3 - mu_1
    #   mu_1 - mu_2
    #   mu_2 - mu_2
    #   mu_3 - mu_2
    #   mu_1 - mu_3
    #   mu_2 - mu_3
    #   mu_3 - mu_3

    mu_interleaved_rep = mu.repeat(num_instances[0], 1)
    mu_band_rep = mu.repeat(1, num_instances[0])
    mu_band_rep = torch.reshape(mu_band_rep, (num_instances[0] * num_instances[0], feature_dim))

    mu_diff = torch.subtract(mu_band_rep, mu_interleaved_rep)
    # Filter out zeros from same cluster subtraction
    eye = torch.eye(num_instances[0])
    #zero = torch.zeros(1, dtype=torch.float32)
    diff_cluster_mask = torch.eq(eye, 0)
    diff_cluster_mask = torch.reshape(diff_cluster_mask, (-1,))
    mu_diff_bool = mu_diff[diff_cluster_mask]
    #intermediate_tensor = tf.reduce_sum(tf.abs(mu_diff),axis=1)
    #zero_vector = tf.zeros(1, dtype=tf.float32)
    #bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
    #mu_diff_bool = tf.boolean_mask(mu_diff, bool_mask)

    mu_norm = torch.norm(mu_diff_bool, p=1, dim=1)
    mu_norm = torch.subtract(torch.mul(delta_d, 2.0), mu_norm)
    mu_norm = torch.clip(mu_norm, min=0.)
    mu_norm = torch.square(mu_norm)

    l_dist = torch.mean(mu_norm)
    
    if num_instances[0]==1:
        l_dist = torch.tensor(0).cuda()
    ### Calculate l_reg
    l_reg = torch.mean(torch.norm(mu, p=1, dim=1))

    if num_instances[0]==0:
        l_var = torch.tensor(0).cuda()
        l_dist = torch.tensor(0).cuda()
        l_reg = torch.tensor(0).cuda()
    
    param_scale = 1.
    l_var = param_var * l_var
    l_dist = param_dist * l_dist
    l_reg = param_reg * l_reg

    loss = param_scale * (l_var + l_dist + l_reg)

    #if torch.is_tensor(loss):
    #    loss = loss.item()
    #if torch.is_tensor(l_var):
    #    l_var = l_var.item()
    #if torch.is_tensor(l_dist):
    #    l_dist = l_dist.item()
    #if torch.is_tensor(l_reg):
    #    l_reg = l_reg.item()

    return loss, l_var, l_dist, l_reg



# ForestFormer3D
@MODELS.register_module()
class DefaultSegmentorV4(nn.Module):
    def __init__(self, backbone=None, decoder=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.decoder = build_model(decoder)

        # 一些参数, 暂时放在这里
        self.query_point_num = 300

        num_channels = decoder.in_channels

        self.Embed = Seq().append(MLP([num_channels, num_channels], bias=False))
        self.Embed.append(torch.nn.Linear(num_channels, 5))

        self.BiSemantic = (
            Seq()  
            .append(MLP([num_channels, num_channels], bias=False))  
            .append(torch.nn.Linear(num_channels, 2))  
            .append(torch.nn.LogSoftmax(dim=-1))  
        )

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]

        offset = input_dict['offset']
        batch = offset2batch(offset)

        encoder_out = self.backbone(input_dict)

        # 为了适配 Forestformer3D 中的解码器,这里把特征按批次拆分
        x = []
        coords = []
        segments = []
        instances = []
        for batch_id in torch.unique(batch):
            mask = torch.where(batch == batch_id)
            feat = encoder_out[mask]
            x.append(feat)

            coords.append(input_dict['coord'][mask])
            segments.append(input_dict['segment'][mask])
            instances.append(input_dict['instance'][mask])

        embed_logits = [self.Embed(y) for y in x]
        bi_semantic_logits = [self.BiSemantic(y) for y in x]

        # Initialize cumulative losses
        total_discriminative_loss = 0
        total_semantic_loss_bi = 0 
        batch_size = len(offset)

        ## 训练两个 MLP 部分
        for i in range(batch_size):
            current_coord = coords[i]
            current_segment = segments[i]
            current_instance = instances[i]

            n_voxels = current_coord.shape[0]

            pts_instance_mask = current_instance
            device = pts_instance_mask.device
            valid_mask = current_instance > 0 
            valid_voxel_indices = torch.where(valid_mask)[0]

            if len(valid_voxel_indices) == 0:
                continue

            voxel_instance_labels = current_instance[valid_mask]
            filtered_embed_logits = embed_logits[i][valid_voxel_indices]
            batch_idx = torch.full_like(voxel_instance_labels, i, device=device)

            discriminative_losses = discriminative_loss(
                filtered_embed_logits,
                voxel_instance_labels,
                batch_idx,
                5
            )
            total_discriminative_loss += discriminative_losses.get("ins_loss", 0)

            bi_semantic_logit = bi_semantic_logits[i]
            bi_y = (current_instance > 0).long()  

            semantic_loss_bi = torch.nn.functional.nll_loss(
                bi_semantic_logit,
                bi_y.to(torch.int64)
            )
            total_semantic_loss_bi += semantic_loss_bi

        total_discriminative_loss /= batch_size
        total_semantic_loss_bi /= batch_size

        loss_final = {
            'discriminative_loss': total_discriminative_loss,
            'semantic_loss_bi': total_semantic_loss_bi
        }

        ############# 选取 Query ############

        queries = []
        queries_inslabel = []
        queries_idx = []

        # if self.prepare_epoch:
            # if self.epoch > self.prepare_epoch:
        if True:
            if True:
                total_qscore_loss = 0  # 预留qscore损失变量（若需使用可补充计算）

                for i in range(batch_size):
                    current_instance = instances[i]  # 当前批次实例标签
                    current_coord = coords[i]        # 当前批次体素坐标
                    device = current_instance.device
                    n_voxels = current_coord.shape[0]

                    # 1. 生成体素索引（因每个体素1个点，voxel_superpoints直接为体素序号）
                    # 替代原inverse_mapping逻辑，无需映射，直接用0~n_voxels-1作为体素标识
                    voxel_superpoints = torch.arange(n_voxels, device=device)

                    # 2. 筛选有效目标体素（instance > 0）
                    instance_mask = current_instance > 0
                    valid_voxel_indices = torch.unique(voxel_superpoints[instance_mask])

                    # 若有效体素不足10个，跳过当前批次query挑选
                    if valid_voxel_indices.numel() < 10:
                        queries.append([])
                        queries_inslabel.append([])
                        queries_idx.append([])
                        continue

                    # 3. 体素级实例标签（直接使用当前批次的instance标签）
                    voxel_instance_labels = current_instance  # 每个体素对应一个实例ID

                    # 4. 筛选语义预测为前景（wood_class=1）的体素（二值语义输出：1=前景）
                    semantic_predictions_bi = torch.argmax(bi_semantic_logits[i], dim=1)
                    wood_class = 1  # 前景类别（与二值语义标签一致）
                    tree_indices = torch.where(semantic_predictions_bi == wood_class)[0]

                    # 5. FPS采样挑选query（从前景体素中采样）
                    if tree_indices.numel() == 0:
                        queries.append([])
                        queries_inslabel.append([])
                        queries_idx.append([])
                        continue

                    # 采样比例：不超过设定的query_point_num，最多采样全部前景体素
                    sample_ratio = min(
                        self.query_point_num / tree_indices.numel(),
                        torch.tensor(1.0, device=device)
                    )
                    # 生成批次标记（FPS函数所需）
                    batch_tensor_4 = torch.zeros(tree_indices.numel(), dtype=torch.long, device=device)
                    # FPS采样（基于嵌入特征）
                    topk_indices_4 = fps(
                        embed_logits[i][tree_indices],  # 前景体素的嵌入特征
                        batch_tensor_4,
                        ratio=sample_ratio
                    )
                    # 得到最终选中的体素索引（对应原批次的体素序号）
                    selected_indices_case4 = tree_indices[topk_indices_4]

                    # 6. 收集query相关信息（体素特征、实例标签、索引）
                    # query特征：使用encoder_out的特征（与原代码x[i]一致）
                    queries.append(x[i][selected_indices_case4])
                    # query的实例标签：选中体素对应的instance ID
                    queries_inslabel.append(voxel_instance_labels[selected_indices_case4])
                    # query的体素索引
                    queries_idx.append(selected_indices_case4)

                # 7. 过滤空query，保留有效批次
                if not all(len(q) == 0 for q in queries):
                    # 筛选非空query及对应原始批次信息
                    filtered_results = [
                        (x[i], queries[i], instances[i], queries_inslabel[i], queries_idx[i], i)
                        for i in range(len(queries))
                        if len(queries[i]) > 0
                    ]
                    if not filtered_results:
                        # 无有效query，跳过解码器逻辑
                        return loss_final

                    # 解包筛选后的数据（保持批次对应关系）
                    x_filtered, queries_filtered, instances_filtered, queries_inslabel_filtered, queries_idx_filtered, original_indices = zip(*filtered_results)
                    x_filtered = list(x_filtered)
                    queries_filtered = list(queries_filtered)
                    original_indices = list(original_indices)  # 记录原始批次索引

                    # 8. 解码器推理
                    x_decoded = self.decoder(x_filtered, queries_filtered)

                    # 9. 构建GT标签（完全基于segment和instance，适配criterion要求）
                    sp_gt_instances = []
                    for idx in range(len(x_filtered)):
                        # 当前筛选批次对应的原始批次索引
                        orig_i = original_indices[idx]
                        # 原始数据：当前批次的instance和segment（与体素一一对应）
                        current_instance = instances[orig_i]  # 0-N，0=非树，1-N=树实例（long类型）
                        current_segment = segments[orig_i]  # 0=非树，1=树（0/1标签，long类型）
                        device = current_instance.device
                        n_voxels = current_instance.shape[0]  # 当前批次体素数（n_points）

                        # ---------------------- 修正1：sp_sem_masks（语义掩码）- 适配criterion ----------------------
                        # criterion要求：(num_semantic_classes + 1, n_voxels)，float类型
                        # 注：num_semantic_classes是criterion的输入参数（假设为2，对应非树/树+1个背景类？需与你的配置一致）
                        # 若你的num_semantic_classes=2，则num_semantic_classes + 1 = 3（匹配criterion的(n_classes+1)维）
                        num_semantic_classes = 2  # 必须与criterion初始化时的num_semantic_classes一致！
                        sp_sem_masks = torch.zeros((num_semantic_classes + 1, n_voxels), device=device, dtype=torch.float32)
                        
                        # current_segment是0=非树，1=树，映射到语义类别索引（0和1）
                        # 第0维：背景类（若criterion需要），第1维：非树，第2维：树（根据你的类别定义调整）
                        for voxel_idx in range(n_voxels):
                            seg_label = current_segment[voxel_idx].item()
                            if seg_label == 0:
                                sp_sem_masks[1, voxel_idx] = 1.0  # 非树 → 第1维
                            elif seg_label == 1:
                                sp_sem_masks[2, voxel_idx] = 1.0  # 树 → 第2维
                            # 第0维：背景类（若无需可设为0，或根据criterion要求调整）

                        # ---------------------- 修正2：sp_inst_masks（实例掩码）- 适配criterion ----------------------
                        # 1. 提取前景实例（instance > 0）
                        foreground_inst_ids = torch.unique(current_instance[current_instance > 0]).long()  # 1-N的树实例（long类型）
                        n_gts_i = len(foreground_inst_ids)  # 前景实例数（n_gts_i）

                        # 2. 构建实例掩码：(n_gts_i, n_voxels)，float类型（criterion要求）
                        sp_inst_masks = torch.zeros((n_gts_i, n_voxels), device=device, dtype=torch.float32)
                        for inst_idx, inst_id in enumerate(foreground_inst_ids):
                            # 找到该实例对应的体素，标记为1.0
                            sp_inst_masks[inst_idx, current_instance == inst_id] = 1.0

                        # ---------------------- 3. labels_3d（前景实例标签）- 保持正确 ----------------------
                        labels_3d = foreground_inst_ids  # 形状：(n_gts_i,)，long类型（符合要求）

                        # ---------------------- 4. ratio_inspoint（体素占比）- 保持正确 ----------------------
                        ratio_inspoint = torch.ones_like(labels_3d, dtype=torch.float32, device=device)  # 每个实例占比=1.0

                        # ---------------------- 5. query_inslabel（查询实例标签）- 保持正确 ----------------------
                        query_inslabel = queries_inslabel_filtered[idx].long()  # 确保为long类型，与criterion匹配

                        # ---------------------- 6. 构建最终GT实例（完全对齐criterion要求） ----------------------
                        gt_instances = {
                            'sp_sem_masks': sp_sem_masks,  # 语义掩码：(num_semantic_classes+1, n_voxels)
                            'sp_inst_masks': sp_inst_masks,  # 实例掩码：(n_gts_i, n_voxels)
                            'labels_3d': labels_3d,  # 实例标签：(n_gts_i,)
                            'ratio_inspoint': ratio_inspoint,  # 点占比：(n_gts_i,)
                            'query_inslabel': query_inslabel  # query实例标签：与queries数量一致
                        }

                    ##################################################
                    #################################################
                    #################################################
                    ###################################################
                    ###################################################
                    #################################################
                    #################################################
                        print(sp_sem_masks.shape)
                        exit()

                        sp_gt_instances.append(gt_instances)

                    # 10. 计算解码器损失并更新（此时输入完全匹配criterion要求）
                    decoder_loss = self.criteria(x_decoded, sp_gt_instances)

                    print(decoder_loss)
                    exit()

                    loss_final.update(decoder_loss)

        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)




@MODELS.register_module()
class DINOEnhancedSegmentor(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone) if backbone is not None else None
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.backbone is not None and self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        if self.backbone is not None:
            if self.freeze_backbone:
                with torch.no_grad():
                    point = self.backbone(point)
            else:
                point = self.backbone(point)
            point_list = [point]
            while "unpooling_parent" in point_list[-1].keys():
                point_list.append(point_list[-1].pop("unpooling_parent"))
            for i in reversed(range(1, len(point_list))):
                point = point_list[i]
                parent = point_list[i - 1]
                assert "pooling_inverse" in point.keys()
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = point_list[0]
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = [point.feat]
        else:
            feat = []
        dino_coord = input_dict["dino_coord"]
        dino_feat = input_dict["dino_feat"]
        dino_offset = input_dict["dino_offset"]
        idx = torch_cluster.knn(
            x=dino_coord,
            y=point.origin_coord,
            batch_x=offset2batch(dino_offset),
            batch_y=offset2batch(point.origin_offset),
            k=1,
        )[1]

        feat.append(dino_feat[idx])
        feat = torch.concatenate(feat, dim=-1)
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
