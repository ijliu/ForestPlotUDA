"""
Preprocessing Script for ScanNet 20/200

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import argparse
import json
import laspy
import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from scipy.spatial import KDTree

def read_lasfile(file_path):
    with laspy.open(file_path) as fh:
        # print('Points from Header:', fh.header.point_count)
        las = fh.read()
        instance = las.treeID
        classification_mapping = {
            1: 1,    # Low-vegetation -> 1
            2: 0,    # Terrain -> 0
            4: 2,    # Stem -> 2
            5: 3,    # Live branches -> 3
            6: 2     # Woody branches -> 2
        }
        original_class = las.classification
        
        converted_label = np.full_like(original_class, -1, dtype=int)
        for orig_code, target_label in classification_mapping.items():
            converted_label[original_class == orig_code] = target_label
        
        valid_mask = converted_label != -1  # 只保留有效映射的点
        xyz = np.vstack((las.x[valid_mask], 
                         las.y[valid_mask], 
                         las.z[valid_mask])).transpose()
        instance = instance[valid_mask]
        # 将 instance 中的点标签连续,0 是非树,1-N是树

        # 提取树实例的掩码（大于0的部分）
        tree_mask = instance > 0

        if np.any(tree_mask):
            # 对树实例标签去重并映射为连续索引（0开始），再加1转为1-N
            _, inverse = np.unique(instance[tree_mask], return_inverse=True)
            instance[tree_mask] = inverse + 1  # 直接在原数组上更新，0保持不变

        segment = np.zeros_like(instance)
        mask = np.where(instance > 0)
        segment[mask] = 1

    return xyz, np.array(instance, dtype=np.int64), np.array(segment, dtype=np.int64)


def save_to_txt(xyz, label, instance, save_path):
    """将xyz和label合并并保存为TXT文件"""
    # 合并xyz（N,3）和label（N,1）为（N,4）的数组
    data = np.hstack((xyz, label, instance))
    
    # 保存为TXT，格式：x y z label（空格分隔）
    np.savetxt(
        save_path,
        data,
        delimiter=' '  # 空格分隔
    )
    print(f"已保存到: {save_path}")

def handle_process(
    dataset_root, scene_path, output_path, radius, dev_paths, test_paths
):
    path_names = Path(scene_path).parts
    scene_id = path_names[0] + "_" + path_names[1][:-4]

    if scene_path in dev_paths:
        output_path = os.path.join(output_path, "train", f"{scene_id}")
        split_name = "train"
    elif scene_path in test_paths:
        output_path = os.path.join(output_path, "test", f"{scene_id}")
        split_name = "test"
    else:
        pass

    print(f"Processing: {scene_id} in {split_name}")

    coords, instance, segment = read_lasfile(Path(dataset_root, scene_path))

    # 防止精度丢失
    coords = coords - coords.min(0)

    print(segment.dtype)
    print(coords.dtype)
    print(instance.dtype)

    unique_labels = np.unique(segment)
    counts = np.bincount(segment)
    result = {int(lbl): int(counts[lbl]) for lbl in unique_labels}

    if radius > 0:
        max_radius = 40
        delta_r = 1
        min_points = 10000

        N = coords.shape[0]
        point_index = np.arange(N)

        # 提取xy坐标（仅用xy构建KDTree，忽略z轴）
        xy_coords = coords[:, :2]
        # 构建xy坐标的KDTree
        kdtree = KDTree(xy_coords)

        x_max = xy_coords[:, 0].max()
        y_max = xy_coords[:, 1].max()

        center_x = []
        cx = 0.0
        while cx <= x_max:
            center_x.append(cx)
            cx += radius

        center_y = []
        cy = 0.0
        while cy <= y_max:
            center_y.append(cy)
            cy += radius

        index = 0
        for cx in center_x:
            for cy in center_y:
                center = (cx, cy)
                current_r = radius  # 初始化当前半径
                valid_data = False  # 标记是否找到满足条件的数据

                while current_r <= max_radius:
                    # 根据当前半径查询点
                    block_index = kdtree.query_ball_point(center, r=current_r)
                    tmp_coord = coords[block_index]
                    
                    # 检查coord是否有内容
                    if len(tmp_coord) == 0:
                        # 无内容，直接跳出循环（扩大半径也可能无点）
                        break
                    
                    # 检查点数量是否满足要求
                    if len(tmp_coord) >= min_points:
                        # 点数量足够，获取对应数据
                        tmp_segment = segment[block_index]
                        tmp_point_index = point_index[block_index]
                        valid_data = True
                        break
                    else:
                        # 点数量不足，扩大半径继续尝试
                        current_r += delta_r
                
                # 过滤无效数据（无内容或未达到最小点数量）
                if not valid_data:
                    continue

                save_dict = dict(
                    coord=tmp_coord.astype(np.float32),
                    segment=tmp_segment.astype(np.uint8),
                    point_index = tmp_point_index.astype(np.long)
                )

                os.makedirs(output_path + f"-{index:06d}", exist_ok=True)
                for key in save_dict.keys():
                    np.save(os.path.join(output_path + f"-{index:06d}", f"{key}.npy"), save_dict[key])

                # save_to_txt(tmp_coord, tmp_segment.reshape(-1, 1), f"{output_path}-{index:06d}/{scene_id}" + ".txt")

                index += 1

    else:
        save_dict = dict(
            coord=coords.astype(np.float32),
            segment=segment.astype(np.uint16),
            instance=instance.astype(np.uint16),
        )

        # print(np.unique(segment))
        unique_labels = np.unique(segment)
        counts = np.bincount(segment)
        result = {int(lbl): int(counts[lbl]) for lbl in unique_labels}
        
        # Save processed data
        os.makedirs(output_path, exist_ok=True)
        for key in save_dict.keys():
            np.save(os.path.join(output_path, f"{key}.npy"), save_dict[key])

        save_to_txt(coords, segment.reshape(-1, 1), instance.reshape(-1, 1), os.path.join(output_path, f"{scene_id}.txt"))

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the ScanNet dataset containing scene folders",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val folders will be located",
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    parser.add_argument(
        '--radius',
        default=-1,
        required=True,
        type=int,
    )

    config = parser.parse_args()
    df = pd.read_csv(
        Path(config.dataset_root) / "data_split_metadata.csv",
    )

    dev_paths = df[df['split'] == 'dev']['path'].tolist()
    test_paths = df[df['split'] == 'test']['path'].tolist()

    # Create output directories
    train_output_dir = os.path.join(config.output_root, "train")
    os.makedirs(train_output_dir, exist_ok=True)
    val_output_dir = os.path.join(config.output_root, "val")
    os.makedirs(val_output_dir, exist_ok=True)
    test_output_dir = os.path.join(config.output_root, "test")
    os.makedirs(test_output_dir, exist_ok=True)

    # Load scene paths
    scene_paths = sorted(dev_paths + test_paths)

    total = {0:0, 1:0, 2:0, 3:0}
    for fn in scene_paths:
        nums = handle_process(config.dataset_root, fn, config.output_root, config.radius, dev_paths, test_paths)

        for key in nums.keys():
            total[key] = total[key] + nums[key]
    print(total)
    exit()

    # Preprocess data.
    print("Processing scenes...")
    pool = ProcessPoolExecutor(max_workers=config.num_workers)
    _ = list(
        pool.map(
            handle_process,
            repeat(config.dataset_root),
            scene_paths,
            repeat(config.output_root),
            repeat(dev_paths),
            repeat(test_paths)
        )
    )
