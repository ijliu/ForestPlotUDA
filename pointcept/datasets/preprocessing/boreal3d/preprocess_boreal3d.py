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
        classification_mapping = {
            0: 1,    # Low-vegetation -> 1
            1: 0,    # Terrain -> 0
            2: 3,    # Wood -> 2
            3: 2,    # Leaf -> 3
        }
        original_class = las.semantic_id
        
        converted_label = np.full_like(original_class, -1, dtype=int)
        for orig_code, target_label in classification_mapping.items():
            converted_label[original_class == orig_code] = target_label
        
        valid_mask = converted_label != -1  
        xyz = np.vstack((las.x[valid_mask], 
                         las.y[valid_mask], 
                         las.z[valid_mask])).transpose()
        label = converted_label[valid_mask]

    return xyz, label


def save_to_txt(xyz, label, save_path):
    """将xyz和label合并并保存为TXT文件"""
    # 合并xyz（N,3）和label（N,1）为（N,4）的数组
    data = np.hstack((xyz, label))
    
    # 保存为TXT，格式：x y z label（空格分隔）
    np.savetxt(
        save_path,
        data,
        fmt='%.6f %.6f %.6f %d',  # xyz保留6位小数，label为整数
        delimiter=' '  # 空格分隔
    )
    print(f"已保存到: {save_path}")

def handle_process(
    dataset_root, scene_path, output_path, radius, train_paths, test_paths, val_paths
):
    scene_id = Path(scene_path).parts[-1]
    if scene_path in train_paths:
        output_path = os.path.join(output_path, "train", f"{scene_id}")
        split_name = "train"
    elif scene_path in test_paths:
        output_path = os.path.join(output_path, "test", f"{scene_id}")
        split_name = "test"
    elif scene_path in val_paths:
        output_path = os.path.join(output_path, "val", f"{scene_id}")
        split_name = "val"
    else:
        print("ERROR")
        exit()

    print(f"Processing: {scene_id} in {split_name}")

    coords, segment = read_lasfile(Path(dataset_root, scene_path, "UAV.laz"))

    # 防止精度丢失
    coords = coords - coords.min(0)

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
                    point_index = tmp_point_index.astype(np.uint8)
                )

                os.makedirs(output_path + f"-{index:06d}", exist_ok=True)
                for key in save_dict.keys():
                    np.save(os.path.join(output_path + f"-{index:06d}", f"{key}.npy"), save_dict[key])

                # save_to_txt(tmp_coord, tmp_segment.reshape(-1, 1), f"{output_path}-{index:06d}/{scene_id}" + ".txt")

                index += 1

    else:
        save_dict = dict(
            coord=coords.astype(np.float32),
            segment=segment.astype(np.uint8),
        )

        # print(np.unique(segment))
        unique_labels = np.unique(segment)
        counts = np.bincount(segment)
        result = {int(lbl): int(counts[lbl]) for lbl in unique_labels}
        save_to_txt(coords, segment.reshape(-1, 1), scene_id + ".txt")

        # Save processed data
        os.makedirs(output_path, exist_ok=True)
        for key in save_dict.keys():
            np.save(os.path.join(output_path, f"{key}.npy"), save_dict[key])

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
    meta_root = Path(config.dataset_root) / "meta_data"

    print(meta_root)
    train_txt = meta_root / "train.txt"
    test_txt = meta_root / "test.txt"
    val_txt = meta_root / "val.txt"

    train_paths = ["laz/" + name.rstrip() for name in open(train_txt).readlines()]
    test_paths = ["laz/" + name.rstrip() for name in open(test_txt).readlines()]
    val_paths = ["laz/" + name.rstrip() for name in open(val_txt).readlines()]

    # Create output directories
    train_output_dir = os.path.join(config.output_root, "train")
    os.makedirs(train_output_dir, exist_ok=True)
    val_output_dir = os.path.join(config.output_root, "val")
    os.makedirs(val_output_dir, exist_ok=True)
    test_output_dir = os.path.join(config.output_root, "test")
    os.makedirs(test_output_dir, exist_ok=True)

    # Load scene paths
    scene_paths = sorted(train_paths + test_paths + val_paths)

    total = {0:0, 1:0, 2:0, 3:0}
    for fn in scene_paths:
        nums = handle_process(config.dataset_root, fn, config.output_root, config.radius, train_paths, test_paths, val_paths)

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
