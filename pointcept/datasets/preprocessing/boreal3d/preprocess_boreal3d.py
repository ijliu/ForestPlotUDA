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
    dataset_root, scene_path, output_path, train_paths, test_paths, val_paths
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

    save_dict = dict(
        coord=coords.astype(np.float32),
        segment=segment.astype(np.uint8),
    )

    # Save processed data
    os.makedirs(output_path, exist_ok=True)
    for key in save_dict.keys():
        np.save(os.path.join(output_path, f"{key}.npy"), save_dict[key])



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

    # total = {0:0, 1:0, 2:0, 3:0}
    # for fn in scene_paths:
    #     nums = handle_process(config.dataset_root, fn, config.output_root,  train_paths, test_paths, val_paths)

    #     for key in nums.keys():
    #         total[key] = total[key] + nums[key]
    # print(total)
    # exit()

    # Preprocess data.
    print("Processing scenes...")
    pool = ProcessPoolExecutor(max_workers=config.num_workers)
    _ = list(
        pool.map(
            handle_process,
            repeat(config.dataset_root),
            scene_paths,
            repeat(config.output_root),
            repeat(train_paths),
            repeat(test_paths),
            repeat(val_paths),
        )
    )
