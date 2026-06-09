# 必须在最开始设置 multiprocessing 启动方法
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import torch
import numpy as np
from pathlib import Path
from pc_views import PCViews
from PIL import Image
import imageio
from sklearn.neighbors import NearestNeighbors
import os
import sys
import concurrent.futures
import time
from tqdm import tqdm
import logging
import traceback

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_single_plot(plot_dir, output_base_dir, gpu_id=None, num_view=4):
    """
    处理单个点云文件夹，生成12个视角的三通道图像和对应的标签图像
    
    Args:
        plot_dir (Path): 包含coord.npy和segment.npy的文件夹路径
        output_base_dir (Path): 基础输出目录，将创建images和masks子目录
        gpu_id (int, optional): 指定GPU ID，用于多GPU处理
    """
    start_time = time.time()
    plot_name = plot_dir.name
    logger.info(f"Starting processing {plot_name} on GPU {gpu_id if gpu_id is not None else 'CPU'}")
    
    try:
        # 检查必要的文件是否存在
        coord_path = plot_dir / "coord.npy"
        segment_path = plot_dir / "segment.npy"
        
        if not coord_path.exists() or not segment_path.exists():
            logger.warning(f"Missing files in {plot_dir}. Skipping.")
            return False, plot_name, 0
        
        # 1. 加载点云
        coords = np.load(coord_path)
        segment = np.load(segment_path)
        
        # 2. 准备多通道所需信息
        # 高度信息 (Z轴) - 保持原始值
        original_heights = coords[:, 2].copy()
        
        # 计算局部点密度 - 森林点云的第三通道
        radius = 0.5  # 50cm半径
        try:
            nbrs = NearestNeighbors(radius=radius, algorithm='kd_tree', n_jobs=1).fit(coords)
            density_counts = nbrs.radius_neighbors_graph(coords).sum(axis=1).A1
            
            # 归一化密度到0-1
            max_density = np.percentile(density_counts, 95)  # 避免异常值
            if max_density > 0:
                densities = np.minimum(density_counts / max_density, 1.0)
            else:
                densities = np.zeros_like(density_counts)
        except Exception as e:
            logger.warning(f"Failed to compute density for {plot_name}: {e}")
            densities = np.zeros(len(coords))
        
        # 3. 坐标预处理
        scale = np.abs(coords).max() / 30.0
        coords = coords / scale
        points_centered = coords - coords.mean(0)
        
        points_transformed = points_centered[:, [0, 2, 1]]  # [x, z, y]
        points_transformed[:, 2] = -points_transformed[:, 2]  # [x, z, -y]
        
        if points_transformed.ndim == 2:
            points_transformed = points_transformed[np.newaxis, ...]
        
        # 4. 转换为PyTorch张量 - 选择GPU
        if gpu_id is not None and torch.cuda.device_count() > gpu_id:
            device = torch.device(f"cuda:{gpu_id}")
            logger.debug(f"{plot_name} using GPU {gpu_id}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.debug(f"{plot_name} using device {device}")
            
        points = torch.from_numpy(points_transformed).float().to(device)
        
        # 5. 生成图像
        pcv = PCViews()
        depth_imgs, index = pcv.get_img_and_index(points)
        
        # 将数据转为CPU数组
        depth_imgs = depth_imgs.cpu().numpy()  # (12, 512, 512)
        index = index.cpu().numpy()            # (12, 512, 512)
        
        # 6. 处理标签映射
        bg_label = -1
        label_imgs = np.full(index.shape, bg_label, dtype=np.int32)
        
        for view_idx in range(index.shape[0]):
            idx_map = index[view_idx]
            valid_mask = (idx_map >= 0) & (idx_map < len(segment))        
            valid_indices = idx_map[valid_mask].astype(int)
            if valid_indices.size > 0 and valid_indices.max() < len(segment):
                label_imgs[view_idx][valid_mask] = segment[valid_indices]
        
        label_imgs = label_imgs + 1
        
        # 7. 高程图 - 最简单的线性归一化
        min_height = np.min(original_heights)
        max_height = np.max(original_heights)
        height_range = max_height - min_height
        
        # 8. 为每个视角生成三通道图像
        images_dir = output_base_dir / "images"
        masks_dir = output_base_dir / "masks"
        
        for view_idx in range(num_view):
            # 创建三通道图像 (512, 512, 3)
            multi_channel_img = np.zeros((512, 512, 3), dtype=np.uint8)
            
            # 通道1: 深度图 (标准深度可视化)
            depth_img = depth_imgs[view_idx].copy()
            depth_img = np.nan_to_num(depth_img, nan=0.0, posinf=0.0, neginf=0.0)
            
            valid_mask = depth_img > 0
            if np.any(valid_mask):
                valid_depths = depth_img[valid_mask]
                if valid_depths.size > 0:
                    min_d, max_d = np.min(valid_depths), np.max(valid_depths)
                    if max_d > min_d:
                        normalized = 254 * (1 - (valid_depths - min_d) / (max_d - min_d)) + 1
                        depth_vis = np.zeros_like(depth_img, dtype=np.uint8)
                        depth_vis[valid_mask] = normalized.astype(np.uint8)
                    else:
                        depth_vis = np.zeros_like(depth_img, dtype=np.uint8)
                else:
                    depth_vis = np.zeros_like(depth_img, dtype=np.uint8)
            else:
                depth_vis = np.zeros_like(depth_img, dtype=np.uint8)
            
            multi_channel_img[..., 0] = depth_vis  # R通道 = 深度
            
            # 通道2: 高程图 - 最简单的线性归一化
            ele_vis = np.zeros((512, 512), dtype=np.uint8)  # 背景=0(黑色)
            
            idx_map = index[view_idx]
            height_valid_mask = (idx_map >= 0) & (idx_map < len(original_heights))
            
            if np.any(height_valid_mask):
                valid_indices = idx_map[height_valid_mask].astype(int)
                heights = original_heights[valid_indices]
                
                if height_range > 0:
                    normalized_heights = (heights - min_height) / height_range
                    normalized_heights = np.clip(normalized_heights, 0, 1)
                    height_values = (normalized_heights * 255).astype(np.uint8)
                else:
                    height_values = np.zeros_like(heights, dtype=np.uint8)
                
                ele_flat = ele_vis.ravel()
                ele_flat[height_valid_mask.ravel()] = height_values
                ele_vis = ele_flat.reshape(512, 512)
            
            multi_channel_img[..., 1] = ele_vis  # G通道 = 高程
            
            # 通道3: 局部点密度 - 森林特征
            density_vis = np.zeros((512, 512), dtype=np.uint8)  # 背景=0(黑色)
            
            if np.any(height_valid_mask):
                valid_indices = idx_map[height_valid_mask].astype(int)
                if valid_indices.size > 0 and valid_indices.max() < len(densities):
                    density_values = densities[valid_indices]
                    density_vis_flat = density_vis.ravel()
                    density_vis_flat[height_valid_mask.ravel()] = (density_values * 255).astype(np.uint8)
                    density_vis = density_vis_flat.reshape(512, 512)
            
            multi_channel_img[..., 2] = density_vis  # B通道 = 点密度
            
            # 保存三通道图像
            image_filename = f"{plot_name}-view{view_idx+1:02d}.png"
            image_path = images_dir / image_filename
            Image.fromarray(multi_channel_img).save(image_path)
            
            # 保存标签图像
            label_filename = f"{plot_name}-view{view_idx+1:02d}.png"
            label_path = masks_dir / label_filename
            label_img = label_imgs[view_idx].astype(np.uint16)
            imageio.imwrite(str(label_path), label_img)
        
        processing_time = time.time() - start_time
        logger.info(f"Successfully processed {plot_name} in {processing_time:.2f} seconds")
        return True, plot_name, processing_time
        
    except Exception as e:
        logger.error(f"Error processing {plot_name}: {str(e)}")
        logger.error(traceback.format_exc())
        return False, plot_name, time.time() - start_time

def batch_process_with_multiprocessing(input_dirs, output_dir, max_workers=None, use_gpu=True):
    """
    使用多进程批量处理点云数据
    
    Args:
        input_dirs (list): 点云文件夹路径列表
        output_dir (str): 输出目录
        max_workers (int, optional): 最大工作进程数，默认为CPU核心数
        use_gpu (bool): 是否使用GPU加速
    """
    output_base_dir = Path(output_dir)
    images_dir = output_base_dir / "images"
    masks_dir = output_base_dir / "masks"
    
    # 创建输出目录
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # 确定工作进程数
    if max_workers is None:
        max_workers = max(1, os.cpu_count() - 1)  # 保留一个核心给系统
    
    logger.info(f"Starting batch processing with {max_workers} workers")
    logger.info(f"Total plots to process: {len(input_dirs)}")
    
    # 准备GPU分配
    gpu_assignments = []
    if use_gpu and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"Available GPUs: {num_gpus}")
        # 循环分配GPU
        gpu_assignments = [i % num_gpus for i in range(len(input_dirs))]
    else:
        gpu_assignments = [None] * len(input_dirs)
        logger.info("Using CPU only")
    
    # 创建进程池
    success_count = 0
    fail_count = 0
    total_time = 0
    
    # 使用tqdm创建进度条
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_plot = {
            executor.submit(process_single_plot, plot_dir, output_base_dir, gpu_assignments[idx]): plot_dir.name
            for idx, plot_dir in enumerate(input_dirs)
        }
        
        # 显示进度
        with tqdm(total=len(input_dirs), desc="Processing plots") as pbar:
            for future in concurrent.futures.as_completed(future_to_plot):
                plot_name = future_to_plot[future]
                try:
                    success, name, proc_time = future.result()
                    if success:
                        success_count += 1
                    else:
                        fail_count += 1
                    total_time += proc_time
                except Exception as e:
                    logger.error(f"Task for {plot_name} generated an exception: {e}")
                    fail_count += 1
                finally:
                    pbar.update(1)
    
    # 打印总结
    avg_time = total_time / max(1, success_count + fail_count)
    logger.info(f"\n{'='*50}")
    logger.info(f"Processing completed!")
    logger.info(f"Success: {success_count} plots")
    logger.info(f"Failed: {fail_count} plots")
    logger.info(f"Average processing time: {avg_time:.2f} seconds per plot")
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"Output images directory: {images_dir.absolute()}")
    logger.info(f"Output masks directory: {masks_dir.absolute()}")
    logger.info(f"{'='*50}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process point cloud data into multi-view images (Multiprocessing Version)')
    parser.add_argument('--input_root', type=str, required=True,
                        help='Root directory containing plot folders')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory for images and masks')
    parser.add_argument('--plots', nargs='+', default=None,
                        help='Specific plot folders to process. If not specified, process all subdirectories.')
    parser.add_argument('--max_workers', type=int, default=10,
                        help='Maximum number of worker processes (default: CPU cores - 1)')
    parser.add_argument('--no_gpu', action='store_true',
                        help='Disable GPU usage, use CPU only')
    
    args = parser.parse_args()
    
    # 获取输入目录
    input_root = Path(args.input_root)
    
    if not input_root.exists():
        logger.error(f"Input root directory {input_root} does not exist!")
        sys.exit(1)
    
    # 确定要处理的plot目录
    if args.plots:
        input_dirs = [input_root / plot_name for plot_name in args.plots]
    else:
        # 获取所有子目录
        input_dirs = [d for d in input_root.iterdir() if d.is_dir()]
        logger.info(f"Found {len(input_dirs)} plot directories in {input_root}")
    
    # 按名称排序以便一致处理
    input_dirs.sort(key=lambda x: x.name)
    
    # 检查输出目录
    output_dir = Path(args.output_dir)
    logger.info(f"Output directory: {output_dir.absolute()}")
    
    # 启动多进程处理
    batch_process_with_multiprocessing(
        input_dirs,
        args.output_dir,
        max_workers=args.max_workers,
        use_gpu=not args.no_gpu
    )