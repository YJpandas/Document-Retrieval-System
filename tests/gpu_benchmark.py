#!/usr/bin/env python
"""
GPU加速测试脚本，用于测试和比较GPU和CPU的检索性能
"""
import time
import argparse
import numpy as np
from typing import Dict, List, Any
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils.logger import get_logger
from core.utils.config_loader import config_loader
from core.utils.gpu_accelerator import gpu_accelerator
from core.retrieval.retrieval_core import retrieval_core

logger = get_logger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="GPU加速测试工具")
    parser.add_argument("--query", type=str, default="人工智能", help="测试查询文本")
    parser.add_argument("--iterations", type=int, default=5, help="每个测试的迭代次数")
    parser.add_argument("--algorithms", type=str, default="tfidf,vector_space", help="要测试的算法，逗号分隔")
    parser.add_argument("--force-cpu", action="store_true", help="强制使用CPU进行测试")
    parser.add_argument("--output", type=str, default="gpu_benchmark_results.png", help="输出图表文件名")
    return parser.parse_args()


def run_benchmark(query: str, iterations: int = 5, algorithms: List[str] = None, force_cpu: bool = False) -> Dict[str, Any]:
    """
    运行基准测试，比较GPU和CPU性能
    
    Args:
        query: 查询文本
        iterations: 每个测试的迭代次数
        algorithms: 要测试的算法列表
        force_cpu: 是否强制使用CPU
        
    Returns:
        Dict: 包含测试结果的字典
    """
    if algorithms is None:
        algorithms = ["tfidf", "vector_space"]
    
    results = {
        "cpu_times": {algo: [] for algo in algorithms},
        "gpu_times": {algo: [] for algo in algorithms}
    }
    
    # 确保检索系统已初始化
    retrieval_core.initialize()
    
    # 获取GPU信息
    gpu_info = gpu_accelerator.get_device_info()
    
    print("\n===== 开始性能测试 =====")
    print(f"查询文本: '{query}'")
    print(f"测试算法: {', '.join(algorithms)}")
    print(f"每个测试的迭代次数: {iterations}")
    print(f"GPU信息: {'可用' if gpu_info['gpu_available'] and not force_cpu else '不可用或被禁用'}")
    if gpu_info['gpu_available'] and not force_cpu:
        print(f"  设备: {gpu_info.get('device', 'unknown')}")
        print(f"  GPU名称: {gpu_info.get('gpu_name', 'unknown')}")
        if 'gpu_memory_total' in gpu_info:
            print(f"  总内存: {gpu_info['gpu_memory_total']:.2f}GB")
    print("=======================\n")
    
    # 测试每个算法
    for algo in algorithms:
        print(f"\n测试算法: {algo}")
        
        # GPU测试
        if gpu_info['gpu_available'] and not force_cpu:
            print(f"  使用GPU运行中...")
            # 设置GPU模式为启用
            original_gpu_setting = gpu_accelerator.use_gpu
            gpu_accelerator.use_gpu = True
            
            # 预热
            retrieval_core.search(query, algorithm=algo)
            
            # 运行测试
            for i in range(iterations):
                start_time = time.time()
                result = retrieval_core.search(query, algorithm=algo)
                elapsed = time.time() - start_time
                results["gpu_times"][algo].append(elapsed)
                print(f"    迭代 {i+1}/{iterations}: {elapsed:.4f}秒, {len(result.results)} 个结果")
            
            avg_gpu = np.mean(results["gpu_times"][algo])
            print(f"  GPU平均耗时: {avg_gpu:.4f}秒")
        else:
            print(f"  GPU不可用或被禁用，跳过GPU测试")
        
        # CPU测试
        print(f"  使用CPU运行中...")
        # 设置GPU模式为禁用
        original_gpu_setting = gpu_accelerator.use_gpu
        gpu_accelerator.use_gpu = False
        
        # 运行测试
        for i in range(iterations):
            start_time = time.time()
            result = retrieval_core.search(query, algorithm=algo)
            elapsed = time.time() - start_time
            results["cpu_times"][algo].append(elapsed)
            print(f"    迭代 {i+1}/{iterations}: {elapsed:.4f}秒, {len(result.results)} 个结果")
        
        avg_cpu = np.mean(results["cpu_times"][algo])
        print(f"  CPU平均耗时: {avg_cpu:.4f}秒")
        
        # 恢复原始GPU设置
        gpu_accelerator.use_gpu = original_gpu_setting
        
        # 计算加速比
        if gpu_info['gpu_available'] and not force_cpu:
            speedup = avg_cpu / avg_gpu if avg_gpu > 0 else 0
            print(f"  GPU加速比: {speedup:.2f}x")
    
    return results


def plot_results(results: Dict[str, Any], output_file: str = "gpu_benchmark_results.png") -> Figure:
    """
    绘制基准测试结果
    
    Args:
        results: 基准测试结果
        output_file: 输出文件名
        
    Returns:
        Figure: matplotlib图表对象
    """
    algorithms = list(results["cpu_times"].keys())
    cpu_times = [np.mean(results["cpu_times"][algo]) for algo in algorithms]
    
    # 检查是否有GPU结果
    has_gpu_results = any(len(results["gpu_times"][algo]) > 0 for algo in algorithms)
    
    if has_gpu_results:
        gpu_times = [np.mean(results["gpu_times"][algo]) if results["gpu_times"][algo] else 0 for algo in algorithms]
        speedups = [cpu / gpu if gpu > 0 else 0 for cpu, gpu in zip(cpu_times, gpu_times)]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6)) if has_gpu_results else plt.subplots(1, 1, figsize=(8, 6))
    
    # 绘制时间对比
    bar_width = 0.35
    index = np.arange(len(algorithms))
    
    bars1 = ax1.bar(index, cpu_times, bar_width, label='CPU')
    if has_gpu_results:
        bars2 = ax1.bar(index + bar_width, gpu_times, bar_width, label='GPU')
    
    ax1.set_xlabel('算法')
    ax1.set_ylabel('平均查询时间 (秒)')
    ax1.set_title('CPU vs GPU 查询性能')
    ax1.set_xticks(index + bar_width / 2)
    ax1.set_xticklabels(algorithms)
    ax1.legend()
    
    # 为每个柱状图添加标签
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.4f}s',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(bars1)
    if has_gpu_results:
        add_labels(bars2)
        
        # 绘制加速比
        ax2.bar(algorithms, speedups, color='green')
        ax2.set_xlabel('算法')
        ax2.set_ylabel('加速比')
        ax2.set_title('GPU加速比 (CPU时间 / GPU时间)')
        
        # 为加速比添加标签
        for i, v in enumerate(speedups):
            ax2.annotate(f'{v:.2f}x',
                        xy=(i, v),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"\n结果图表已保存至: {output_file}")
    
    return fig


def main():
    """主函数"""
    args = parse_args()
    
    # 解析算法列表
    algorithms = [algo.strip() for algo in args.algorithms.split(",")]
    
    # 运行基准测试
    results = run_benchmark(
        query=args.query,
        iterations=args.iterations,
        algorithms=algorithms,
        force_cpu=args.force_cpu
    )
    
    # 绘制结果
    plot_results(results, args.output)


if __name__ == "__main__":
    main()
