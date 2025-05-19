#!/usr/bin/env python
"""
文档检索系统主入口
"""
import os
import argparse
import time
from typing import Dict, Any

from core.utils.logger import get_logger, setup_logging
from core.utils.config_loader import config_loader
from core.utils.gpu_accelerator import gpu_accelerator
from core.utils.resource_monitor import resource_monitor
from core.retrieval.retrieval_core import retrieval_core
from visualization.charts.gpu_performance import gpu_performance_visualizer

# 设置日志
setup_logging()
logger = get_logger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="文档检索系统")
    parser.add_argument("--rebuild-index", action="store_true", help="重建索引")
    parser.add_argument("--query", type=str, help="执行查询")
    parser.add_argument("--algorithm", type=str, choices=["tfidf", "vector_space", "boolean"], help="检索算法")
    parser.add_argument("--max-results", type=int, default=10, help="最大结果数量")
    parser.add_argument("--benchmark", action="store_true", help="运行性能基准测试")
    parser.add_argument("--gpu-disabled", action="store_true", help="禁用GPU加速")
    return parser.parse_args()


def print_gpu_info():
    """打印GPU信息"""
    gpu_info = gpu_accelerator.get_device_info()
    
    print("\n===== GPU加速信息 =====")
    if gpu_info.get("gpu_available", False):
        print(f"GPU状态: {'启用' if gpu_info.get('use_gpu', False) else '禁用'}")
        print(f"设备: {gpu_info.get('device', 'unknown')}")
        if "gpu_name" in gpu_info:
            print(f"GPU型号: {gpu_info['gpu_name']}")
        if "gpu_memory_total" in gpu_info:
            print(f"GPU总内存: {gpu_info['gpu_memory_total']:.2f} GB")
        if "mixed_precision" in gpu_info:
            print(f"混合精度: {'启用' if gpu_info['mixed_precision'] else '禁用'}")
    else:
        print("GPU状态: 不可用")
    print("=======================\n")


def run_benchmark(query: str = "人工智能", algorithms: list = None) -> Dict[str, Any]:
    """
    运行性能基准测试
    
    Args:
        query: 测试查询
        algorithms: 要测试的算法列表
        
    Returns:
        Dict: 测试结果
    """
    if algorithms is None:
        algorithms = ["tfidf", "vector_space"]
    
    iterations = 5
    
    # 准备结果数据结构
    results = {
        "cpu_times": {algo: [] for algo in algorithms},
        "gpu_times": {algo: [] for algo in algorithms}
    }
    
    print("\n===== 开始性能基准测试 =====")
    print(f"查询: '{query}'")
    print(f"算法: {algorithms}")
    print(f"每个测试的迭代次数: {iterations}")
    
    # 启动资源监控
    resource_monitor.start_monitoring()
    
    try:
        # 为每个算法运行测试
        for algo in algorithms:
            print(f"\n测试算法: {algo}")
            
            # GPU测试
            if gpu_accelerator.is_gpu_available():
                print("  使用GPU运行中...")
                
                # 设置GPU模式
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
                    print(f"    迭代 {i+1}/{iterations}: {elapsed:.4f}秒, {len(result.results)} 结果")
                
                # 计算平均
                gpu_avg = sum(results["gpu_times"][algo]) / len(results["gpu_times"][algo])
                print(f"  GPU平均耗时: {gpu_avg:.4f}秒")
                
                # 恢复原始设置
                gpu_accelerator.use_gpu = original_gpu_setting
            else:
                print("  GPU不可用，跳过GPU测试")
            
            # CPU测试
            print("  使用CPU运行中...")
            
            # 禁用GPU
            original_gpu_setting = gpu_accelerator.use_gpu
            gpu_accelerator.use_gpu = False
            
            # 运行测试
            for i in range(iterations):
                start_time = time.time()
                result = retrieval_core.search(query, algorithm=algo)
                elapsed = time.time() - start_time
                results["cpu_times"][algo].append(elapsed)
                print(f"    迭代 {i+1}/{iterations}: {elapsed:.4f}秒, {len(result.results)} 结果")
            
            # 计算平均
            cpu_avg = sum(results["cpu_times"][algo]) / len(results["cpu_times"][algo])
            print(f"  CPU平均耗时: {cpu_avg:.4f}秒")
            
            # 恢复原始设置
            gpu_accelerator.use_gpu = original_gpu_setting
            
            # 计算加速比
            if gpu_accelerator.is_gpu_available():
                speedup = cpu_avg / gpu_avg
                print(f"  加速比: {speedup:.2f}x")
    finally:
        # 停止资源监控
        resource_monitor.stop_monitoring()
    
    # 获取资源使用情况
    resource_data = resource_monitor.get_average_usage()
    print(f"\n资源使用情况:")
    print(f"  CPU平均使用率: {resource_data['cpu_avg']:.2f}%")
    print(f"  内存平均使用: {resource_data['memory_avg_mb'] / 1024:.2f} GB")
    if gpu_accelerator.is_gpu_available():
        print(f"  GPU平均使用率: {resource_data['gpu_avg']:.2f}%")
        print(f"  GPU内存平均使用: {resource_data['gpu_memory_avg_mb'] / 1024:.2f} GB")
    
    print("\n===== 基准测试完成 =====\n")
    
    # 生成可视化和报告
    gpu_performance_visualizer.plot_time_comparison(
        results["cpu_times"], 
        results["gpu_times"],
        title="检索算法CPU vs GPU性能对比"
    )
    
    report_path = gpu_performance_visualizer.create_performance_report(results)
    print(f"性能报告已生成: {report_path}\n")
    
    return results


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置
    config = config_loader.get_config("app_config")
    
    # 初始化GPU（可以通过参数禁用）
    if args.gpu_disabled:
        logger.info("通过命令行参数禁用GPU加速")
        gpu_accelerator.use_gpu = False
    
    # 打印GPU信息
    print_gpu_info()
    
    # 初始化检索核心
    logger.info("初始化检索核心...")
    retrieval_core.initialize()
    
    # 处理命令行操作
    if args.rebuild_index:
        logger.info("正在重建索引...")
        result = retrieval_core.refresh_index(rebuild=True)
        if result:
            logger.info("索引重建成功")
        else:
            logger.error("索引重建失败")
    
    if args.benchmark:
        logger.info("运行性能基准测试...")
        algorithms = ["tfidf", "vector_space"]
        if args.algorithm:
            algorithms = [args.algorithm]
        
        query = args.query if args.query else "人工智能 大数据 机器学习"
        run_benchmark(query=query, algorithms=algorithms)
    
    if args.query:
        logger.info(f"执行查询: {args.query}")
        start_time = time.time()
        
        results = retrieval_core.search(
            query=args.query,
            algorithm=args.algorithm,
            max_results=args.max_results
        )
        
        elapsed_time = time.time() - start_time
        
        # 打印查询结果
        print(f"\n===== 查询结果: '{args.query}' =====")
        print(f"找到 {len(results.results)} 个结果，耗时 {elapsed_time:.4f}秒")
        print(f"使用算法: {results.algorithm}")
        print(f"GPU加速: {'是' if gpu_accelerator.is_gpu_available() and gpu_accelerator.use_gpu else '否'}")
        
        # 打印结果
        for i, result in enumerate(results.results):
            print(f"\n{i+1}. {result.document.title if result.document else 'Unknown'}")
            print(f"   相关度: {result.score:.4f}")
            print(f"   文件: {result.document.file_path if result.document else 'Unknown'}")
            if result.highlight:
                print(f"   摘要: {result.highlight[:200]}...")
        
        print("\n===============================\n")
    
    # 如果没有任何操作参数，打印帮助信息
    if not (args.rebuild_index or args.query or args.benchmark):
        print("请指定操作参数。使用 --help 查看帮助。")


if __name__ == "__main__":
    main()
