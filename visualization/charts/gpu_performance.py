"""
GPU加速性能可视化模块
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple

from core.utils.logger import get_logger

logger = get_logger(__name__)


class GPUPerformanceVisualizer:
    """GPU加速性能可视化类"""
    
    def __init__(self, output_dir: str = "visualization/output"):
        """
        初始化可视化工具
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_time_comparison(self, 
                            cpu_times: Dict[str, List[float]], 
                            gpu_times: Dict[str, List[float]],
                            title: str = "CPU vs GPU 性能对比",
                            output_file: str = "performance_comparison.png") -> str:
        """
        绘制CPU和GPU时间对比图
        
        Args:
            cpu_times: CPU时间数据 {算法: [时间列表]}
            gpu_times: GPU时间数据 {算法: [时间列表]}
            title: 图表标题
            output_file: 输出文件名
            
        Returns:
            str: 输出文件路径
        """
        # 准备数据
        algorithms = list(cpu_times.keys())
        cpu_avg = [np.mean(cpu_times[algo]) for algo in algorithms]
        
        has_gpu_data = any(len(gpu_times.get(algo, [])) > 0 for algo in algorithms)
        if has_gpu_data:
            gpu_avg = [np.mean(gpu_times.get(algo, [0])) for algo in algorithms]
            
            # 计算加速比
            speedups = [cpu/gpu if gpu > 0 else 0 for cpu, gpu in zip(cpu_avg, gpu_avg)]
        
        # 创建图表
        fig, axes = plt.subplots(1, 2 if has_gpu_data else 1, figsize=(12, 6) if has_gpu_data else (8, 6))
        
        # 第一个子图：时间对比
        ax1 = axes[0] if has_gpu_data else axes
        
        # 设置条形图
        bar_width = 0.35
        index = np.arange(len(algorithms))
        
        bars1 = ax1.bar(index, cpu_avg, bar_width, label='CPU')
        if has_gpu_data:
            bars2 = ax1.bar(index + bar_width, gpu_avg, bar_width, label='GPU')
        
        # 设置图表属性
        ax1.set_xlabel('检索算法')
        ax1.set_ylabel('平均查询时间 (秒)')
        ax1.set_title(title)
        ax1.set_xticks(index + bar_width / 2 if has_gpu_data else index)
        ax1.set_xticklabels(algorithms)
        ax1.legend()
        
        # 添加数据标签
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.4f}s',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        add_labels(bars1)
        if has_gpu_data:
            add_labels(bars2)
            
            # 第二个子图：加速比
            ax2 = axes[1]
            
            # 绘制加速比柱状图
            ax2.bar(algorithms, speedups, color='green')
            ax2.set_xlabel('检索算法')
            ax2.set_ylabel('加速比 (CPU/GPU)')
            ax2.set_title('GPU 加速比')
            
            # 添加加速比标签
            for i, v in enumerate(speedups):
                ax2.annotate(f'{v:.2f}x',
                            xy=(i, v),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        # 调整布局并保存
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, output_file)
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"性能对比图表已保存到: {output_path}")
        return output_path
    
    def plot_batch_size_impact(self, 
                              batch_sizes: List[int], 
                              times: List[float],
                              algorithm: str = "TFIDF",
                              output_file: str = "batch_size_impact.png") -> str:
        """
        绘制批处理大小对性能的影响
        
        Args:
            batch_sizes: 批处理大小列表
            times: 对应的处理时间列表
            algorithm: 算法名称
            output_file: 输出文件名
            
        Returns:
            str: 输出文件路径
        """
        plt.figure(figsize=(10, 6))
        
        # 绘制折线图
        plt.plot(batch_sizes, times, marker='o', linestyle='-', linewidth=2)
        
        # 设置图表属性
        plt.xlabel('批处理大小')
        plt.ylabel('处理时间 (秒)')
        plt.title(f'批处理大小对{algorithm}处理性能的影响')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 添加数据标签
        for i, (batch, time) in enumerate(zip(batch_sizes, times)):
            plt.annotate(f'{time:.4f}s',
                        xy=(batch, time),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha='center')
        
        # 保存图表
        output_path = os.path.join(self.output_dir, output_file)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"批处理性能图表已保存到: {output_path}")
        return output_path
    
    def plot_resource_usage(self, 
                           time_points: List[float], 
                           cpu_usage: List[float], 
                           gpu_usage: List[float],
                           memory_usage: List[float],
                           output_file: str = "resource_usage.png") -> str:
        """
        绘制资源使用率图表
        
        Args:
            time_points: 时间点列表
            cpu_usage: CPU使用率列表
            gpu_usage: GPU使用率列表
            memory_usage: 内存使用率列表
            output_file: 输出文件名
            
        Returns:
            str: 输出文件路径
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # 第一个子图：CPU和GPU使用率
        ax1.plot(time_points, cpu_usage, marker='o', linestyle='-', label='CPU使用率')
        ax1.plot(time_points, gpu_usage, marker='s', linestyle='-', label='GPU使用率')
        
        ax1.set_ylabel('使用率 (%)')
        ax1.set_title('计算资源使用情况')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # 第二个子图：内存使用率
        ax2.plot(time_points, memory_usage, marker='^', linestyle='-', color='green')
        
        ax2.set_xlabel('时间 (秒)')
        ax2.set_ylabel('内存使用 (MB)')
        ax2.set_title('内存使用情况')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # 保存图表
        output_path = os.path.join(self.output_dir, output_file)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"资源使用图表已保存到: {output_path}")
        return output_path
    
    def create_performance_report(self, 
                                 benchmark_results: Dict[str, Any],
                                 output_file: str = "performance_report.html") -> str:
        """
        创建性能报告
        
        Args:
            benchmark_results: 基准测试结果
            output_file: 输出文件名
            
        Returns:
            str: 输出文件路径
        """
        # 获取基准测试数据
        algorithms = list(benchmark_results.get("cpu_times", {}).keys())
        cpu_times = benchmark_results.get("cpu_times", {})
        gpu_times = benchmark_results.get("gpu_times", {})
        
        # 计算平均时间和加速比
        data = []
        for algo in algorithms:
            cpu_avg = np.mean(cpu_times.get(algo, [0]))
            gpu_avg = np.mean(gpu_times.get(algo, [0])) if gpu_times.get(algo) else 0
            speedup = cpu_avg / gpu_avg if gpu_avg > 0 else 0
            
            data.append({
                "算法": algo,
                "CPU平均时间(秒)": f"{cpu_avg:.4f}",
                "GPU平均时间(秒)": f"{gpu_avg:.4f}" if gpu_avg > 0 else "N/A",
                "加速比": f"{speedup:.2f}x" if speedup > 0 else "N/A"
            })
        
        # 创建HTML报告
        df = pd.DataFrame(data)
        
        # 生成HTML表格
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>GPU加速性能报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333366; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
                th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #e6f7ff; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>GPU加速性能报告</h1>
            
            <div class="summary">
                <h2>性能总结</h2>
                <p>测试日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>测试算法: {', '.join(algorithms)}</p>
                <p>GPU可用: {'是' if any(gpu_times.values()) else '否'}</p>
                {'<p>平均加速比: ' + f"{np.mean([cpu_avg/gpu_avg for cpu_avg, gpu_avg in zip([np.mean(cpu_times.get(algo, [0])) for algo in algorithms], [np.mean(gpu_times.get(algo, [0])) for algo in algorithms]) if gpu_avg > 0]):.2f}x</p>" if any(gpu_times.values()) else ''}
            </div>
            
            <h2>详细性能数据</h2>
            {df.to_html(index=False)}
            
            <div style="margin-top: 20px;">
                <h2>结论</h2>
                <p>{'GPU加速显著提升了检索性能，平均加速比为' + f"{np.mean([cpu_avg/gpu_avg for cpu_avg, gpu_avg in zip([np.mean(cpu_times.get(algo, [0])) for algo in algorithms], [np.mean(gpu_times.get(algo, [0])) for algo in algorithms]) if gpu_avg > 0]):.2f}x" if any(gpu_times.values()) else 'GPU未使用或不可用，所有计算均在CPU上完成。'}</p>
            </div>
        </body>
        </html>
        """
        
        # 保存HTML报告
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        logger.info(f"性能报告已保存到: {output_path}")
        return output_path


# 提供全局访问点
gpu_performance_visualizer = GPUPerformanceVisualizer()
