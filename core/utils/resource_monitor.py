"""
系统资源监控模块，用于监控CPU、GPU和内存使用情况
"""
import os
import time
import threading
import psutil
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable

# 尝试导入GPU监控库
try:
    import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False

from core.utils.logger import get_logger

logger = get_logger(__name__)


class ResourceMonitor:
    """系统资源监控类，用于跟踪CPU、GPU和内存使用情况"""
    
    def __init__(self):
        """初始化资源监控器"""
        self.gpu_available = False
        self.monitor_thread = None
        self.stop_flag = threading.Event()
        
        # 监控数据
        self.timestamps = []
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory_usage = []
        
        # 初始化GPU监控
        self._init_gpu_monitor()
    
    def _init_gpu_monitor(self):
        """初始化GPU监控"""
        if not HAS_NVML:
            logger.warning("未找到NVML库，无法监控GPU资源")
            return
            
        try:
            # 初始化NVML
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            
            if self.device_count > 0:
                self.gpu_available = True
                logger.info(f"GPU监控初始化成功，检测到{self.device_count}个GPU设备")
            else:
                logger.warning("未检测到GPU设备")
        except Exception as e:
            logger.error(f"GPU监控初始化失败: {str(e)}")
    
    def start_monitoring(self, interval: float = 1.0):
        """
        开始资源监控
        
        Args:
            interval: 监控间隔时间(秒)
        """
        if self.monitor_thread is not None and self.monitor_thread.is_alive():
            logger.warning("监控已在运行中")
            return
            
        # 重置监控数据
        self.timestamps = []
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_usage = []
        self.gpu_memory_usage = []
        
        # 重置停止标志
        self.stop_flag.clear()
        
        # 启动监控线程
        self.monitor_thread = threading.Thread(
            target=self._monitor_resources,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"资源监控已启动，间隔: {interval}秒")
    
    def stop_monitoring(self):
        """停止资源监控"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            logger.warning("没有运行中的监控")
            return
            
        # 设置停止标志
        self.stop_flag.set()
        
        # 等待线程结束
        self.monitor_thread.join(timeout=5.0)
        logger.info("资源监控已停止")
    
    def _monitor_resources(self, interval: float):
        """
        资源监控主循环
        
        Args:
            interval: 监控间隔时间(秒)
        """
        start_time = time.time()
        
        while not self.stop_flag.is_set():
            try:
                current_time = time.time() - start_time
                
                # 获取CPU使用率
                cpu_percent = psutil.cpu_percent()
                
                # 获取内存使用率
                memory = psutil.virtual_memory()
                memory_mb = memory.used / (1024 * 1024)  # 转换为MB
                
                # 获取GPU使用率
                gpu_percent = 0
                gpu_memory_mb = 0
                
                if self.gpu_available and HAS_NVML:
                    try:
                        # 只监控第一个GPU
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        
                        # GPU使用率
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_percent = utilization.gpu
                        
                        # GPU内存使用
                        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        gpu_memory_mb = memory_info.used / (1024 * 1024)  # 转换为MB
                    except Exception as e:
                        logger.error(f"获取GPU信息失败: {str(e)}")
                
                # 添加到监控数据
                self.timestamps.append(current_time)
                self.cpu_usage.append(cpu_percent)
                self.memory_usage.append(memory_mb)
                self.gpu_usage.append(gpu_percent)
                self.gpu_memory_usage.append(gpu_memory_mb)
                
                # 等待下一个间隔
                time.sleep(interval)
            except Exception as e:
                logger.error(f"资源监控异常: {str(e)}")
                time.sleep(interval)
    
    def get_monitoring_data(self) -> Dict[str, List]:
        """
        获取监控数据
        
        Returns:
            Dict: 包含各项监控数据的字典
        """
        return {
            "timestamps": self.timestamps,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "gpu_usage": self.gpu_usage,
            "gpu_memory_usage": self.gpu_memory_usage
        }
    
    def get_average_usage(self) -> Dict[str, float]:
        """
        获取平均资源使用情况
        
        Returns:
            Dict: 包含各项资源平均使用率的字典
        """
        return {
            "cpu_avg": np.mean(self.cpu_usage) if self.cpu_usage else 0,
            "memory_avg_mb": np.mean(self.memory_usage) if self.memory_usage else 0,
            "gpu_avg": np.mean(self.gpu_usage) if self.gpu_usage else 0,
            "gpu_memory_avg_mb": np.mean(self.gpu_memory_usage) if self.gpu_memory_usage else 0
        }
    
    def monitor_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        监控函数执行期间的资源使用情况
        
        Args:
            func: 要监控的函数
            *args: 函数参数
            **kwargs: 函数关键字参数
            
        Returns:
            Tuple: (函数返回值, 资源使用统计)
        """
        # 开始监控
        self.start_monitoring(interval=0.1)
        
        try:
            # 执行函数
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
        finally:
            # 停止监控
            self.stop_monitoring()
        
        # 计算统计信息
        avg_usage = self.get_average_usage()
        avg_usage["execution_time"] = execution_time
        
        return result, avg_usage
    
    def __del__(self):
        """清理资源"""
        if HAS_NVML and self.gpu_available:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


# 提供全局访问点
resource_monitor = ResourceMonitor()
