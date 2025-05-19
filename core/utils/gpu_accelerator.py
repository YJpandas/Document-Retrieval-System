"""
GPU加速工具模块，提供GPU处理功能
"""
import os
import numpy as np
from typing import Dict, List, Any, Optional, Union
import logging

# 尝试导入GPU加速相关库
try:
    import torch
    import numba
    import cupy as cp
    HAS_GPU_SUPPORT = True
except ImportError:
    HAS_GPU_SUPPORT = False

from core.utils.logger import get_logger
from core.utils.config_loader import config_loader

logger = get_logger(__name__)


class GPUAccelerator:
    """GPU加速器，提供GPU计算功能"""
    
    _instance = None
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(GPUAccelerator, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """初始化GPU加速器"""
        if self._initialized:
            return
            
        # 加载配置
        self.config = config_loader.get_config("app_config") or {}
        self.hardware_config = self.config.get("hardware", {})
        
        # 初始化GPU设置
        self.use_gpu = self.hardware_config.get("use_gpu", False)
        self.gpu_id = self.hardware_config.get("gpu_id", 0)
        self.mixed_precision = self.hardware_config.get("mixed_precision", False)
        self.batch_size = self.hardware_config.get("batch_size", 32)
        
        # 初始化GPU
        self.gpu_available = False
        self.device = None
        
        if self.use_gpu and HAS_GPU_SUPPORT:
            self._init_gpu()
        else:
            if self.use_gpu and not HAS_GPU_SUPPORT:
                logger.warning("配置启用了GPU，但未找到GPU支持库。将使用CPU模式。")
            self.use_gpu = False
            self.device = "cpu"
            
        self._initialized = True
        logger.info(f"GPU加速器初始化完成: use_gpu={self.use_gpu}, device={self.device}")
    
    def _init_gpu(self):
        """初始化GPU设备"""
        try:
            if torch.cuda.is_available():
                self.gpu_available = True
                torch.cuda.set_device(self.gpu_id)
                self.device = f"cuda:{self.gpu_id}"
                
                # 启用自动混合精度
                if self.mixed_precision:
                    if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                        self.autocast = torch.cuda.amp.autocast
                    else:
                        logger.warning("当前PyTorch版本不支持自动混合精度。")
                        self.autocast = None
                else:
                    self.autocast = None
                
                gpu_name = torch.cuda.get_device_name(self.gpu_id)
                total_memory = torch.cuda.get_device_properties(self.gpu_id).total_memory / (1024 ** 3)
                logger.info(f"成功初始化GPU: {gpu_name}, 总内存: {total_memory:.2f}GB")
            else:
                logger.warning("未检测到CUDA设备，将使用CPU模式。")
                self.use_gpu = False
                self.device = "cpu"
        except Exception as e:
            logger.error(f"GPU初始化失败: {str(e)}")
            self.use_gpu = False
            self.device = "cpu"
    
    def to_device(self, data):
        """
        将数据转移到合适的设备上（GPU或CPU）
        
        Args:
            data: 要转移的数据，可以是Tensor、数组或字典
            
        Returns:
            转移后的数据
        """
        if not self.use_gpu or not self.gpu_available:
            return data
            
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, np.ndarray):
            try:
                # 使用cupy转换
                return cp.asarray(data)
            except:
                # 回退到torch
                return torch.from_numpy(data).to(self.device)
        elif isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.to_device(v) for v in data]
        else:
            return data
    
    def to_cpu(self, data):
        """
        将数据从GPU转移到CPU
        
        Args:
            data: 要转移的数据
            
        Returns:
            转移后的数据
        """
        if not self.use_gpu or not self.gpu_available:
            return data
            
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif hasattr(data, 'get'):  # CuPy数组
            try:
                return cp.asnumpy(data)
            except:
                return data
        elif isinstance(data, dict):
            return {k: self.to_cpu(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.to_cpu(v) for v in data]
        else:
            return data
    
    def batched_process(self, items, process_func, batch_size=None):
        """
        批量处理数据
        
        Args:
            items: 要处理的数据列表
            process_func: 处理函数
            batch_size: 批大小
            
        Returns:
            处理结果
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = process_func(batch)
            results.extend(batch_results)
        return results
    
    def cosine_similarity_gpu(self, query_vector, doc_vectors):
        """
        使用GPU计算余弦相似度
        
        Args:
            query_vector: 查询向量 (1, dim)
            doc_vectors: 文档向量 (n_docs, dim)
            
        Returns:
            相似度分数 (n_docs,)
        """
        if not self.use_gpu or not self.gpu_available:
            # 使用NumPy计算
            return np.dot(doc_vectors, query_vector) / (
                np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vector)
            )
        
        try:
            # 使用PyTorch计算
            query_tensor = torch.tensor(query_vector, device=self.device).float()
            docs_tensor = torch.tensor(doc_vectors, device=self.device).float()
            
            # 归一化
            query_norm = torch.norm(query_tensor)
            docs_norm = torch.norm(docs_tensor, dim=1)
            
            # 避免除零
            query_norm = torch.clamp(query_norm, min=1e-8)
            docs_norm = torch.clamp(docs_norm, min=1e-8)
            
            # 计算余弦相似度
            query_normalized = query_tensor / query_norm
            docs_normalized = docs_tensor / docs_norm.unsqueeze(1)
            
            similarities = torch.matmul(docs_normalized, query_normalized)
            return self.to_cpu(similarities)
        except Exception as e:
            logger.error(f"GPU余弦相似度计算失败: {str(e)}，回退到CPU")
            # 回退到NumPy计算
            return np.dot(doc_vectors, query_vector) / (
                np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vector)
            )
    
    def vector_matrix_multiply(self, vector, matrix):
        """
        向量与矩阵乘法，用于检索计算
        
        Args:
            vector: 向量
            matrix: 矩阵
            
        Returns:
            乘法结果
        """
        if not self.use_gpu or not self.gpu_available:
            return np.dot(matrix, vector)
            
        try:
            # 转为torch张量并移至GPU
            vector_tensor = torch.tensor(vector, device=self.device).float()
            matrix_tensor = torch.tensor(matrix, device=self.device).float()
            
            # 矩阵乘法
            result = torch.matmul(matrix_tensor, vector_tensor)
            
            # 移回CPU并转为numpy
            return self.to_cpu(result)
        except Exception as e:
            logger.error(f"GPU矩阵乘法计算失败: {str(e)}，回退到CPU")
            return np.dot(matrix, vector)
    
    def is_gpu_available(self):
        """
        检查GPU是否可用
        
        Returns:
            bool: GPU是否可用
        """
        return self.use_gpu and self.gpu_available
    
    def get_device_info(self):
        """
        获取当前设备信息
        
        Returns:
            Dict: 设备信息
        """
        info = {
            'use_gpu': self.use_gpu,
            'gpu_available': self.gpu_available,
            'device': self.device,
            'mixed_precision': self.mixed_precision
        }
        
        if self.use_gpu and self.gpu_available and hasattr(torch, 'cuda'):
            try:
                info['gpu_name'] = torch.cuda.get_device_name(self.gpu_id)
                info['gpu_memory_total'] = torch.cuda.get_device_properties(self.gpu_id).total_memory / (1024 ** 3)
                info['gpu_memory_allocated'] = torch.cuda.memory_allocated(self.gpu_id) / (1024 ** 3)
                info['gpu_memory_reserved'] = torch.cuda.memory_reserved(self.gpu_id) / (1024 ** 3)
            except:
                pass
                
        return info


# 提供全局访问点
gpu_accelerator = GPUAccelerator()
