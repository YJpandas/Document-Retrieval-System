"""
检索核心模块，整合索引和检索功能
"""
import os
import time
import threading
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor

from core.utils.logger import get_logger
from core.utils.config_loader import config_loader
from core.utils.text_processor import text_processor
from core.utils.gpu_accelerator import gpu_accelerator
from core.datasource.datasource_manager import Document, datasource_manager
from core.retrieval.index.indexer import indexer
from core.retrieval.index.search_engine import search_engine, SearchResult, SearchResults
from core.rules.rule_engine import rule_engine

logger = get_logger(__name__)


class RetrievalCore:
    """检索核心类，整合索引和检索功能，提供统一的检索接口"""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """单例模式实现"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RetrievalCore, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """初始化检索核心"""
        with self._lock:
            if self._initialized:
                return
            
            # 加载配置
            self.config = config_loader.get_config("app_config") or {}
            
            # 组件引用
            self.indexer = indexer
            self.search_engine = search_engine
            self.datasource_manager = datasource_manager
            self.rule_engine = rule_engine
            
            # 检查并初始化GPU加速器
            # 这里不需要显式初始化，因为gpu_accelerator是单例模式
            # 在导入时已经自动初始化
            hardware_config = self.config.get('hardware', {})
            if hardware_config.get('use_gpu', False):
                gpu_info = gpu_accelerator.get_device_info()
                if gpu_info.get('gpu_available', False):
                    logger.info(f"检索核心将使用GPU加速: {gpu_info.get('device', 'unknown')}")
                else:
                    logger.warning("配置文件启用了GPU加速，但未检测到可用的GPU")
            
            # 标志位
            self._is_initialized = False
            self._is_indexing = False
            
            # 完成初始化
            self._initialized = True
            logger.info("检索核心初始化完成")
    
    def initialize(self) -> bool:
        """
        初始化检索系统，加载索引和文档
        
        Returns:
            bool: 初始化成功返回True，否则返回False
        """
        if self._is_initialized:
            logger.info("检索系统已初始化，跳过")
            return True
        
        with self._lock:
            try:
                # 尝试加载索引
                index_loaded = self.indexer.load_index()
                docs_loaded = self.indexer.load_documents()
                
                # 如果索引加载失败，尝试重建索引
                if not index_loaded or not docs_loaded:
                    logger.info("索引加载失败，尝试重建索引")
                    # 加载原始文档
                    self.datasource_manager.scan_directory()
                    # 构建索引
                    self.indexer.build_index(rebuild=True)
                    # 保存索引和文档
                    self.indexer.save_index()
                    self.indexer.save_documents()
                
                self._is_initialized = True
                logger.info("检索系统初始化成功")
                return True
            except Exception as e:
                logger.error(f"检索系统初始化失败: {str(e)}")
                return False
    
    def search(self, query: str, algorithm: str = None, max_results: int = None,
              min_score: float = None, highlight: bool = True, apply_rules: bool = True,
              sort_by: str = "relevance", **kwargs) -> SearchResults:
        """
        执行文档检索
        
        Args:
            query: 搜索查询
            algorithm: 检索算法，如果为None则使用默认算法
            max_results: 最大结果数量，如果为None则使用默认值
            min_score: 最小相关性得分，如果为None则使用默认值
            highlight: 是否高亮结果
            apply_rules: 是否应用自定义规则
            sort_by: 排序方式，支持"relevance"和"time"
            **kwargs: 其他参数
            
        Returns:
            SearchResults: 搜索结果集合
        """
        # 确保系统已初始化
        if not self._is_initialized:
            self.initialize()
        
        # 记录GPU使用情况
        gpu_info = gpu_accelerator.get_device_info()
        use_gpu = gpu_info.get('use_gpu', False) and gpu_info.get('gpu_available', False)
        if use_gpu:
            logger.info(f"检索将使用GPU加速: {gpu_info.get('device', 'unknown')}")
            
        # 前置处理：应用规则引擎对查询进行处理
        if apply_rules:
            original_query = query
            query, rule_metadata = self.rule_engine.process_query(query)
            logger.info(f"规则处理：原查询 '{original_query}' -> 处理后查询 '{query}'")
        else:
            rule_metadata = {}
        
        # 执行搜索
        start_time = time.time()
        search_results = self.search_engine.search(
            query=query,
            algorithm=algorithm,
            max_results=max_results,
            min_score=min_score,
            highlight=highlight,
            metadata={'rule_metadata': rule_metadata, 'use_gpu': use_gpu}
        )
        
        # 后置处理：应用排序和自定义规则
        if apply_rules and search_results.results:
            # 应用规则引擎对结果进行处理
            processed_results = self.rule_engine.process_results(search_results)
            search_results = processed_results
        
        # 应用自定义排序
        if sort_by == "time" and search_results.results:
            # 按文档最后修改时间排序
            search_results.results.sort(
                key=lambda r: r.document.last_modified if r.document else 0,
                reverse=True
            )
        
        # 统计和记录
        elapsed_time = time.time() - start_time
        search_results.elapsed_time = elapsed_time
        
        logger.info(f"检索完成: 查询='{query}', "
                   f"找到={len(search_results.results)}/{search_results.total_docs}, "
                   f"耗时={elapsed_time:.4f}秒, GPU={use_gpu}")
        
        return search_results
    
    def refresh_index(self, rebuild: bool = False) -> bool:
        """
        刷新索引
        
        Args:
            rebuild: 是否重建索引
            
        Returns:
            bool: 成功返回True，否则返回False
        """
        if self._is_indexing:
            logger.warning("索引正在进行中，跳过")
            return False
        
        with self._lock:
            self._is_indexing = True
            try:
                # 扫描文档目录
                start_time = time.time()
                
                self.datasource_manager.scan_directory()
                documents = self.datasource_manager.get_all_documents()
                
                # 更新索引
                self.indexer.build_index(documents, rebuild=rebuild)
                
                # 保存索引和文档
                self.indexer.save_index()
                self.indexer.save_documents()
                
                elapsed_time = time.time() - start_time
                logger.info(f"索引刷新完成: 文档数={len(documents)}, 耗时={elapsed_time:.2f}秒")
                
                return True
            except Exception as e:
                logger.error(f"索引刷新失败: {str(e)}")
                return False
            finally:
                self._is_indexing = False
    
    def add_document(self, file_path: str) -> bool:
        """
        添加单个文档到索引
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            bool: 成功返回True，否则返回False
        """
        try:
            # 解析文档
            document = self.datasource_manager.parse_file(file_path)
            if not document:
                logger.error(f"文档解析失败: {file_path}")
                return False
            
            # 添加到索引
            result = self.indexer.add_document(document)
            
            # 保存索引和文档
            if result:
                self.indexer.save_index()
                self.indexer.save_documents()
            
            return result
        except Exception as e:
            logger.error(f"添加文档失败: {file_path}, 错误: {str(e)}")
            return False
    
    def remove_document(self, doc_id: str) -> bool:
        """
        从索引中移除文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            bool: 成功返回True，否则返回False
        """
        try:
            # 从索引中移除
            result = self.indexer.remove_document(doc_id)
            
            # 保存索引和文档
            if result:
                self.indexer.save_index()
                self.indexer.save_documents()
            
            return result
        except Exception as e:
            logger.error(f"移除文档失败: {doc_id}, 错误: {str(e)}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取检索系统统计信息
        
        Returns:
            Dict: 统计信息
        """
        stats = {
            'total_documents': len(self.indexer.get_all_documents()),
            'total_terms': len(self.indexer.inverted_index.get_terms()),
            'avg_doc_length': self.indexer.inverted_index.get_avg_doc_length(),
            'is_initialized': self._is_initialized,
            'is_indexing': self._is_indexing,
            'last_update': self.indexer.inverted_index.last_update
        }
        
        # 文档类型统计
        doc_types = {}
        for doc in self.indexer.get_all_documents():
            doc_type = doc.doc_type
            if doc_type not in doc_types:
                doc_types[doc_type] = 0
            doc_types[doc_type] += 1
        
        stats['document_types'] = doc_types
        
        # 添加GPU信息
        gpu_info = gpu_accelerator.get_device_info()
        stats['gpu'] = {
            'available': gpu_info.get('gpu_available', False),
            'enabled': gpu_info.get('use_gpu', False),
            'device': gpu_info.get('device', 'cpu')
        }
        
        # 如果GPU可用，添加详细信息
        if gpu_info.get('gpu_available', False):
            if 'gpu_name' in gpu_info:
                stats['gpu']['name'] = gpu_info['gpu_name']
            if 'gpu_memory_total' in gpu_info:
                stats['gpu']['memory_total'] = gpu_info['gpu_memory_total']
            if 'gpu_memory_allocated' in gpu_info:
                stats['gpu']['memory_used'] = gpu_info['gpu_memory_allocated']
            if 'mixed_precision' in gpu_info:
                stats['gpu']['mixed_precision'] = gpu_info['mixed_precision']
        
        return stats
    
    def batch_index(self, directory: str = None, max_workers: int = 4) -> Dict[str, Any]:
        """
        批量索引文档
        
        Args:
            directory: 要索引的目录，默认为配置中的raw_documents目录
            max_workers: 最大工作线程数
            
        Returns:
            Dict: 包含处理结果的字典
        """
        if self._is_indexing:
            logger.warning("索引正在进行中，跳过")
            return {'success': False, 'error': '索引正在进行中'}
        
        with self._lock:
            self._is_indexing = True
            try:
                start_time = time.time()
                
                # 获取硬件配置信息
                configured_max_threads = self.config.get('hardware', {}).get('max_threads', 4)
                if max_workers is None or max_workers <= 0:
                    max_workers = configured_max_threads
                
                # 扫描文档目录
                self.datasource_manager.scan_directory(directory)
                documents = self.datasource_manager.get_all_documents()
                
                # 批量索引
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = []
                    batch_size = max(1, len(documents) // max_workers)
                    
                    for i in range(0, len(documents), batch_size):
                        batch = documents[i:i + batch_size]
                        futures.append(executor.submit(self._index_batch, batch))
                    
                    # 等待所有批次完成
                    for future in futures:
                        future.result()
                
                # 检查是否可以使用GPU计算权重
                use_gpu = gpu_accelerator.is_gpu_available()
                if use_gpu:
                    logger.info(f"使用GPU加速计算文档权重")
                
                # 计算文档权重
                self.indexer.inverted_index.compute_doc_weights('tfidf')
                
                # 保存索引和文档
                self.indexer.save_index()
                self.indexer.save_documents()
                
                elapsed_time = time.time() - start_time
                logger.info(f"批量索引完成: 文档数={len(documents)}, 耗时={elapsed_time:.2f}秒, GPU={use_gpu}")
                
                return {
                    'success': True,
                    'document_count': len(documents),
                    'elapsed_time': elapsed_time,
                    'use_gpu': use_gpu
                }
            except Exception as e:
                logger.error(f"批量索引失败: {str(e)}")
                return {'success': False, 'error': str(e)}
            finally:
                self._is_indexing = False
    
    def _index_batch(self, documents: List[Document]):
        """
        索引文档批次
        
        Args:
            documents: 文档列表
        """
        for doc in documents:
            try:
                # 处理文档文本
                tokens = text_processor.process_text(doc.content)
                
                # 添加到索引
                self.indexer.inverted_index.add_document(doc.doc_id, tokens, positions=True)
                
                # 更新文档映射
                self.indexer.documents[doc.doc_id] = doc
            except Exception as e:
                logger.error(f"索引文档失败: {doc.doc_id}, 错误: {str(e)}")


# 提供全局访问点
retrieval_core = RetrievalCore()