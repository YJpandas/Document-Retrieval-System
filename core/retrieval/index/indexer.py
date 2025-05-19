"""
索引器模块，负责创建和管理倒排索引
"""
import os
import json
import pickle
import time
import threading
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from core.utils.logger import get_logger
from core.utils.config_loader import config_loader
from core.utils.text_processor import text_processor
from core.datasource.datasource_manager import Document, datasource_manager

logger = get_logger(__name__)


class InvertedIndex:
    """倒排索引类，存储词项到文档的映射关系"""
    
    def __init__(self):
        """初始化倒排索引"""
        # 词项 -> 文档ID列表的映射
        self.index = defaultdict(list)
        # 词项 -> (文档ID, 词频, 位置列表) 的映射
        self.positional_index = defaultdict(list)
        # 文档ID -> 文档长度(词项数量)的映射
        self.doc_lengths = {}
        # 词项的文档频率(包含该词项的文档数量)
        self.doc_freqs = defaultdict(int)
        # 总文档数
        self.total_docs = 0
        # 平均文档长度
        self.avg_doc_length = 0
        # 文档权重(针对TF-IDF和BM25)
        self.doc_weights = {}
        # 最后更新时间
        self.last_update = time.time()
    
    def add_document(self, doc_id: str, tokens: List[str], positions: bool = True):
        """
        向索引中添加文档
        
        Args:
            doc_id: 文档ID
            tokens: 文档分词结果
            positions: 是否记录位置信息
        """
        if not tokens:
            return
        
        # 更新文档长度
        self.doc_lengths[doc_id] = len(tokens)
        
        # 计算文档中每个词项的频率
        term_freqs = defaultdict(int)
        # 记录词项在文档中的位置
        term_positions = defaultdict(list)
        
        for pos, token in enumerate(tokens):
            term_freqs[token] += 1
            if positions:
                term_positions[token].append(pos)
        
        # 更新倒排索引
        for term, freq in term_freqs.items():
            # 如果文档ID不在该词项的倒排列表中，更新文档频率
            if doc_id not in [doc_tuple[0] for doc_tuple in self.positional_index.get(term, [])]:
                self.doc_freqs[term] += 1
            
            # 更新位置信息
            if positions:
                self.positional_index[term].append((doc_id, freq, term_positions[term]))
            else:
                self.positional_index[term].append((doc_id, freq, []))
            
            # 更新基本索引
            if doc_id not in self.index[term]:
                self.index[term].append(doc_id)
        
        # 更新总文档数
        self.total_docs = len(self.doc_lengths)
        
        # 更新平均文档长度
        if self.total_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs
        
        # 更新时间戳
        self.last_update = time.time()
    
    def remove_document(self, doc_id: str):
        """
        从索引中移除文档
        
        Args:
            doc_id: 要移除的文档ID
        """
        # 从文档长度中移除
        if doc_id in self.doc_lengths:
            del self.doc_lengths[doc_id]
        
        # 从索引中移除
        for term in list(self.index.keys()):
            if doc_id in self.index[term]:
                self.index[term].remove(doc_id)
                # 如果索引为空，删除该词项
                if not self.index[term]:
                    del self.index[term]
                    if term in self.doc_freqs:
                        del self.doc_freqs[term]
                else:
                    # 更新文档频率
                    self.doc_freqs[term] -= 1
        
        # 从位置索引中移除
        for term in list(self.positional_index.keys()):
            self.positional_index[term] = [
                (d_id, freq, positions) for d_id, freq, positions in self.positional_index[term]
                if d_id != doc_id
            ]
            # 如果位置索引为空，删除该词项
            if not self.positional_index[term]:
                del self.positional_index[term]
        
        # 更新总文档数
        self.total_docs = len(self.doc_lengths)
        
        # 更新平均文档长度
        if self.total_docs > 0:
            self.avg_doc_length = sum(self.doc_lengths.values()) / self.total_docs
        else:
            self.avg_doc_length = 0
        
        # 如果文档被移除，也应该从文档权重中移除
        if doc_id in self.doc_weights:
            del self.doc_weights[doc_id]
        
        # 更新时间戳
        self.last_update = time.time()
    
    def get_postings(self, term: str) -> List[str]:
        """
        获取词项的倒排列表
        
        Args:
            term: 词项
            
        Returns:
            List[str]: 包含该词项的文档ID列表
        """
        return self.index.get(term, [])
    
    def get_positional_postings(self, term: str) -> List[Tuple[str, int, List[int]]]:
        """
        获取词项的位置信息
        
        Args:
            term: 词项
            
        Returns:
            List[Tuple]: (文档ID, 词频, 位置列表)的列表
        """
        return self.positional_index.get(term, [])
    
    def get_doc_freq(self, term: str) -> int:
        """
        获取词项的文档频率
        
        Args:
            term: 词项
            
        Returns:
            int: 包含该词项的文档数量
        """
        return self.doc_freqs.get(term, 0)
    
    def get_term_freq(self, term: str, doc_id: str) -> int:
        """
        获取词项在指定文档中的频率
        
        Args:
            term: 词项
            doc_id: 文档ID
            
        Returns:
            int: 词项在文档中的频率
        """
        for d_id, freq, _ in self.positional_index.get(term, []):
            if d_id == doc_id:
                return freq
        return 0
    
    def get_terms(self) -> List[str]:
        """
        获取索引中的所有词项
        
        Returns:
            List[str]: 词项列表
        """
        return list(self.index.keys())
    
    def get_doc_ids(self) -> List[str]:
        """
        获取索引中的所有文档ID
        
        Returns:
            List[str]: 文档ID列表
        """
        return list(self.doc_lengths.keys())
    
    def get_doc_length(self, doc_id: str) -> int:
        """
        获取文档长度
        
        Args:
            doc_id: 文档ID
            
        Returns:
            int: 文档长度(词项数量)
        """
        return self.doc_lengths.get(doc_id, 0)
    
    def get_avg_doc_length(self) -> float:
        """
        获取平均文档长度
        
        Returns:
            float: 平均文档长度
        """
        return self.avg_doc_length
    
    def get_total_docs(self) -> int:
        """
        获取索引中的总文档数
        
        Returns:
            int: 总文档数
        """
        return self.total_docs
    
    def compute_doc_weights(self, scheme: str = 'tfidf'):
        """
        计算文档权重(用于TF-IDF和向量空间模型)
        
        Args:
            scheme: 权重计算方案, 'tfidf' 或 'bm25'
        """
        self.doc_weights = {}
        
        if scheme.lower() == 'tfidf':
            for doc_id in self.doc_lengths.keys():
                weights = {}
                for term in self.get_terms():
                    tf = self.get_term_freq(term, doc_id)
                    if tf > 0:
                        df = self.get_doc_freq(term)
                        # TF-IDF权重计算
                        if df > 0:
                            idf = np.log(self.total_docs / df)
                            weights[term] = tf * idf
                
                # 归一化权重向量
                vec_len = np.sqrt(sum(w * w for w in weights.values()))
                if vec_len > 0:
                    self.doc_weights[doc_id] = {term: w / vec_len for term, w in weights.items()}
                else:
                    self.doc_weights[doc_id] = weights
        
        elif scheme.lower() == 'bm25':
            # BM25参数
            k1 = 1.2
            b = 0.75
            
            for doc_id in self.doc_lengths.keys():
                weights = {}
                doc_len = self.get_doc_length(doc_id)
                
                for term in self.get_terms():
                    tf = self.get_term_freq(term, doc_id)
                    if tf > 0:
                        df = self.get_doc_freq(term)
                        if df > 0:
                            # BM25权重计算
                            idf = np.log((self.total_docs - df + 0.5) / (df + 0.5) + 1)
                            tf_normalized = tf / (tf + k1 * (1 - b + b * doc_len / self.avg_doc_length))
                            weights[term] = idf * tf_normalized
                
                self.doc_weights[doc_id] = weights
    
    def save(self, file_path: str):
        """
        将索引保存到文件
        
        Args:
            file_path: 保存路径
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            with open(file_path, 'wb') as f:
                pickle.dump(self, f)
            
            logger.info(f"索引已保存到: {file_path}")
            return True
        except Exception as e:
            logger.error(f"保存索引失败: {str(e)}")
            return False
    
    @staticmethod
    def load(file_path: str) -> Optional['InvertedIndex']:
        """
        从文件加载索引
        
        Args:
            file_path: 索引文件路径
            
        Returns:
            InvertedIndex: 加载的索引对象，加载失败返回None
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"索引文件不存在: {file_path}")
                return None
            
            with open(file_path, 'rb') as f:
                index = pickle.load(f)
            
            logger.info(f"索引已从 {file_path} 加载")
            return index
        except Exception as e:
            logger.error(f"加载索引失败: {str(e)}")
            return None


class Indexer:
    """索引管理器，负责文档索引的创建、更新和管理"""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """单例模式实现"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(Indexer, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """初始化索引管理器"""
        with self._lock:
            if self._initialized:
                return
            
            # 加载配置
            app_config = config_loader.get_config("app_config") or {}
            self.index_path = app_config.get("paths", {}).get("index_db", "data/index_db/")
            
            # 确保索引目录存在
            os.makedirs(self.index_path, exist_ok=True)
            
            # 创建倒排索引
            self.inverted_index = InvertedIndex()
            
            # 文档ID到Document对象的映射
            self.documents = {}
            
            # 完成初始化
            self._initialized = True
            logger.info("索引管理器初始化完成")
    
    def build_index(self, documents: List[Document] = None, rebuild: bool = False):
        """
        为文档构建索引
        
        Args:
            documents: 要索引的文档列表，如果为None，则使用数据源管理器中的所有文档
            rebuild: 是否重建索引，如果为True，则清空现有索引
        """
        with self._lock:
            if rebuild:
                logger.info("重建索引...")
                self.inverted_index = InvertedIndex()
                self.documents = {}
            
            # 如果未提供文档，使用数据源管理器中的所有文档
            if documents is None:
                documents = datasource_manager.get_all_documents()
            
            # 如果没有文档，直接返回
            if not documents:
                logger.warning("没有文档可索引")
                return
            
            logger.info(f"开始索引 {len(documents)} 个文档")
            
            # 使用tqdm显示进度
            for doc in tqdm(documents, desc="索引进度"):
                # 如果文档已索引且未更新，跳过
                if doc.doc_id in self.documents:
                    existing_doc = self.documents[doc.doc_id]
                    if existing_doc.last_modified >= doc.last_modified:
                        continue
                    
                    # 如果文档已更新，先移除旧索引
                    self.inverted_index.remove_document(doc.doc_id)
                
                # 处理文档文本
                tokens = text_processor.process_text(doc.content)
                
                # 添加到索引
                self.inverted_index.add_document(doc.doc_id, tokens, positions=True)
                
                # 更新文档映射
                self.documents[doc.doc_id] = doc
            
            # 计算文档权重
            self.inverted_index.compute_doc_weights('tfidf')
            
            logger.info(f"索引构建完成，共索引了 {self.inverted_index.get_total_docs()} 个文档，"
                        f"{len(self.inverted_index.get_terms())} 个唯一词项")
    
    def add_document(self, doc: Document):
        """
        向索引中添加单个文档
        
        Args:
            doc: 文档对象
            
        Returns:
            bool: 添加成功返回True，否则返回False
        """
        with self._lock:
            try:
                # 检查文档ID是否已存在
                if doc.doc_id in self.documents:
                    existing_doc = self.documents[doc.doc_id]
                    # 如果文档未更新，跳过
                    if existing_doc.last_modified >= doc.last_modified:
                        return True
                    
                    # 如果文档已更新，先移除旧索引
                    self.inverted_index.remove_document(doc.doc_id)
                
                # 处理文档文本
                tokens = text_processor.process_text(doc.content)
                
                # 添加到索引
                self.inverted_index.add_document(doc.doc_id, tokens, positions=True)
                
                # 更新文档映射
                self.documents[doc.doc_id] = doc
                
                # 重新计算文档权重
                self.inverted_index.compute_doc_weights('tfidf')
                
                return True
            except Exception as e:
                logger.error(f"添加文档到索引失败: {doc.doc_id}, 错误: {str(e)}")
                return False
    
    def remove_document(self, doc_id: str):
        """
        从索引中移除文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            bool: 移除成功返回True，否则返回False
        """
        with self._lock:
            try:
                # 检查文档是否存在
                if doc_id not in self.documents:
                    logger.warning(f"文档不存在: {doc_id}")
                    return False
                
                # 从索引中移除
                self.inverted_index.remove_document(doc_id)
                
                # 从文档映射中移除
                del self.documents[doc_id]
                
                # 重新计算文档权重
                self.inverted_index.compute_doc_weights('tfidf')
                
                return True
            except Exception as e:
                logger.error(f"从索引中移除文档失败: {doc_id}, 错误: {str(e)}")
                return False
    
    def update_document(self, doc: Document):
        """
        更新文档索引
        
        Args:
            doc: 更新后的文档对象
            
        Returns:
            bool: 更新成功返回True，否则返回False
        """
        with self._lock:
            # 先移除旧文档，再添加新文档
            self.remove_document(doc.doc_id)
            return self.add_document(doc)
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        获取文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            Document: 文档对象，如果不存在返回None
        """
        return self.documents.get(doc_id)
    
    def get_all_documents(self) -> List[Document]:
        """
        获取所有索引的文档
        
        Returns:
            List[Document]: 文档对象列表
        """
        return list(self.documents.values())
    
    def save_index(self, file_name: str = "inverted_index.pkl"):
        """
        保存索引到文件
        
        Args:
            file_name: 索引文件名
            
        Returns:
            bool: 保存成功返回True，否则返回False
        """
        with self._lock:
            file_path = os.path.join(self.index_path, file_name)
            return self.inverted_index.save(file_path)
    
    def load_index(self, file_name: str = "inverted_index.pkl") -> bool:
        """
        从文件加载索引
        
        Args:
            file_name: 索引文件名
            
        Returns:
            bool: 加载成功返回True，否则返回False
        """
        with self._lock:
            file_path = os.path.join(self.index_path, file_name)
            
            # 尝试加载索引
            loaded_index = InvertedIndex.load(file_path)
            if loaded_index:
                self.inverted_index = loaded_index
                
                # 还需要重新加载文档
                doc_ids = self.inverted_index.get_doc_ids()
                self.documents = {}
                
                # 从数据源加载文档
                for doc_id in doc_ids:
                    doc = datasource_manager.get_document(doc_id)
                    if doc:
                        self.documents[doc_id] = doc
                
                logger.info(f"索引已加载，包含 {len(doc_ids)} 个文档，找到 {len(self.documents)} 个对应文档")
                return True
            
            return False
    
    def save_documents(self, file_name: str = "documents.json"):
        """
        保存文档映射到文件
        
        Args:
            file_name: 文档映射文件名
            
        Returns:
            bool: 保存成功返回True，否则返回False
        """
        with self._lock:
            file_path = os.path.join(self.index_path, file_name)
            
            try:
                # 确保目录存在
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                
                # 将文档对象转换为字典
                docs_dict = {doc_id: doc.to_dict() for doc_id, doc in self.documents.items()}
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(docs_dict, f, ensure_ascii=False, indent=2)
                
                logger.info(f"文档映射已保存到: {file_path}")
                return True
            except Exception as e:
                logger.error(f"保存文档映射失败: {str(e)}")
                return False
    
    def load_documents(self, file_name: str = "documents.json") -> bool:
        """
        从文件加载文档映射
        
        Args:
            file_name: 文档映射文件名
            
        Returns:
            bool: 加载成功返回True，否则返回False
        """
        with self._lock:
            file_path = os.path.join(self.index_path, file_name)
            
            try:
                if not os.path.exists(file_path):
                    logger.error(f"文档映射文件不存在: {file_path}")
                    return False
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    docs_dict = json.load(f)
                
                # 将字典转换为文档对象
                from core.datasource.datasource_manager import Document
                self.documents = {doc_id: Document.from_dict(doc_data) 
                                 for doc_id, doc_data in docs_dict.items()}
                
                logger.info(f"从 {file_path} 加载了 {len(self.documents)} 个文档")
                return True
            except Exception as e:
                logger.error(f"加载文档映射失败: {str(e)}")
                return False


# 提供全局访问点
indexer = Indexer()