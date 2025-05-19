"""
向量空间模型实现模块
"""
import math
import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import Counter, defaultdict

from core.utils.logger import get_logger
from core.utils.gpu_accelerator import gpu_accelerator

logger = get_logger(__name__)


class VectorSpaceModel:
    """向量空间检索模型实现"""
    
    def __init__(self, use_idf: bool = True, normalize: bool = True):
        """
        初始化向量空间模型
        
        Args:
            use_idf: 是否使用IDF加权
            normalize: 是否对向量进行归一化
        """
        self.use_idf = use_idf
        self.normalize = normalize
        
        # 文档向量
        self.document_vectors = {}  # doc_id -> {term: weight}
        # 词项空间
        self.term_space = set()
        # 文档集
        self.documents = {}  # doc_id -> 分词后的文档
        # 文档频率
        self.document_freq = defaultdict(int)  # term -> df
        # 总文档数
        self.total_docs = 0
        # 文档向量长度(用于相似度计算)
        self.vector_norms = {}  # doc_id -> 向量长度
    
    def add_document(self, doc_id: str, tokens: List[str]):
        """
        添加文档到模型
        
        Args:
            doc_id: 文档ID
            tokens: 分词后的文档
        """
        self.documents[doc_id] = tokens
        self.total_docs = len(self.documents)
        
        # 统计词频
        term_freq = Counter(tokens)
        
        # 更新词项空间
        self.term_space.update(term_freq.keys())
        
        # 更新文档频率
        for term in term_freq:
            self.document_freq[term] += 1
        
        # 文档向量初始化为词频
        self.document_vectors[doc_id] = {term: freq for term, freq in term_freq.items()}
    
    def fit(self):
        """训练向量空间模型，计算文档向量"""
        self.total_docs = len(self.documents)
        
        # 构建完整的文档向量
        for doc_id, vector in self.document_vectors.items():
            # 如果使用IDF加权
            if self.use_idf:
                for term in vector:
                    df = self.document_freq[term]
                    if df > 0:  # 防止除零错误
                        # TF-IDF
                        idf = math.log(self.total_docs / df)
                        vector[term] *= idf
            
            # 计算向量范数(L2范数)
            norm = math.sqrt(sum(w * w for w in vector.values()))
            self.vector_norms[doc_id] = norm
            
            # 如果需要归一化
            if self.normalize and norm > 0:
                for term in vector:
                    vector[term] /= norm
    
    def get_document_vector(self, doc_id: str) -> Dict[str, float]:
        """
        获取文档的向量表示
        
        Args:
            doc_id: 文档ID
            
        Returns:
            Dict[str, float]: 文档向量
        """
        return self.document_vectors.get(doc_id, {})
    
    def get_term_weight(self, doc_id: str, term: str) -> float:
        """
        获取词项在文档中的权重
        
        Args:
            doc_id: 文档ID
            term: 词项
            
        Returns:
            float: 词项权重
        """
        return self.document_vectors.get(doc_id, {}).get(term, 0.0)
    
    def _build_query_vector(self, query_tokens: List[str]) -> Dict[str, float]:
        """
        构建查询向量
        
        Args:
            query_tokens: 查询分词结果
            
        Returns:
            Dict[str, float]: 查询向量
        """
        if not query_tokens:
            return {}
        
        # 统计词频
        term_freq = Counter(query_tokens)
        
        # 构建查询向量
        query_vector = {term: freq for term, freq in term_freq.items()}
        
        # 如果使用IDF加权
        if self.use_idf:
            for term in query_vector:
                df = self.document_freq.get(term, 0)
                if df > 0:  # 防止除零错误
                    # TF-IDF
                    idf = math.log(self.total_docs / df)
                    query_vector[term] *= idf
        
        # 计算向量范数
        norm = math.sqrt(sum(w * w for w in query_vector.values()))
        
        # 如果需要归一化
        if self.normalize and norm > 0:
            for term in query_vector:
                query_vector[term] /= norm
        
        return query_vector
    
    def cosine_similarity(self, query_vector: Dict[str, float], doc_id: str) -> float:
        """
        计算查询向量和文档向量的余弦相似度
        
        Args:
            query_vector: 查询向量
            doc_id: 文档ID
            
        Returns:
            float: 余弦相似度
        """
        if not query_vector or doc_id not in self.document_vectors:
            return 0.0
        
        doc_vector = self.document_vectors[doc_id]
        
        # 计算点积
        dot_product = sum(query_vector.get(term, 0) * weight 
                          for term, weight in doc_vector.items())
        
        # 如果向量已经归一化，直接返回点积
        if self.normalize:
            return dot_product
        
        # 计算查询向量长度
        query_norm = math.sqrt(sum(w * w for w in query_vector.values()))
        
        # 计算文档向量长度
        doc_norm = self.vector_norms.get(doc_id, 0)
        
        # 计算余弦相似度
        if query_norm > 0 and doc_norm > 0:
            return dot_product / (query_norm * doc_norm)
        else:
            return 0.0
    
    def query(self, query_tokens: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
        """
        执行查询，返回最相关的文档
        
        Args:
            query_tokens: 查询分词结果
            top_n: 返回的最大结果数量
            
        Returns:
            List[Tuple[str, float]]: (文档ID, 相似度得分)列表，按相似度降序排序
        """
        if not query_tokens or not self.document_vectors:
            return []
        
        # 构建查询向量
        query_vector = self._build_query_vector(query_tokens)
        
        # 检查是否可以使用GPU加速
        use_gpu = gpu_accelerator.is_gpu_available()
        
        if use_gpu:
            logger.info("使用GPU加速进行向量空间检索")
            # 将文档向量和查询向量转换为矩阵格式，以便进行GPU加速计算
            try:
                # 提取所有文档ID
                doc_ids = list(self.document_vectors.keys())
                
                # 获取所有词项
                all_terms = sorted(list(self.term_space))
                term_to_index = {term: i for i, term in enumerate(all_terms)}
                
                # 建立查询向量矩阵
                query_vec_matrix = np.zeros(len(all_terms))
                for term, weight in query_vector.items():
                    if term in term_to_index:
                        query_vec_matrix[term_to_index[term]] = weight
                
                # 建立文档向量矩阵
                doc_matrix = np.zeros((len(doc_ids), len(all_terms)))
                for i, doc_id in enumerate(doc_ids):
                    doc_vec = self.document_vectors[doc_id]
                    for term, weight in doc_vec.items():
                        if term in term_to_index:
                            doc_matrix[i, term_to_index[term]] = weight
                
                # 使用GPU计算余弦相似度
                similarities = gpu_accelerator.cosine_similarity_gpu(query_vec_matrix, doc_matrix)
                
                # 创建结果列表
                similarity_pairs = [(doc_ids[i], float(sim)) for i, sim in enumerate(similarities)]
                
                # 按相似度降序排序
                sorted_similarities = sorted(similarity_pairs, key=lambda x: x[1], reverse=True)
                
                # 返回前N个结果
                return sorted_similarities[:top_n]
            except Exception as e:
                logger.error(f"GPU向量空间检索失败，回退到CPU: {str(e)}")
                # 继续使用CPU方法
        
        # CPU方式计算（默认方法或回退方案）
        # 计算每个文档的相似度
        similarities = {}
        for doc_id in self.document_vectors:
            similarity = self.cosine_similarity(query_vector, doc_id)
            similarities[doc_id] = similarity
        
        # 按相似度降序排序
        sorted_similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # 返回前N个结果
        return sorted_similarities[:top_n]