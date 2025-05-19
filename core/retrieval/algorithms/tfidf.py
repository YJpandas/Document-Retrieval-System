"""
TF-IDF算法实现模块
"""
import math
import numpy as np
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import Counter

from core.utils.logger import get_logger
from core.utils.gpu_accelerator import gpu_accelerator

logger = get_logger(__name__)


class TFIDFModel:
    """TF-IDF检索模型实现"""
    
    def __init__(self):
        """初始化TF-IDF模型"""
        # 文档集
        self.documents = {}  # doc_id -> 分词后的文档
        # 文档频率 - 包含词项的文档数量
        self.document_freq = {}  # term -> 包含该词项的文档数
        # 词频 - 词项在文档中出现的次数
        self.term_freq = {}  # (doc_id, term) -> 词频
        # TF-IDF权重
        self.tfidf_weights = {}  # (doc_id, term) -> TF-IDF权重
        # 文档向量长度(用于余弦相似度计算)
        self.doc_vector_lengths = {}  # doc_id -> 向量长度
    
    def add_document(self, doc_id: str, tokens: List[str]):
        """
        添加文档到模型
        
        Args:
            doc_id: 文档ID
            tokens: 分词后的文档
        """
        # 存储文档
        self.documents[doc_id] = tokens
        
        # 计算词频
        term_counts = Counter(tokens)
        
        # 更新文档频率和词频
        for term, count in term_counts.items():
            # 更新文档频率
            if term not in self.document_freq:
                self.document_freq[term] = 0
            self.document_freq[term] += 1
            
            # 更新词频
            self.term_freq[(doc_id, term)] = count
    
    def fit(self):
        """训练TF-IDF模型，计算所有文档的TF-IDF权重"""
        # 计算IDF
        total_docs = len(self.documents)
        idf = {}  # 词项的逆文档频率
        
        for term, doc_freq in self.document_freq.items():
            # 计算IDF: log(N / df)
            idf[term] = math.log(total_docs / doc_freq)
        
        # 计算TF-IDF
        for doc_id, tokens in self.documents.items():
            doc_length = len(tokens)
            
            # 文档向量(用于计算向量长度)
            doc_vector = []
            
            # 为文档中的每个词项计算TF-IDF
            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                # 计算TF: count / doc_length (也可以用其他TF计算方式)
                tf = count / doc_length
                
                # 计算TF-IDF: tf * idf
                tfidf = tf * idf.get(term, 0)
                
                # 存储TF-IDF权重
                self.tfidf_weights[(doc_id, term)] = tfidf
                
                # 添加到文档向量
                doc_vector.append(tfidf)
            
            # 计算文档向量长度(L2范数)
            self.doc_vector_lengths[doc_id] = np.linalg.norm(doc_vector)
    
    def get_tfidf(self, doc_id: str, term: str) -> float:
        """
        获取指定文档中词项的TF-IDF值
        
        Args:
            doc_id: 文档ID
            term: 词项
            
        Returns:
            float: TF-IDF值
        """
        return self.tfidf_weights.get((doc_id, term), 0.0)
    
    def get_document_vector(self, doc_id: str) -> Dict[str, float]:
        """
        获取文档的TF-IDF向量
        
        Args:
            doc_id: 文档ID
            
        Returns:
            Dict[str, float]: 词项到TF-IDF值的映射
        """
        if doc_id not in self.documents:
            return {}
        
        vector = {}
        for term in set(self.documents[doc_id]):
            vector[term] = self.get_tfidf(doc_id, term)
        
        return vector
    
    def query(self, query_tokens: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
        """
        执行查询，返回最相关的文档
        
        Args:
            query_tokens: 查询分词结果
            top_n: 返回的最大结果数量
            
        Returns:
            List[Tuple[str, float]]: (文档ID, 相似度得分)列表，按相似度降序排序
        """
        if not query_tokens or not self.documents:
            return []
        
        # 计算查询的TF-IDF向量
        query_vector = {}
        query_length = len(query_tokens)
        
        # 计算查询中每个词项的TF
        term_counts = Counter(query_tokens)
        
        # 总文档数
        total_docs = len(self.documents)
        
        # 计算查询的TF-IDF向量
        for term, count in term_counts.items():
            # 计算TF: count / query_length
            tf = count / query_length
            
            # 计算IDF(如果词项在文档集中)
            if term in self.document_freq:
                idf = math.log(total_docs / self.document_freq[term])
            else:
                idf = 0
            
            # 计算TF-IDF
            query_vector[term] = tf * idf
        
        # 检查是否可以使用GPU加速
        use_gpu = gpu_accelerator.is_gpu_available()
        
        if use_gpu:
            logger.info("使用GPU加速进行TF-IDF检索")
            try:
                # 获取所有文档ID
                doc_ids = list(self.documents.keys())
                
                # 获取查询中的所有词项
                query_terms = set(query_vector.keys())
                
                # 构建文档矩阵
                # 只考虑查询中出现的词项，以提高效率
                dimensions = len(query_terms)
                doc_matrix = np.zeros((len(doc_ids), dimensions))
                
                # 为查询词项建立索引映射
                term_to_idx = {term: i for i, term in enumerate(query_terms)}
                
                # 构建查询向量
                query_vec_array = np.zeros(dimensions)
                for term, weight in query_vector.items():
                    if term in term_to_idx:
                        query_vec_array[term_to_idx[term]] = weight
                
                # 构建文档矩阵
                for i, doc_id in enumerate(doc_ids):
                    for term in query_terms:
                        if term in term_to_idx:
                            tfidf = self.get_tfidf(doc_id, term)
                            doc_matrix[i, term_to_idx[term]] = tfidf
                
                # 计算余弦相似度
                # 使用GPU加速
                similarities = gpu_accelerator.cosine_similarity_gpu(query_vec_array, doc_matrix)
                
                # 构建结果
                result_pairs = [(doc_ids[i], float(sim)) for i, sim in enumerate(similarities)]
                
                # 按相似度降序排序
                sorted_results = sorted(result_pairs, key=lambda x: x[1], reverse=True)
                
                # 返回前N个结果
                return sorted_results[:top_n]
            except Exception as e:
                logger.error(f"GPU TF-IDF检索失败，回退到CPU: {str(e)}")
                # 继续使用CPU方法
        
        # CPU方式计算（默认方法或回退方案）
        # 计算查询向量长度
        query_vector_length = np.linalg.norm(list(query_vector.values()))
        
        # 如果查询向量为零向量，返回空结果
        if query_vector_length == 0:
            return []
        
        # 计算余弦相似度
        similarity_scores = {}
        
        for doc_id in self.documents:
            # 计算文档和查询的点积
            dot_product = 0
            for term, query_tfidf in query_vector.items():
                doc_tfidf = self.get_tfidf(doc_id, term)
                dot_product += query_tfidf * doc_tfidf
            
            # 计算余弦相似度
            doc_vector_length = self.doc_vector_lengths.get(doc_id, 0)
            if doc_vector_length > 0:
                similarity = dot_product / (query_vector_length * doc_vector_length)
            else:
                similarity = 0
            
            similarity_scores[doc_id] = similarity
        
        # 按相似度降序排序
        sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 返回前N个结果
        return sorted_scores[:top_n]