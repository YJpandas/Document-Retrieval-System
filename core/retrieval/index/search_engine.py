"""
搜索引擎模块，实现各种检索算法，提供统一的搜索接口
"""
import re
import time
import threading
from typing import Dict, List, Set, Tuple, Any, Optional, Union
import numpy as np
from collections import defaultdict

from core.utils.logger import get_logger
from core.utils.config_loader import config_loader
from core.utils.text_processor import text_processor
from core.datasource.datasource_manager import Document
from core.retrieval.index.indexer import indexer, InvertedIndex

logger = get_logger(__name__)


class SearchResult:
    """搜索结果类，表示单个搜索结果项"""
    
    def __init__(self, doc_id: str, score: float, document: Document = None, 
                 highlights: List[str] = None, metadata: Dict = None):
        """
        初始化搜索结果
        
        Args:
            doc_id: 文档ID
            score: 相关性得分
            document: 文档对象
            highlights: 高亮片段
            metadata: 结果元数据
        """
        self.doc_id = doc_id
        self.score = score
        self.document = document
        self.highlights = highlights or []
        self.metadata = metadata or {}
    
    def __str__(self) -> str:
        return f"SearchResult(doc_id={self.doc_id}, score={self.score:.4f})"
    
    def to_dict(self) -> Dict:
        """
        将搜索结果转换为字典
        
        Returns:
            Dict: 搜索结果的字典表示
        """
        result = {
            'doc_id': self.doc_id,
            'score': self.score,
            'highlights': self.highlights,
            'metadata': self.metadata
        }
        
        if self.document:
            result['title'] = self.document.title
            result['file_path'] = self.document.file_path
            result['doc_type'] = self.document.doc_type
            # 内容太大，只返回前200个字符
            result['content_preview'] = self.document.content[:200] if self.document.content else ""
            result['document_metadata'] = self.document.metadata
        
        return result


class SearchResults:
    """搜索结果集合，包含多个搜索结果及统计信息"""
    
    def __init__(self, query: str, results: List[SearchResult] = None, 
                 elapsed_time: float = 0.0, total_found: int = 0, 
                 total_docs: int = 0, metadata: Dict = None):
        """
        初始化搜索结果集合
        
        Args:
            query: 搜索查询
            results: 搜索结果列表
            elapsed_time: 搜索耗时(秒)
            total_found: 找到的结果总数
            total_docs: 索引的文档总数
            metadata: 结果集元数据
        """
        self.query = query
        self.results = results or []
        self.elapsed_time = elapsed_time
        self.total_found = total_found
        self.total_docs = total_docs
        self.metadata = metadata or {}
    
    def __len__(self) -> int:
        return len(self.results)
    
    def __getitem__(self, index) -> SearchResult:
        return self.results[index]
    
    def __iter__(self):
        return iter(self.results)
    
    def add_result(self, result: SearchResult):
        """添加搜索结果"""
        self.results.append(result)
        self.total_found += 1
    
    def sort_by_score(self):
        """按得分排序结果"""
        self.results.sort(key=lambda r: r.score, reverse=True)
    
    def to_dict(self) -> Dict:
        """
        将搜索结果集合转换为字典
        
        Returns:
            Dict: 搜索结果集合的字典表示
        """
        return {
            'query': self.query,
            'results': [result.to_dict() for result in self.results],
            'elapsed_time': self.elapsed_time,
            'total_found': self.total_found,
            'total_docs': self.total_docs,
            'metadata': self.metadata
        }


class SearchEngine:
    """搜索引擎类，提供各种检索算法，统一搜索接口"""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """单例模式实现"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(SearchEngine, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """初始化搜索引擎"""
        with self._lock:
            if self._initialized:
                return
            
            # 加载配置
            self.config = config_loader.get_config("app_config") or {}
            retrieval_config = self.config.get("retrieval", {})
            
            # 检索配置
            self.default_algorithm = retrieval_config.get("default_algorithm", "tfidf")
            self.max_results = retrieval_config.get("max_results", 20)
            self.min_score = retrieval_config.get("min_score", 0.3)
            self.enable_spell_check = retrieval_config.get("enable_spell_check", True)
            
            # 注册搜索算法
            self._algorithms = {
                "boolean": self._boolean_search,
                "tfidf": self._tfidf_search,
                "vector_space": self._vector_space_search,
                "bm25": self._bm25_search
            }
            
            self._initialized = True
            logger.info("搜索引擎初始化完成")
    
    def search(self, query: str, algorithm: str = None, max_results: int = None,
              min_score: float = None, highlight: bool = True, metadata: Dict = None) -> SearchResults:
        """
        执行搜索
        
        Args:
            query: 搜索查询
            algorithm: 搜索算法，如果为None则使用默认算法
            max_results: 最大结果数量，如果为None则使用默认值
            min_score: 最小相关性得分，如果为None则使用默认值
            highlight: 是否高亮结果
            metadata: 传递给搜索结果集的元数据
            
        Returns:
            SearchResults: 搜索结果集合
        """
        if not query or not query.strip():
            return SearchResults(query, [], 0.0, 0, indexer.inverted_index.get_total_docs(), metadata)
        
        # 使用默认值
        algorithm = algorithm or self.default_algorithm
        max_results = max_results or self.max_results
        min_score = min_score or self.min_score
        
        # 执行搜索
        start_time = time.time()
        
        # 选择搜索算法
        search_func = self._algorithms.get(algorithm.lower())
        if not search_func:
            logger.warning(f"未知的搜索算法: {algorithm}，使用默认算法: {self.default_algorithm}")
            search_func = self._algorithms.get(self.default_algorithm)
        
        # 调用相应的搜索算法
        results = search_func(query, max_results, min_score)
        
        # 计算耗时
        elapsed_time = time.time() - start_time
        
        # 创建搜索结果集
        search_results = SearchResults(
            query=query,
            results=results,
            elapsed_time=elapsed_time,
            total_found=len(results),
            total_docs=indexer.inverted_index.get_total_docs(),
            metadata=metadata
        )
        
        # 如果需要，添加高亮
        if highlight:
            self._add_highlights(search_results, query)
        
        # 记录搜索日志
        logger.info(f"搜索完成: 查询='{query}', 算法={algorithm}, "
                   f"找到={len(results)}/{indexer.inverted_index.get_total_docs()}, "
                   f"耗时={elapsed_time:.4f}秒")
        
        return search_results
    
    def _boolean_search(self, query: str, max_results: int, min_score: float) -> List[SearchResult]:
        """
        布尔检索算法
        
        Args:
            query: 搜索查询
            max_results: 最大结果数量
            min_score: 最小相关性得分
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        # 解析布尔查询
        tokens, operators = self._parse_boolean_query(query)
        
        if not tokens:
            return []
        
        # 处理单个词项的情况
        if len(tokens) == 1:
            term = tokens[0]
            processed_term = text_processor.process_text(term)
            if not processed_term:
                return []
            
            term = processed_term[0]
            # 获取倒排列表
            postings = indexer.inverted_index.get_postings(term)
            
            results = []
            for doc_id in postings:
                # 简单评分：词频
                score = indexer.inverted_index.get_term_freq(term, doc_id) / indexer.inverted_index.get_doc_length(doc_id)
                document = indexer.get_document(doc_id)
                
                if score >= min_score:
                    results.append(SearchResult(doc_id, score, document))
            
            # 按得分排序
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:max_results]
        
        # 处理多个词项的情况
        result_sets = []
        
        for term in tokens:
            processed_term = text_processor.process_text(term)
            if not processed_term:
                continue
            
            term = processed_term[0]
            # 获取倒排列表
            postings = set(indexer.inverted_index.get_postings(term))
            result_sets.append(postings)
        
        # 应用布尔操作
        final_results = result_sets[0]
        for i, op in enumerate(operators):
            if i + 1 < len(result_sets):
                if op == 'AND':
                    final_results = final_results & result_sets[i + 1]
                elif op == 'OR':
                    final_results = final_results | result_sets[i + 1]
                elif op == 'NOT':
                    final_results = final_results - result_sets[i + 1]
        
        # 转换为结果对象
        results = []
        for doc_id in final_results:
            # 简单评分：匹配词项数 / 总词项数
            score = len([term for term in tokens if doc_id in indexer.inverted_index.get_postings(term)]) / len(tokens)
            document = indexer.get_document(doc_id)
            
            if score >= min_score:
                results.append(SearchResult(doc_id, score, document))
        
        # 按得分排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]
    
    def _parse_boolean_query(self, query: str) -> Tuple[List[str], List[str]]:
        """
        解析布尔查询
        
        Args:
            query: 布尔查询字符串
            
        Returns:
            Tuple[List[str], List[str]]: 词项列表和操作符列表
        """
        # 简化的布尔查询解析
        # 支持AND, OR, NOT操作符
        query = query.upper()
        
        # 替换常见的布尔操作符为标准形式
        query = query.replace(' AND ', ' AND ')
        query = query.replace(' OR ', ' OR ')
        query = query.replace(' NOT ', ' NOT ')
        
        # 分割查询
        parts = re.split(r'\s+(AND|OR|NOT)\s+', query)
        
        # 提取词项和操作符
        tokens = []
        operators = []
        
        for i, part in enumerate(parts):
            if i % 2 == 0:  # 词项
                tokens.append(part.lower())
            else:  # 操作符
                operators.append(part)
        
        # 如果没有操作符，默认使用AND
        if not operators and len(tokens) > 1:
            operators = ['AND'] * (len(tokens) - 1)
        
        return tokens, operators
    
    def _tfidf_search(self, query: str, max_results: int, min_score: float) -> List[SearchResult]:
        """
        TF-IDF检索算法
        
        Args:
            query: 搜索查询
            max_results: 最大结果数量
            min_score: 最小相关性得分
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        # 处理查询
        query_terms = text_processor.process_text(query)
        if not query_terms:
            return []
        
        # 计算查询权重
        query_weights = {}
        for term in query_terms:
            if term not in query_weights:
                df = indexer.inverted_index.get_doc_freq(term)
                if df > 0:
                    idf = np.log(indexer.inverted_index.get_total_docs() / df)
                    query_weights[term] = idf
                else:
                    query_weights[term] = 0
        
        # 计算查询向量长度
        query_vec_len = np.sqrt(sum(w * w for w in query_weights.values()))
        if query_vec_len > 0:
            # 归一化查询向量
            query_weights = {term: w / query_vec_len for term, w in query_weights.items()}
        
        # 计算文档得分
        doc_scores = {}
        for term in query_terms:
            if term in indexer.inverted_index.get_terms():
                w_qt = query_weights.get(term, 0)
                if w_qt <= 0:
                    continue
                
                # 获取包含该词项的文档
                postings = indexer.inverted_index.get_positional_postings(term)
                for doc_id, tf, _ in postings:
                    # 获取文档中该词项的权重
                    if doc_id in indexer.inverted_index.doc_weights:
                        w_dt = indexer.inverted_index.doc_weights[doc_id].get(term, 0)
                        
                        # 累计得分
                        if doc_id not in doc_scores:
                            doc_scores[doc_id] = 0
                        doc_scores[doc_id] += w_qt * w_dt
        
        # 创建结果列表
        results = []
        for doc_id, score in doc_scores.items():
            if score >= min_score:
                document = indexer.get_document(doc_id)
                results.append(SearchResult(doc_id, score, document))
        
        # 按得分排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]
    
    def _vector_space_search(self, query: str, max_results: int, min_score: float) -> List[SearchResult]:
        """
        向量空间检索算法
        
        Args:
            query: 搜索查询
            max_results: 最大结果数量
            min_score: 最小相关性得分
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        # 向量空间模型与TF-IDF类似，主要区别在于相似度计算方式
        return self._tfidf_search(query, max_results, min_score)
    
    def _bm25_search(self, query: str, max_results: int, min_score: float) -> List[SearchResult]:
        """
        BM25检索算法
        
        Args:
            query: 搜索查询
            max_results: 最大结果数量
            min_score: 最小相关性得分
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        # 处理查询
        query_terms = text_processor.process_text(query)
        if not query_terms:
            return []
        
        # BM25参数
        k1 = 1.2
        b = 0.75
        
        # 计算文档得分
        doc_scores = {}
        for term in query_terms:
            df = indexer.inverted_index.get_doc_freq(term)
            if df <= 0:
                continue
            
            # 计算IDF
            idf = np.log((indexer.inverted_index.get_total_docs() - df + 0.5) / (df + 0.5) + 1)
            
            # 获取包含该词项的文档
            postings = indexer.inverted_index.get_positional_postings(term)
            for doc_id, tf, _ in postings:
                # 获取文档长度
                doc_len = indexer.inverted_index.get_doc_length(doc_id)
                avg_doc_len = indexer.inverted_index.get_avg_doc_length()
                
                # 计算TF部分
                tf_normalized = tf / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
                
                # 累计得分
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                doc_scores[doc_id] += idf * tf_normalized
        
        # 创建结果列表
        results = []
        for doc_id, score in doc_scores.items():
            if score >= min_score:
                document = indexer.get_document(doc_id)
                results.append(SearchResult(doc_id, score, document))
        
        # 按得分排序
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:max_results]
    
    def _add_highlights(self, results: SearchResults, query: str, max_highlights: int = 3, highlight_length: int = 150):
        """
        为搜索结果添加高亮片段
        
        Args:
            results: 搜索结果集合
            query: 搜索查询
            max_highlights: 每个结果的最大高亮片段数
            highlight_length: 高亮片段的长度
        """
        # 处理查询词项
        query_terms = text_processor.process_text(query)
        if not query_terms:
            return
        
        # 为每个结果添加高亮
        for result in results:
            if not result.document or not result.document.content:
                continue
            
            content = result.document.content
            highlights = []
            
            # 查找每个词项的上下文
            for term in query_terms:
                # 在原始内容中查找词项
                term_pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
                matches = list(term_pattern.finditer(content))
                
                # 如果没有精确匹配，尝试模糊匹配
                if not matches:
                    term_pattern = re.compile(re.escape(term), re.IGNORECASE)
                    matches = list(term_pattern.finditer(content))
                
                # 提取上下文
                for match in matches[:max_highlights]:
                    start = max(0, match.start() - highlight_length // 2)
                    end = min(len(content), match.end() + highlight_length // 2)
                    
                    # 调整上下文边界，避免词被截断
                    if start > 0:
                        while start > 0 and content[start] != ' ':
                            start -= 1
                        start += 1  # 跳过空格
                    
                    if end < len(content):
                        while end < len(content) and content[end] != ' ':
                            end += 1
                    
                    # 提取上下文并高亮关键词
                    context = content[start:end]
                    # 替换为高亮标记 (实际前端展示时会替换为HTML标签)
                    highlighted = term_pattern.sub(f"[HIGHLIGHT]{term}[/HIGHLIGHT]", context)
                    
                    if highlighted not in highlights:
                        highlights.append(highlighted)
                        
                        # 达到最大高亮数量后退出
                        if len(highlights) >= max_highlights:
                            break
            
            # 保存高亮片段
            result.highlights = highlights


# 提供全局访问点
search_engine = SearchEngine()