"""
规则引擎模块，负责处理检索规则
"""
import re
import time
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any, Optional, Union
import threading

from core.utils.logger import get_logger
from core.utils.config_loader import config_loader
from core.rules.rule_parser import RuleParser
from core.retrieval.index.search_engine import SearchResults

logger = get_logger(__name__)


class RuleEngine:
    """规则引擎类，负责处理和应用检索规则"""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """单例模式实现"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RuleEngine, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """初始化规则引擎"""
        with self._lock:
            if self._initialized:
                return
            
            # 加载配置
            self.rules_config = config_loader.get_config("rules_config") or {}
            
            # 规则解析器
            self.rule_parser = RuleParser(self.rules_config)
            
            # 加载基本规则
            self._load_basic_rules()
            
            # 完成初始化
            self._initialized = True
            logger.info("规则引擎初始化完成")
    
    def _load_basic_rules(self):
        """加载基本规则"""
        # 从配置文件加载基本规则
        basic_rules = self.rules_config.get("basic_rules", {})
        
        # 排除词规则
        self.exclusion_words = set(basic_rules.get("exclusion_words", []))
        
        # 提升词规则
        self.boost_words = []
        for boost_rule in basic_rules.get("boost_words", []):
            words = boost_rule.get("words", [])
            weight = boost_rule.get("weight", 1.0)
            self.boost_words.append({"words": set(words), "weight": weight})
        
        # 时间提升规则
        self.time_boost = basic_rules.get("time_boost", {})
        self.time_boost_enabled = self.time_boost.get("enabled", False)
        self.time_decay_rate = self.time_boost.get("decay_rate", 0.1)
        self.max_age_months = self.time_boost.get("max_age_months", 24)
        
        # 加载领域特定规则
        self.domain_rules = self.rules_config.get("domain_rules", {})
        
        # 加载高级规则
        advanced_rules = self.rules_config.get("advanced_rules", {})
        
        # 文档类型权重
        self.doc_type_weights = advanced_rules.get("document_type_weights", {})
        
        # 字段权重
        self.field_weights = advanced_rules.get("field_weights", {})
        
        logger.info("基本规则加载完成")
    
    def process_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        处理查询，应用规则
        
        Args:
            query: 原始查询字符串
            
        Returns:
            Tuple[str, Dict]: 处理后的查询和规则元数据
        """
        if not query:
            return query, {}
        
        # 应用查询处理规则
        processed_query = query
        metadata = {}
        
        # 1. 应用排除词规则
        for word in self.exclusion_words:
            # 如果查询中明确包含排除词，则不处理
            if f"-{word}" in processed_query or f"NOT {word}" in processed_query:
                continue
            
            # 如果查询就是排除词，则不处理
            if processed_query.strip() == word:
                continue
            
            # 检查是否需要隐式添加排除词
            if word in processed_query:
                # 在查询中找到排除词，但不是作为NOT操作符的一部分
                word_pattern = r'(?<!\-|\bNOT\s+)\b{}\b'.format(re.escape(word))
                if re.search(word_pattern, processed_query, re.IGNORECASE):
                    # 不做处理，保留原查询
                    pass
        
        # 2. 检测查询类型，解析查询特殊语法
        domain_match = None
        for domain, rules in self.domain_rules.items():
            domain_words = rules.get("boost_words", [])
            matches = [word for word in domain_words if word.lower() in processed_query.lower()]
            if matches:
                domain_match = domain
                domain_weight = rules.get("weight", 1.0)
                metadata["domain"] = domain
                metadata["domain_weight"] = domain_weight
                metadata["domain_matched_words"] = matches
                break
        
        # 3. 解析查询语法
        syntax_info = self.rule_parser.parse_query_syntax(processed_query)
        if syntax_info:
            metadata["syntax"] = syntax_info
        
        # 4. 记录所有可能的提升词匹配
        boost_matches = []
        for boost_rule in self.boost_words:
            words = boost_rule["words"]
            weight = boost_rule["weight"]
            matches = [word for word in words if word.lower() in processed_query.lower()]
            if matches:
                boost_matches.append({
                    "words": matches,
                    "weight": weight
                })
        
        if boost_matches:
            metadata["boost_matches"] = boost_matches
        
        # 返回处理后的查询和元数据
        return processed_query, metadata
    
    def process_results(self, search_results: SearchResults) -> SearchResults:
        """
        处理搜索结果，应用规则
        
        Args:
            search_results: 原始搜索结果
            
        Returns:
            SearchResults: 处理后的搜索结果
        """
        if not search_results or not search_results.results:
            return search_results
        
        # 获取规则元数据
        rule_metadata = search_results.metadata.get("rule_metadata", {})
        
        # 应用提升规则
        boost_matches = rule_metadata.get("boost_matches", [])
        if boost_matches:
            for result in search_results.results:
                if not result.document or not result.document.content:
                    continue
                
                content = result.document.content.lower()
                for boost_rule in boost_matches:
                    words = boost_rule["words"]
                    weight = boost_rule["weight"]
                    
                    for word in words:
                        if word.lower() in content:
                            # 提升得分
                            result.score *= weight
                            # 记录提升原因
                            if "boost_reasons" not in result.metadata:
                                result.metadata["boost_reasons"] = []
                            result.metadata["boost_reasons"].append(f"包含关键词: {word}")
        
        # 应用领域规则
        domain = rule_metadata.get("domain")
        if domain and domain in self.domain_rules:
            domain_weight = rule_metadata.get("domain_weight", 1.0)
            domain_matched_words = rule_metadata.get("domain_matched_words", [])
            
            for result in search_results.results:
                if not result.document or not result.document.content:
                    continue
                
                content = result.document.content.lower()
                matches = [word for word in domain_matched_words if word.lower() in content]
                if matches:
                    # 提升得分
                    result.score *= domain_weight
                    # 记录提升原因
                    if "boost_reasons" not in result.metadata:
                        result.metadata["boost_reasons"] = []
                    result.metadata["boost_reasons"].append(f"领域匹配: {domain} ({', '.join(matches)})")
        
        # 应用文档类型权重
        if self.doc_type_weights:
            for result in search_results.results:
                if not result.document:
                    continue
                
                doc_type = result.document.doc_type
                if doc_type in self.doc_type_weights:
                    weight = self.doc_type_weights[doc_type]
                    result.score *= weight
                    # 记录调整原因
                    if "boost_reasons" not in result.metadata:
                        result.metadata["boost_reasons"] = []
                    result.metadata["boost_reasons"].append(f"文档类型权重: {doc_type} ({weight})")
        
        # 应用时间提升规则
        if self.time_boost_enabled:
            current_time = time.time()
            for result in search_results.results:
                if not result.document:
                    continue
                
                # 获取文档最后修改时间
                last_modified = result.document.last_modified
                if not last_modified:
                    continue
                
                # 计算文档年龄(月)
                age_in_seconds = current_time - last_modified
                age_in_months = age_in_seconds / (30 * 24 * 60 * 60)  # 简化的月份计算
                
                # 如果文档年龄超过最大年龄，不调整
                if age_in_months > self.max_age_months:
                    continue
                
                # 计算时间衰减因子: e^(-decay_rate * age_in_months)
                import math
                time_factor = math.exp(-self.time_decay_rate * age_in_months)
                
                # 调整得分
                original_score = result.score
                result.score *= time_factor
                
                # 记录调整原因
                if "boost_reasons" not in result.metadata:
                    result.metadata["boost_reasons"] = []
                result.metadata["boost_reasons"].append(
                    f"时间因素: {time_factor:.2f} (年龄: {age_in_months:.1f}个月)"
                )
        
        # 重新按得分排序
        search_results.sort_by_score()
        
        return search_results
    
    def apply_custom_rule(self, rule_name: str, search_results: SearchResults, **kwargs) -> SearchResults:
        """
        应用自定义规则
        
        Args:
            rule_name: 规则名称
            search_results: 搜索结果
            **kwargs: 规则参数
            
        Returns:
            SearchResults: 处理后的搜索结果
        """
        # 自定义规则的扩展点
        # 可以通过实现特定的规则函数来扩展功能
        rule_func = getattr(self, f"_rule_{rule_name}", None)
        if not rule_func:
            logger.warning(f"未找到自定义规则: {rule_name}")
            return search_results
        
        try:
            return rule_func(search_results, **kwargs)
        except Exception as e:
            logger.error(f"应用自定义规则失败: {rule_name}, 错误: {str(e)}")
            return search_results
    
    def _rule_content_length(self, search_results: SearchResults, min_length: int = 100) -> SearchResults:
        """
        内容长度规则: 根据文档内容长度进行过滤或排序
        
        Args:
            search_results: 搜索结果
            min_length: 最小内容长度
            
        Returns:
            SearchResults: 处理后的搜索结果
        """
        for result in search_results.results:
            if not result.document or not result.document.content:
                continue
            
            # 计算内容长度
            content_length = len(result.document.content)
            
            # 记录长度信息
            result.metadata["content_length"] = content_length
            
            # 如果内容太短，降低权重
            if content_length < min_length:
                result.score *= 0.8
                if "adjustment_reasons" not in result.metadata:
                    result.metadata["adjustment_reasons"] = []
                result.metadata["adjustment_reasons"].append(f"内容过短: {content_length} 字符")
        
        # 重新排序
        search_results.sort_by_score()
        return search_results
    
    def _rule_keyword_density(self, search_results: SearchResults, min_density: float = 0.005) -> SearchResults:
        """
        关键词密度规则: 根据查询词在文档中的密度调整得分
        
        Args:
            search_results: 搜索结果
            min_density: 最小关键词密度
            
        Returns:
            SearchResults: 处理后的搜索结果
        """
        query_terms = search_results.query.lower().split()
        if not query_terms:
            return search_results
        
        for result in search_results.results:
            if not result.document or not result.document.content:
                continue
            
            content = result.document.content.lower()
            total_length = len(content.split())
            if total_length == 0:
                continue
            
            # 计算关键词出现次数
            keyword_count = 0
            for term in query_terms:
                keyword_count += content.count(term)
            
            # 计算密度
            density = keyword_count / total_length
            
            # 记录密度信息
            result.metadata["keyword_density"] = density
            
            # 根据密度调整得分
            if density < min_density:
                result.score *= 0.9
                if "adjustment_reasons" not in result.metadata:
                    result.metadata["adjustment_reasons"] = []
                result.metadata["adjustment_reasons"].append(f"关键词密度过低: {density:.4f}")
            elif density > 0.05:  # 密度较高
                result.score *= 1.2
                if "adjustment_reasons" not in result.metadata:
                    result.metadata["adjustment_reasons"] = []
                result.metadata["adjustment_reasons"].append(f"关键词密度较高: {density:.4f}")
        
        # 重新排序
        search_results.sort_by_score()
        return search_results


# 提供全局访问点
rule_engine = RuleEngine()