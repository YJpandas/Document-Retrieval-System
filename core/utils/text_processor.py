"""
文本处理工具模块，提供文本预处理功能
"""
import re
import string
import jieba
import nltk
from typing import List, Set, Optional, Union
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

from core.utils.config_loader import config_loader
from core.utils.logger import get_logger

logger = get_logger(__name__)

# 确保NLTK资源已下载
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("下载NLTK停用词数据...")
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    logger.info("下载NLTK WordNet数据...")
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("下载NLTK Punkt分词器...")
    nltk.download('punkt', quiet=True)


class TextProcessor:
    """文本处理器，提供文本清洗、分词、去停用词等功能"""
    
    _instance = None
    
    def __new__(cls):
        """单例模式实现"""
        if cls._instance is None:
            cls._instance = super(TextProcessor, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """初始化文本处理器"""
        # 加载配置
        self.config = config_loader.get_config("app_config") or {}
        text_config = self.config.get("text_processing", {})
        
        # 语言设置
        self.language = text_config.get("language", "chinese")
        self.enable_stopwords = text_config.get("enable_stopwords", True)
        self.enable_stemming = text_config.get("enable_stemming", True)
        self.min_token_length = text_config.get("min_token_length", 2)
        
        # 加载停用词
        self.stopwords = self._load_stopwords()
        
        # 准备词干提取器和词形还原器
        self.stemmer = PorterStemmer() if self.enable_stemming else None
        self.lemmatizer = WordNetLemmatizer() if self.enable_stemming else None
    
    def _load_stopwords(self) -> Set[str]:
        """
        加载停用词
        
        Returns:
            Set[str]: 停用词集合
        """
        # 加载指定语言的NLTK停用词
        if self.language.lower() == "english":
            stop_words = set(nltk_stopwords.words('english'))
        elif self.language.lower() == "chinese":
            # 中文停用词
            stop_words = set()
            # 尝试从文件加载
            try:
                with open('data/stopwords/chinese_stopwords.txt', 'r', encoding='utf-8') as f:
                    stop_words = set(line.strip() for line in f)
            except FileNotFoundError:
                logger.warning("未找到中文停用词文件，使用基本停用词")
                # 基本中文停用词
                stop_words = {'的', '了', '和', '是', '就', '都', '而', '及', '与', '着',
                             '或', '一个', '没有', '我们', '你们', '他们', '她们', '这个',
                             '那个', '这些', '那些', '不', '在', '有', '个', '中', '为'}
        else:
            stop_words = set()
            logger.warning(f"不支持的语言: {self.language}，使用空停用词集")
        
        return stop_words
    
    def clean_text(self, text: str) -> str:
        """
        清洗文本，去除无用字符
        
        Args:
            text: 原始文本
            
        Returns:
            str: 清洗后的文本
        """
        if not text:
            return ""
        
        # 统一转小写
        text = text.lower()
        
        # 对英文进行处理
        if self.language.lower() == "english":
            # 去除标点符号
            text = re.sub(f'[{string.punctuation}]', ' ', text)
            # 去除数字
            text = re.sub(r'\d+', ' ', text)
            # 去除多余空格
            text = re.sub(r'\s+', ' ', text)
        
        # 对中文进行处理
        elif self.language.lower() == "chinese":
            # 去除标点符号和特殊字符
            text = re.sub(r'[^\w\s\u4e00-\u9fff]+', ' ', text)
            # 去除数字
            text = re.sub(r'\d+', ' ', text)
            # 去除多余空格
            text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        对文本进行分词
        
        Args:
            text: 待分词文本
            
        Returns:
            List[str]: 分词结果
        """
        if not text:
            return []
        
        # 中文分词
        if self.language.lower() == "chinese":
            tokens = jieba.lcut(text)
        # 英文分词
        else:
            tokens = nltk.word_tokenize(text)
        
        # 过滤过短的词
        tokens = [t for t in tokens if len(t) >= self.min_token_length]
        
        return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        去除停用词
        
        Args:
            tokens: 分词结果
            
        Returns:
            List[str]: 去除停用词后的分词结果
        """
        if not self.enable_stopwords:
            return tokens
        
        return [t for t in tokens if t not in self.stopwords]
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """
        词干提取
        
        Args:
            tokens: 分词结果
            
        Returns:
            List[str]: 词干提取后的分词结果
        """
        if not self.enable_stemming or not self.stemmer:
            return tokens
        
        # 仅对英文执行词干提取
        if self.language.lower() == "english":
            return [self.stemmer.stem(t) for t in tokens]
        return tokens
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        词形还原
        
        Args:
            tokens: 分词结果
            
        Returns:
            List[str]: 词形还原后的分词结果
        """
        if not self.enable_stemming or not self.lemmatizer:
            return tokens
        
        # 仅对英文执行词形还原
        if self.language.lower() == "english":
            return [self.lemmatizer.lemmatize(t) for t in tokens]
        return tokens
    
    def process_text(self, text: str) -> List[str]:
        """
        完整文本处理流程：清洗、分词、去停用词、词干提取/词形还原
        
        Args:
            text: 原始文本
            
        Returns:
            List[str]: 处理后的标记列表
        """
        clean_text = self.clean_text(text)
        tokens = self.tokenize(clean_text)
        tokens = self.remove_stopwords(tokens)
        
        # 根据配置选择词干提取或词形还原
        if self.enable_stemming and self.language.lower() == "english":
            # 词干提取优先于词形还原
            tokens = self.stem_tokens(tokens)
        
        return tokens
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        从文本中提取关键词
        
        Args:
            text: 原始文本
            top_n: 返回前N个关键词
            
        Returns:
            List[str]: 关键词列表
        """
        # 中文使用jieba提取关键词
        if self.language.lower() == "chinese":
            import jieba.analyse
            keywords = jieba.analyse.extract_tags(text, topK=top_n)
            return keywords
        
        # 英文使用TF-IDF提取关键词
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            # 处理文本
            processed_text = self.process_text(text)
            if not processed_text:
                return []
            
            # 使用TF-IDF提取关键词
            try:
                vectorizer = TfidfVectorizer(max_features=top_n)
                tfidf_matrix = vectorizer.fit_transform([' '.join(processed_text)])
                feature_names = vectorizer.get_feature_names_out()
                
                # 获取词的TF-IDF值
                tfidf_scores = zip(feature_names, tfidf_matrix.toarray()[0])
                # 按得分排序
                sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
                
                return [word for word, score in sorted_scores[:top_n]]
            except Exception as e:
                logger.error(f"提取关键词失败: {str(e)}")
                return []


# 提供全局访问点
text_processor = TextProcessor()