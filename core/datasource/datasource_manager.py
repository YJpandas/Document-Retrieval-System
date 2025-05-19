"""
数据源管理模块，负责协调各类文档解析器，提供统一的文档处理接口
"""
import os
import glob
import time
import threading
from typing import Dict, List, Optional, Union, Tuple, Set
from datetime import datetime

from core.utils.logger import get_logger
from core.utils.config_loader import config_loader
from core.datasource.parsers.pdf_parser import PDFParser
from core.datasource.parsers.docx_parser import DocxParser
from core.datasource.parsers.xlsx_parser import XlsxParser
from core.datasource.parsers.html_parser import HTMLParser

logger = get_logger(__name__)


class Document:
    """文档类，表示单个文档及其元数据"""
    
    def __init__(self, doc_id: str, file_path: str, title: str, content: str, 
                 doc_type: str, metadata: Dict = None, last_modified: float = None):
        """
        初始化文档对象
        
        Args:
            doc_id: 文档唯一标识符
            file_path: 文档文件路径
            title: 文档标题
            content: 文档内容
            doc_type: 文档类型(pdf, docx, xlsx, html等)
            metadata: 文档元数据
            last_modified: 文档最后修改时间(timestamp)
        """
        self.doc_id = doc_id
        self.file_path = file_path
        self.title = title
        self.content = content
        self.doc_type = doc_type
        self.metadata = metadata or {}
        self.last_modified = last_modified or time.time()
    
    def __str__(self) -> str:
        """文档对象的字符串表示"""
        return f"Document(id={self.doc_id}, title={self.title}, type={self.doc_type})"
    
    def to_dict(self) -> Dict:
        """
        将文档对象转换为字典
        
        Returns:
            Dict: 文档对象的字典表示
        """
        return {
            'doc_id': self.doc_id,
            'file_path': self.file_path,
            'title': self.title,
            'content': self.content,
            'doc_type': self.doc_type,
            'metadata': self.metadata,
            'last_modified': self.last_modified
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Document':
        """
        从字典创建文档对象
        
        Args:
            data: 文档数据字典
            
        Returns:
            Document: 文档对象
        """
        return cls(
            doc_id=data.get('doc_id', ''),
            file_path=data.get('file_path', ''),
            title=data.get('title', ''),
            content=data.get('content', ''),
            doc_type=data.get('doc_type', ''),
            metadata=data.get('metadata', {}),
            last_modified=data.get('last_modified', time.time())
        )


class DataSourceManager:
    """数据源管理器，负责管理各类文档解析器，提供统一的文档解析和获取接口"""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """实现单例模式"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DataSourceManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """初始化数据源管理器"""
        with self._lock:
            if self._initialized:
                return
            
            # 加载配置
            app_config = config_loader.get_config("app_config") or {}
            self.raw_path = app_config.get("paths", {}).get("raw_documents", "data/raw_documents/")
            self.processed_path = app_config.get("paths", {}).get("processed_documents", "data/processed/")
            
            # 确保目录存在
            os.makedirs(self.raw_path, exist_ok=True)
            os.makedirs(self.processed_path, exist_ok=True)
            
            # 注册解析器
            self._parsers = {
                'pdf': PDFParser(),
                'docx': DocxParser(),
                'xlsx': XlsxParser(),
                'html': HTMLParser(),
                # 纯文本文件不需要特殊解析器
                'txt': None
            }
            
            # 文档缓存
            self._documents = {}  # doc_id -> Document
            self._file_to_id = {}  # file_path -> doc_id
            
            # 完成初始化
            self._initialized = True
            logger.info("数据源管理器初始化完成")
    
    def register_parser(self, doc_type: str, parser):
        """
        注册文档解析器
        
        Args:
            doc_type: 文档类型
            parser: 解析器实例
        """
        with self._lock:
            self._parsers[doc_type.lower()] = parser
            logger.info(f"注册{doc_type}文档解析器: {parser.__class__.__name__}")
    
    def get_parser(self, doc_type: str):
        """
        获取指定类型的文档解析器
        
        Args:
            doc_type: 文档类型
            
        Returns:
            解析器实例或None
        """
        return self._parsers.get(doc_type.lower())
    
    def parse_file(self, file_path: str) -> Optional[Document]:
        """
        解析单个文件，生成Document对象
        
        Args:
            file_path: 文件路径
            
        Returns:
            Document: 解析后的文档对象，解析失败返回None
        """
        try:
            # 检查文件是否存在
            if not os.path.isfile(file_path):
                logger.error(f"文件不存在: {file_path}")
                return None
            
            # 获取文件类型和最后修改时间
            file_ext = os.path.splitext(file_path)[1].lower().lstrip('.')
            last_modified = os.path.getmtime(file_path)
            
            # 已经存在且未修改，直接返回缓存
            if file_path in self._file_to_id:
                doc_id = self._file_to_id[file_path]
                doc = self._documents.get(doc_id)
                if doc and doc.last_modified >= last_modified:
                    return doc
            
            # 获取解析器
            parser = self.get_parser(file_ext)
            doc_id = self._generate_doc_id(file_path)
            
            # 处理纯文本文件
            if file_ext == 'txt' or parser is None:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    title = os.path.basename(file_path)
                    doc = Document(
                        doc_id=doc_id,
                        file_path=file_path,
                        title=title,
                        content=content,
                        doc_type=file_ext,
                        last_modified=last_modified
                    )
                    self._cache_document(doc)
                    return doc
                except Exception as e:
                    logger.error(f"解析文本文件失败: {file_path}, 错误: {str(e)}")
                    return None
            
            # 使用对应解析器处理文件
            if parser:
                try:
                    parse_result = parser.parse(file_path)
                    if not parse_result:
                        logger.warning(f"解析结果为空: {file_path}")
                        return None
                    
                    title = parse_result.get('title') or os.path.basename(file_path)
                    content = parse_result.get('content', '')
                    metadata = parse_result.get('metadata', {})
                    
                    doc = Document(
                        doc_id=doc_id,
                        file_path=file_path,
                        title=title,
                        content=content,
                        doc_type=file_ext,
                        metadata=metadata,
                        last_modified=last_modified
                    )
                    self._cache_document(doc)
                    return doc
                except Exception as e:
                    logger.error(f"解析文件失败: {file_path}, 错误: {str(e)}")
                    return None
            else:
                logger.warning(f"不支持的文件类型: {file_ext}")
                return None
            
        except Exception as e:
            logger.error(f"处理文件时发生错误: {file_path}, 错误: {str(e)}")
            return None
    
    def _generate_doc_id(self, file_path: str) -> str:
        """
        生成文档ID
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 文档ID
        """
        import hashlib
        # 使用文件路径的哈希值作为文档ID
        return hashlib.md5(file_path.encode('utf-8')).hexdigest()
    
    def _cache_document(self, doc: Document):
        """
        缓存文档对象
        
        Args:
            doc: 文档对象
        """
        self._documents[doc.doc_id] = doc
        self._file_to_id[doc.file_path] = doc.doc_id
    
    def scan_directory(self, directory: str = None, extensions: List[str] = None) -> List[Document]:
        """
        扫描目录，解析所有支持的文档
        
        Args:
            directory: 要扫描的目录，默认为配置中的raw_documents目录
            extensions: 要处理的文件扩展名列表，默认为所有支持的类型
            
        Returns:
            List[Document]: 解析后的文档对象列表
        """
        directory = directory or self.raw_path
        if not os.path.isdir(directory):
            logger.error(f"目录不存在: {directory}")
            return []
        
        # 如果未指定扩展名，使用所有支持的类型
        if extensions is None:
            extensions = list(self._parsers.keys())
        
        documents = []
        for ext in extensions:
            pattern = os.path.join(directory, f"**/*.{ext}")
            file_paths = glob.glob(pattern, recursive=True)
            
            logger.info(f"在目录 {directory} 中找到 {len(file_paths)} 个 .{ext} 文件")
            
            for file_path in file_paths:
                doc = self.parse_file(file_path)
                if doc:
                    documents.append(doc)
        
        logger.info(f"总共解析了 {len(documents)} 个文档")
        return documents
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        通过文档ID获取文档
        
        Args:
            doc_id: 文档ID
            
        Returns:
            Document: 文档对象，如果不存在返回None
        """
        return self._documents.get(doc_id)
    
    def get_all_documents(self) -> List[Document]:
        """
        获取所有已解析的文档
        
        Returns:
            List[Document]: 文档对象列表
        """
        return list(self._documents.values())
    
    def clear_cache(self):
        """清除文档缓存"""
        with self._lock:
            self._documents.clear()
            self._file_to_id.clear()
        logger.info("文档缓存已清除")
    
    def save_documents(self, output_dir: str = None):
        """
        将已解析的文档保存到处理后的目录
        
        Args:
            output_dir: 输出目录，默认为配置中的processed_documents目录
        """
        output_dir = output_dir or self.processed_path
        os.makedirs(output_dir, exist_ok=True)
        
        import json
        for doc_id, doc in self._documents.items():
            output_path = os.path.join(output_dir, f"{doc_id}.json")
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(doc.to_dict(), f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"保存文档失败: {doc_id}, 错误: {str(e)}")
        
        logger.info(f"已将 {len(self._documents)} 个文档保存到 {output_dir}")
    
    def load_documents(self, input_dir: str = None) -> int:
        """
        从处理后的目录加载文档
        
        Args:
            input_dir: 输入目录，默认为配置中的processed_documents目录
            
        Returns:
            int: 加载的文档数量
        """
        input_dir = input_dir or self.processed_path
        if not os.path.isdir(input_dir):
            logger.warning(f"目录不存在: {input_dir}")
            return 0
        
        import json
        count = 0
        for file_name in os.listdir(input_dir):
            if file_name.endswith('.json'):
                file_path = os.path.join(input_dir, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    doc = Document.from_dict(data)
                    self._cache_document(doc)
                    count += 1
                except Exception as e:
                    logger.error(f"加载文档失败: {file_path}, 错误: {str(e)}")
        
        logger.info(f"从 {input_dir} 加载了 {count} 个文档")
        return count


# 提供全局访问点
datasource_manager = DataSourceManager()