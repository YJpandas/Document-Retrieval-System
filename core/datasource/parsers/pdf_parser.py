"""
PDF解析器，负责解析PDF文件并提取文本内容和元数据
"""
import os
import io
from typing import Dict, Any, Optional, List, Tuple
import PyPDF2
from core.utils.logger import get_logger

logger = get_logger(__name__)


class PDFParser:
    """PDF文件解析器，提取PDF文件的文本内容和元数据"""
    
    def __init__(self):
        """初始化PDF解析器"""
        pass
    
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        解析PDF文件
        
        Args:
            file_path: PDF文件路径
            
        Returns:
            Dict: 包含标题、内容和元数据的字典
        """
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return {}
        
        try:
            result = {
                'title': os.path.basename(file_path),
                'content': '',
                'metadata': {}
            }
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # 提取元数据
                if pdf_reader.metadata:
                    for key, value in pdf_reader.metadata.items():
                        if key.startswith('/'):
                            key = key[1:]  # 去除前缀斜杠
                        result['metadata'][key] = str(value)
                
                # 如果元数据中有标题，则使用元数据中的标题
                if 'Title' in result['metadata'] and result['metadata']['Title']:
                    result['title'] = result['metadata']['Title']
                
                # 提取文本内容
                content_parts = []
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    try:
                        text = page.extract_text()
                        if text:
                            content_parts.append(text)
                    except Exception as e:
                        logger.warning(f"解析PDF第{page_num+1}页失败: {str(e)}")
                
                result['content'] = '\n\n'.join(content_parts)
                
                # 添加页数信息
                result['metadata']['page_count'] = len(pdf_reader.pages)
                
                return result
                
        except Exception as e:
            logger.error(f"解析PDF文件失败: {file_path}, 错误: {str(e)}")
            return {}
    
    def extract_images(self, file_path: str, output_dir: str = None) -> List[str]:
        """
        从PDF文件中提取图片（可选功能）
        
        Args:
            file_path: PDF文件路径
            output_dir: 图片输出目录，如果为None则不保存
            
        Returns:
            List[str]: 保存的图片文件路径列表
        """
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return []
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        image_paths = []
        try:
            # 注意：PyPDF2的图片提取功能有限，这里提供一个简单的实现
            # 实际应用中可能需要使用其他库如pdfminer或pymupdf
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    for img_num, xobj in enumerate(page.images):
                        try:
                            if output_dir:
                                filename = f"page{page_num+1}_img{img_num+1}.{xobj.subtype.lower()}"
                                output_path = os.path.join(output_dir, filename)
                                
                                with open(output_path, 'wb') as img_file:
                                    img_file.write(xobj.data)
                                
                                image_paths.append(output_path)
                        except Exception as e:
                            logger.warning(f"提取PDF图片失败: 第{page_num+1}页, 图片{img_num+1}, 错误: {str(e)}")
            
            return image_paths
                
        except Exception as e:
            logger.error(f"从PDF提取图片失败: {file_path}, 错误: {str(e)}")
            return []