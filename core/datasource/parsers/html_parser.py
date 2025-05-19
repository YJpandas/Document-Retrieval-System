"""
HTML解析器，负责解析HTML文档并提取文本内容和元数据
"""
import os
import re
from typing import Dict, Any, List, Optional, Tuple
from bs4 import BeautifulSoup
from core.utils.logger import get_logger

logger = get_logger(__name__)


class HTMLParser:
    """HTML文档解析器，提取HTML文件的文本内容和元数据"""
    
    def __init__(self):
        """初始化HTML解析器"""
        # 定义需要过滤的标签列表
        self.filter_tags = ['script', 'style', 'noscript', 'iframe', 'head', 'meta', 'link', 'svg']
        # 定义重要内容标签
        self.content_tags = ['p', 'div', 'article', 'section', 'main', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td']
        # 定义可能包含广告或无关内容的类名或ID模式
        self.noise_patterns = [
            re.compile(r'ad[s-]|banner|foot|footer|footnote|promo', re.I),
            re.compile(r'combx|comment|com-|contact|header|menu|sidebar|tool|widget', re.I),
            re.compile(r'popup|share|social|sponsor|tags|related', re.I)
        ]
    
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        解析HTML文档
        
        Args:
            file_path: HTML文件路径
            
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
            
            # 读取HTML文件
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                html_content = file.read()
            
            # 解析HTML
            soup = BeautifulSoup(html_content, 'html5lib')
            
            # 提取元数据
            metadata = {}
            
            # 提取标题
            if soup.title and soup.title.string:
                result['title'] = soup.title.string.strip()
                metadata['title'] = result['title']
            
            # 提取meta标签信息
            for meta in soup.find_all('meta'):
                meta_name = meta.get('name', meta.get('property', ''))
                meta_content = meta.get('content', '')
                if meta_name and meta_content:
                    # 规范化元数据名称
                    meta_name = meta_name.lower().replace(':', '_')
                    metadata[meta_name] = meta_content
            
            # 提取链接数据
            links = []
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                text = link.get_text(strip=True)
                if href and text:
                    links.append({'href': href, 'text': text})
            
            if links:
                metadata['links'] = links[:100]  # 限制链接数量
            
            # 将元数据添加到结果中
            result['metadata'] = metadata
            
            # 提取正文内容
            # 1. 移除不需要的标签
            for tag in soup.find_all(self.filter_tags):
                tag.decompose()
            
            # 2. 过滤可能的广告或无关内容
            for element in soup.find_all(True, {'class': True}):
                for pattern in self.noise_patterns:
                    class_attr = element.get('class', [])
                    if isinstance(class_attr, list):
                        class_str = ' '.join(class_attr)
                    else:
                        class_str = class_attr
                        
                    if pattern.search(class_str):
                        element.decompose()
                        break
            
            # 3. 提取正文内容
            content_parts = []
            
            # 尝试找到主要内容区域
            main_content = soup.find(['article', 'main', 'div', 'section'], {'id': re.compile(r'content|article|main', re.I)})
            if not main_content:
                main_content = soup.find(['article', 'main', 'div', 'section'], {'class': re.compile(r'content|article|main|text', re.I)})
            
            # 如果找到了主要内容区域，则从该区域提取文本
            if main_content:
                # 提取标题
                for h in main_content.find_all(['h1', 'h2', 'h3'], limit=3):
                    text = h.get_text(strip=True)
                    if text and len(text) > 5:
                        content_parts.append(f"【标题】{text}")
                
                # 提取段落
                for tag in main_content.find_all(self.content_tags):
                    text = tag.get_text(strip=True)
                    if text and len(text) > 20:  # 忽略太短的段落
                        content_parts.append(text)
            else:
                # 如果没有找到主要内容区域，则提取所有可能的内容
                # 提取所有标题
                for h in soup.find_all(['h1', 'h2', 'h3'], limit=5):
                    text = h.get_text(strip=True)
                    if text and len(text) > 5:
                        content_parts.append(f"【标题】{text}")
                
                # 提取所有段落
                for p in soup.find_all('p'):
                    text = p.get_text(strip=True)
                    if text and len(text) > 20:  # 忽略太短的段落
                        content_parts.append(text)
                
                # 如果内容太少，尝试提取更多元素
                if len(content_parts) < 5:
                    for tag in soup.find_all(self.content_tags):
                        text = tag.get_text(strip=True)
                        if text and len(text) > 30 and text not in content_parts:
                            content_parts.append(text)
            
            # 合并内容
            result['content'] = '\n\n'.join(content_parts)
            
            return result
            
        except Exception as e:
            logger.error(f"解析HTML文件失败: {file_path}, 错误: {str(e)}")
            return {}
    
    def extract_images(self, file_path: str, output_dir: str = None) -> List[Dict[str, str]]:
        """
        从HTML文档中提取图片信息（可选功能）
        
        Args:
            file_path: HTML文件路径
            output_dir: 图片输出目录，如果为None则不保存
            
        Returns:
            List[Dict]: 图片信息列表，包含URL、alt文本等
        """
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return []
        
        try:
            # 读取HTML文件
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                html_content = file.read()
            
            # 解析HTML
            soup = BeautifulSoup(html_content, 'html5lib')
            
            image_info = []
            for img in soup.find_all('img', src=True):
                src = img.get('src', '')
                alt = img.get('alt', '')
                title = img.get('title', '')
                
                if src and (src.startswith('http') or src.startswith('/')):
                    info = {
                        'src': src,
                        'alt': alt or title,
                        'title': title or alt
                    }
                    image_info.append(info)
            
            # 注意：这个函数只提取图片信息，不会下载图片
            # 如果需要下载图片，需要实现额外的功能
            
            return image_info
            
        except Exception as e:
            logger.error(f"从HTML提取图片信息失败: {file_path}, 错误: {str(e)}")
            return []
