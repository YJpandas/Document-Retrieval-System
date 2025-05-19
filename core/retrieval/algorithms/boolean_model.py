"""
布尔检索模型实现模块
"""
import re
from enum import Enum
from typing import Dict, List, Set, Tuple, Any, Optional, Union
from collections import defaultdict

from core.utils.logger import get_logger

logger = get_logger(__name__)


class TokenType(Enum):
    """词法分析器的词元类型"""
    TERM = 0        # 词项
    AND = 1         # AND 操作符
    OR = 2          # OR 操作符
    NOT = 3         # NOT 操作符
    LPAREN = 4      # 左括号
    RPAREN = 5      # 右括号
    NEAR = 6        # NEAR 操作符
    PHRASE = 7      # 短语查询
    UNKNOWN = 8     # 未知类型


class Token:
    """词法分析器的词元"""
    
    def __init__(self, token_type: TokenType, value: str):
        """
        初始化词元
        
        Args:
            token_type: 词元类型
            value: 词元值
        """
        self.type = token_type
        self.value = value
    
    def __str__(self) -> str:
        return f"Token({self.type}, '{self.value}')"
    
    def __repr__(self) -> str:
        return self.__str__()


class QueryParser:
    """布尔查询解析器"""
    
    def __init__(self):
        """初始化查询解析器"""
        # 操作符优先级
        self.precedence = {
            TokenType.OR: 1,
            TokenType.AND: 2,
            TokenType.NOT: 3,
            TokenType.NEAR: 3,
            TokenType.PHRASE: 4
        }
    
    def tokenize(self, query: str) -> List[Token]:
        """
        将查询字符串分词
        
        Args:
            query: 查询字符串
            
        Returns:
            List[Token]: 词元列表
        """
        # 规范化查询
        query = query.strip()
        
        # 短语查询处理
        phrase_pattern = r'"([^"]+)"'
        phrases = re.findall(phrase_pattern, query)
        
        # 替换短语为占位符，以便后续处理
        for i, phrase in enumerate(phrases):
            placeholder = f"__PHRASE_{i}__"
            query = query.replace(f'"{phrase}"', placeholder)
        
        # 替换特殊操作符为标准形式
        query = re.sub(r'\bAND\b', ' AND ', query, flags=re.IGNORECASE)
        query = re.sub(r'\bOR\b', ' OR ', query, flags=re.IGNORECASE)
        query = re.sub(r'\bNOT\b', ' NOT ', query, flags=re.IGNORECASE)
        query = re.sub(r'\bNEAR/(\d+)\b', r' NEAR/\1 ', query, flags=re.IGNORECASE)
        
        # 确保括号周围有空格
        query = query.replace('(', ' ( ').replace(')', ' ) ')
        
        # 分割查询字符串
        tokens = []
        for part in query.split():
            # 处理短语占位符
            if part.startswith('__PHRASE_') and part.endswith('__'):
                index = int(part[9:-2])
                if index < len(phrases):
                    tokens.append(Token(TokenType.PHRASE, phrases[index]))
                    continue
            
            # 处理操作符和词项
            if part.upper() == 'AND':
                tokens.append(Token(TokenType.AND, 'AND'))
            elif part.upper() == 'OR':
                tokens.append(Token(TokenType.OR, 'OR'))
            elif part.upper() == 'NOT':
                tokens.append(Token(TokenType.NOT, 'NOT'))
            elif part == '(':
                tokens.append(Token(TokenType.LPAREN, '('))
            elif part == ')':
                tokens.append(Token(TokenType.RPAREN, ')'))
            elif part.upper().startswith('NEAR/'):
                try:
                    distance = int(part.split('/')[1])
                    tokens.append(Token(TokenType.NEAR, part.upper()))
                except:
                    # 如果NEAR格式错误，则视为普通词项
                    tokens.append(Token(TokenType.TERM, part.lower()))
            else:
                tokens.append(Token(TokenType.TERM, part.lower()))
        
        # 处理默认操作符(连续的词项之间默认为AND)
        i = 0
        result = []
        while i < len(tokens):
            result.append(tokens[i])
            
            if i < len(tokens) - 1:
                if tokens[i].type in (TokenType.TERM, TokenType.PHRASE, TokenType.RPAREN) and \
                   tokens[i+1].type in (TokenType.TERM, TokenType.PHRASE, TokenType.LPAREN, TokenType.NOT):
                    # 在这些情况下，插入一个AND操作符
                    result.append(Token(TokenType.AND, 'AND'))
            
            i += 1
        
        return result
    
    def parse(self, tokens: List[Token]) -> Union[List[Token], None]:
        """
        将中缀表达式转换为后缀表达式(逆波兰表示法)
        
        Args:
            tokens: 词元列表
            
        Returns:
            List[Token]: 后缀表达式词元列表
        """
        output = []
        operator_stack = []
        
        for token in tokens:
            if token.type in (TokenType.TERM, TokenType.PHRASE):
                # 词项或短语直接输出
                output.append(token)
            elif token.type == TokenType.LPAREN:
                # 左括号入栈
                operator_stack.append(token)
            elif token.type == TokenType.RPAREN:
                # 处理右括号: 弹出所有操作符直到左括号
                while operator_stack and operator_stack[-1].type != TokenType.LPAREN:
                    output.append(operator_stack.pop())
                
                # 弹出左括号
                if operator_stack and operator_stack[-1].type == TokenType.LPAREN:
                    operator_stack.pop()
                else:
                    # 括号不匹配错误
                    logger.error("括号不匹配")
                    return None
            else:
                # 操作符处理
                while operator_stack and \
                      operator_stack[-1].type != TokenType.LPAREN and \
                      self.precedence.get(operator_stack[-1].type, 0) >= self.precedence.get(token.type, 0):
                    output.append(operator_stack.pop())
                
                operator_stack.append(token)
        
        # 将剩余的操作符弹出到输出队列
        while operator_stack:
            if operator_stack[-1].type == TokenType.LPAREN:
                # 括号不匹配错误
                logger.error("括号不匹配")
                return None
            output.append(operator_stack.pop())
        
        return output


class BooleanModel:
    """布尔检索模型实现"""
    
    def __init__(self):
        """初始化布尔检索模型"""
        # 倒排索引: term -> 包含该词项的文档ID集合
        self.inverted_index = defaultdict(set)
        # 位置索引: (doc_id, term) -> term在doc_id中的位置列表
        self.positional_index = defaultdict(list)
        # 文档集
        self.documents = {}  # doc_id -> 文档内容(分词后)
        # 布尔查询解析器
        self.query_parser = QueryParser()
    
    def add_document(self, doc_id: str, tokens: List[str]):
        """
        添加文档到模型
        
        Args:
            doc_id: 文档ID
            tokens: 分词后的文档
        """
        self.documents[doc_id] = tokens
        
        # 构建倒排索引
        for position, term in enumerate(tokens):
            self.inverted_index[term].add(doc_id)
            self.positional_index[(doc_id, term)].append(position)
    
    def get_posting_list(self, term: str) -> Set[str]:
        """
        获取词项的倒排列表
        
        Args:
            term: 词项
            
        Returns:
            Set[str]: 包含该词项的文档ID集合
        """
        return self.inverted_index.get(term, set())
    
    def get_positions(self, doc_id: str, term: str) -> List[int]:
        """
        获取词项在文档中的位置
        
        Args:
            doc_id: 文档ID
            term: 词项
            
        Returns:
            List[int]: 位置列表
        """
        return self.positional_index.get((doc_id, term), [])
    
    def parse_query(self, query: str) -> Union[List[Token], None]:
        """
        解析查询字符串
        
        Args:
            query: 查询字符串
            
        Returns:
            List[Token]: 后缀表达式词元列表
        """
        tokens = self.query_parser.tokenize(query)
        return self.query_parser.parse(tokens)
    
    def evaluate_query(self, postfix_tokens: List[Token]) -> Set[str]:
        """
        评估后缀表达式，执行布尔检索
        
        Args:
            postfix_tokens: 后缀表达式词元列表
            
        Returns:
            Set[str]: 匹配的文档ID集合
        """
        if not postfix_tokens:
            return set()
        
        operand_stack = []
        
        for token in postfix_tokens:
            if token.type == TokenType.TERM:
                # 词项: 获取其倒排列表并入栈
                posting_list = self.get_posting_list(token.value)
                operand_stack.append(posting_list)
            elif token.type == TokenType.PHRASE:
                # 短语查询: 获取短语匹配的文档ID集合
                phrase_terms = token.value.lower().split()
                phrase_docs = self._phrase_search(phrase_terms)
                operand_stack.append(phrase_docs)
            elif token.type == TokenType.AND:
                # AND操作: 两个操作数的交集
                if len(operand_stack) < 2:
                    logger.error("AND操作符缺少操作数")
                    return set()
                
                right = operand_stack.pop()
                left = operand_stack.pop()
                result = left.intersection(right)
                operand_stack.append(result)
            elif token.type == TokenType.OR:
                # OR操作: 两个操作数的并集
                if len(operand_stack) < 2:
                    logger.error("OR操作符缺少操作数")
                    return set()
                
                right = operand_stack.pop()
                left = operand_stack.pop()
                result = left.union(right)
                operand_stack.append(result)
            elif token.type == TokenType.NOT:
                # NOT操作: 所有文档减去操作数
                if len(operand_stack) < 1:
                    logger.error("NOT操作符缺少操作数")
                    return set()
                
                operand = operand_stack.pop()
                result = set(self.documents.keys()) - operand
                operand_stack.append(result)
            elif token.type == TokenType.NEAR:
                # NEAR操作: 两个词项在指定距离内
                if len(operand_stack) < 2:
                    logger.error("NEAR操作符缺少操作数")
                    return set()
                
                try:
                    distance = int(token.value.split('/')[1])
                except:
                    distance = 5  # 默认距离
                
                right_term_docs = operand_stack.pop()
                left_term_docs = operand_stack.pop()
                
                # 获取同时包含两个词项的文档
                common_docs = left_term_docs.intersection(right_term_docs)
                result = set()
                
                # 对于每个文档，检查两个词项是否在指定距离内
                for doc_id in common_docs:
                    # 由于我们没有直接存储哪个词项对应哪个位置，
                    # 这里需要从位置索引中获取原始词项
                    # 这是简化实现，实际使用时需要记录词项和位置的对应关系
                    left_positions = []
                    right_positions = []
                    
                    for term, positions in self.positional_index.items():
                        if term[0] == doc_id and term[1] in left_term_docs:
                            left_positions.extend(positions)
                        elif term[0] == doc_id and term[1] in right_term_docs:
                            right_positions.extend(positions)
                    
                    # 检查是否有满足距离条件的位置对
                    for pos1 in left_positions:
                        for pos2 in right_positions:
                            if abs(pos1 - pos2) <= distance:
                                result.add(doc_id)
                                break
                        if doc_id in result:
                            break
                
                operand_stack.append(result)
        
        # 最终结果应该只有一个元素在栈中
        if len(operand_stack) != 1:
            logger.error("查询评估错误")
            return set()
        
        return operand_stack[0]
    
    def _phrase_search(self, phrase_terms: List[str]) -> Set[str]:
        """
        短语搜索
        
        Args:
            phrase_terms: 短语中的词项列表
            
        Returns:
            Set[str]: 匹配短语的文档ID集合
        """
        if not phrase_terms:
            return set()
        
        # 获取包含所有词项的文档
        docs = set(self.documents.keys())
        for term in phrase_terms:
            docs = docs.intersection(self.get_posting_list(term))
        
        # 如果没有包含所有词项的文档，直接返回空集
        if not docs:
            return set()
        
        # 检查每个文档中的词项是否按顺序连续出现
        result = set()
        for doc_id in docs:
            # 获取每个词项在文档中的位置
            positions = []
            for term in phrase_terms:
                pos = self.get_positions(doc_id, term)
                if not pos:
                    break
                positions.append(pos)
            
            # 如果不是所有词项都有位置信息，跳过该文档
            if len(positions) != len(phrase_terms):
                continue
            
            # 检查是否有连续的位置
            match = False
            for pos1 in positions[0]:
                expected_pos = pos1 + 1
                all_matched = True
                
                for i in range(1, len(phrase_terms)):
                    if expected_pos not in positions[i]:
                        all_matched = False
                        break
                    expected_pos += 1
                
                if all_matched:
                    match = True
                    break
            
            if match:
                result.add(doc_id)
        
        return result
    
    def query(self, query_str: str) -> List[str]:
        """
        执行布尔查询
        
        Args:
            query_str: 查询字符串
            
        Returns:
            List[str]: 匹配的文档ID列表
        """
        # 解析查询
        postfix_tokens = self.parse_query(query_str)
        if not postfix_tokens:
            return []
        
        # 评估查询
        result_set = self.evaluate_query(postfix_tokens)
        
        # 返回结果列表
        return list(result_set)