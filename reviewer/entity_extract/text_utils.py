"""
文本处理工具模块 (V2)

提供文本清理、Unicode 规范化和文档提取功能。
"""

import re
import unicodedata


def clean_text(text: str) -> str:
    """
    清理文本：移除控制字符，规范化 Unicode
    
    专为中文文本优化，处理 PDF 提取中常见的问题：
    - 移除控制字符（如 \\x01）
    - Unicode 规范化（NFKC），将兼容汉字转为标准汉字
    - 清理多余空格
    
    Args:
        text: 原始文本
        
    Returns:
        清理后的文本
    """
    # 移除控制字符（保留 \\n 和 \\t）
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)
    
    # 清理多余空格
    text = re.sub(r' +', ' ', text)
    
    # Unicode 规范化（NFKC）
    # 将兼容汉字转换为标准汉字：⾼ → 高, ⼩ → 小
    text = unicodedata.normalize('NFKC', text)
    
    return text


def extract_pdf_text(pdf_path: str) -> str:
    """
    从 PDF 提取并清理文本
    
    Args:
        pdf_path: PDF 文件路径
        
    Returns:
        清理后的文本
    """
    try:
        import PyPDF2
    except ImportError:
        raise ImportError("需要安装 PyPDF2: pip install PyPDF2")
    
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text_parts = [
            page.extract_text() 
            for page in reader.pages 
            if page.extract_text()
        ]
        text = '\n'.join(text_parts)
    
    return clean_text(text)


def extract_docx_text(docx_path: str) -> str:
    """
    从 DOCX 提取并清理文本
    
    Args:
        docx_path: DOCX 文件路径
        
    Returns:
        清理后的文本
    """
    try:
        from docx import Document
    except ImportError:
        raise ImportError("需要安装 python-docx: pip install python-docx")
    
    doc = Document(docx_path)
    text_parts = [
        paragraph.text 
        for paragraph in doc.paragraphs 
        if paragraph.text.strip()
    ]
    text = '\n'.join(text_parts)
    
    return clean_text(text)
