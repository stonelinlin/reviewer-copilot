# Method 1: Using Templates (Recommended for consistency)
import sys
import os

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # demo/
project_root = os.path.dirname(current_dir)  # reviewer-copilot/
sys.path.insert(0, project_root)

import PyPDF2
import html
import json
import re  # 用于文本清理
import unicodedata  # 用于 Unicode 规范化
from reviewer.entity_extract.core import data
from reviewer.entity_extract.openai import OpenAILanguageModel
from reviewer.entity_extract.core.tokenizer import UnicodeTokenizer
from reviewer.entity_extract.ext.template_builder import extract_with_template

# 读取PDF文件
input_file = 'resume.pdf'
print(f"正在读取文件: {input_file}")

with open(input_file, 'rb') as f:
    reader = PyPDF2.PdfReader(f)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    original_text = '\n'.join(text_parts)

# 清理控制字符（\x00-\x1f 和 \x7f-\x9f），保留换行符(\n)和制表符(\t)
print("正在清理文本中的控制字符...")
original_text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', ' ', original_text)
# 清理多余的空格
original_text = re.sub(r' +', ' ', original_text)

# Unicode 规范化：将兼容汉字转换为标准汉字
# NFKC: Normalization Form KC (Compatibility Composition)
print("正在进行 Unicode 规范化（将兼容汉字转为标准汉字）...")
original_text = unicodedata.normalize('NFKC', original_text)

print(f"文件读取成功，清理后共 {len(original_text)} 个字符\n")

doc = data.Document(text=original_text, document_id=input_file)

# 创建Qwen模型
qwen_model = OpenAILanguageModel(
    model_id='qwen-plus',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key='sk-677242548d444be7ab177421fb87d5f6'
)

# 使用模板进行提取
print("正在调用Qwen模型进行信息抽取...")
result = extract_with_template(
    document=doc, 
    template="resume", 
    model=qwen_model
)
print("信息抽取完成！\n")

# 显示抽取结果
print("抽取到的实体信息:")
for entity in result.extractions:
    position_info = ""
    if entity.char_interval:
        start, end = entity.char_interval.start_pos, entity.char_interval.end_pos
        position_info = f" (位置: {start}-{end})"
    print(f"• {entity.extraction_class}: {entity.extraction_text}{position_info}")


# 生成HTML可视化
def generate_custom_html(result, original_text, output_file):
    """生成自定义的左右布局HTML可视化页面"""
    
    # 准备抽取结果数据
    extractions = []
    for idx, entity in enumerate(result.extractions):
        extraction_data = {
            'id': idx,
            'class': entity.extraction_class,
            'text': entity.extraction_text,
            'start': entity.char_interval.start_pos if entity.char_interval else -1,
            'end': entity.char_interval.end_pos if entity.char_interval else -1,
            'matched': False  # 标记是否在原文中找到匹配
        }
        extractions.append(extraction_data)
    
    # 智能匹配:如果langextract没有提供位置,尝试在原文中查找
    for ext in extractions:
        if ext['start'] < 0:  # 未提供位置信息
            # 尝试在原文中精确查找
            text_to_find = ext['text']
            pos = original_text.find(text_to_find)
            if pos >= 0:
                ext['start'] = pos
                ext['end'] = pos + len(text_to_find)
                ext['matched'] = True
                print(f"智能匹配成功: [{ext['class']}] {text_to_find} -> 位置 {pos}-{ext['end']}")
            else:
                print(f"警告: 未能在原文中找到 [{ext['class']}] {text_to_find}")
        else:
            ext['matched'] = True
    
    # 生成带高亮的原文HTML
    highlighted_text = original_text
    # 按位置倒序排序,避免位置偏移
    sorted_extractions = sorted([e for e in extractions if e['matched']], key=lambda x: x['start'], reverse=True)
    
    for ext in sorted_extractions:
        if ext['start'] >= 0:
            # 使用原始ID
            original_id = ext['id']
            # 为每个抽取的文本添加span标记
            before = highlighted_text[:ext['start']]
            matched = highlighted_text[ext['start']:ext['end']]
            after = highlighted_text[ext['end']:]
            highlighted_text = f"{before}<span class='highlight' data-id='{original_id}' data-class='{html.escape(ext['class'])}'>{html.escape(matched)}</span>{after}"
    
    # 读取HTML模板
    template_path = os.path.join(os.path.dirname(__file__), 'template.html')
    with open(template_path, 'r', encoding='utf-8') as f:
        html_template = f.read()
    
    # 准备注入的数据
    extractions_json = json.dumps(extractions, ensure_ascii=False)
    original_text_json = json.dumps(highlighted_text, ensure_ascii=False)
    
    # 替换模板中的占位符
    html_content = html_template.replace('{{EXTRACTIONS_DATA}}', extractions_json)
    html_content = html_content.replace('{{ORIGINAL_TEXT_DATA}}', original_text_json)
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)


# 生成可视化HTML
print("\n正在生成可视化文件...")
output_html = "resume_visualization.html"
generate_custom_html(result, original_text, output_html)
print(f"✅ 可视化结果已保存到 {output_html}")
print(f"请在浏览器中打开 {output_html} 查看结果")
