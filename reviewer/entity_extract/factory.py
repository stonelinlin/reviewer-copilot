"""
获取 Qwen 模型（简化版工厂函数）
"""
from reviewer.entity_extract.openai import OpenAILanguageModel

QWEN_API_KEY = 'sk-677242548d444be7ab177421fb87d5f6'
QWEN_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
QWEN_MODEL_ID = 'qwen-plus'


def get_model(temperature: float = 0.3):
    """获取 Qwen 模型（唯一选择）"""
    return OpenAILanguageModel(
        model_id=QWEN_MODEL_ID,
        base_url=QWEN_BASE_URL,
        api_key=QWEN_API_KEY,
        temperature=temperature,
    )
