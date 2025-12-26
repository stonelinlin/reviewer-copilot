
"""Main extraction API for LangExtract."""

from __future__ import annotations

from collections.abc import Iterable
import typing
from typing import cast
import warnings

from reviewer.entity_extract import annotation
from reviewer.entity_extract import io
from reviewer.entity_extract import prompting
from reviewer.entity_extract import resolver
from reviewer.entity_extract.core import data
from reviewer.entity_extract.core import format_handler as fh
from reviewer.entity_extract.core import tokenizer as tokenizer_lib


def extract(
    text_or_documents: typing.Any,
    prompt_description: str | None = None,
    examples: typing.Sequence[typing.Any] | None = None,
    format_type: typing.Any = None,
    max_char_buffer: int = 1000,
    temperature: float | None = None,
    batch_length: int = 10,
    max_workers: int = 10,
    additional_context: str | None = None,
    resolver_params: dict | None = None,
    debug: bool = False,
    extraction_passes: int = 1,
    model: typing.Any = None,
    *,
    fetch_urls: bool = True,
    show_progress: bool = True,
    tokenizer: tokenizer_lib.Tokenizer | None = None,
) -> list[data.AnnotatedDocument] | data.AnnotatedDocument:
  if not examples:
    raise ValueError(
        "Examples are required for reliable extraction. Please provide at least"
        " one ExampleData object with sample extractions."
    )

  if debug:
    from reviewer.entity_extract.core import debug_utils
    debug_utils.configure_debug_logging()

  if format_type is None:
    format_type = data.FormatType.JSON

  if max_workers is not None and batch_length < max_workers:
    warnings.warn(
        f"batch_length ({batch_length}) < max_workers ({max_workers}). "
        f"Only {batch_length} workers will be used. "
        "Set batch_length >= max_workers for optimal parallelization.",
        UserWarning,
    )

  if (
      fetch_urls
      and isinstance(text_or_documents, str)
      and io.is_url(text_or_documents)
  ):
    text_or_documents = io.download_text_from_url(text_or_documents)

  # 文本预处理
  if isinstance(text_or_documents, str):
    from reviewer.entity_extract.text_utils import clean_text
    text_or_documents = clean_text(text_or_documents)
  
  # 使用 Qwen 模型（固定）
  if model is None:
    from reviewer.entity_extract.factory import get_model
    model = get_model(temperature=temperature if temperature else 0.3)
  
  # 使用 UnicodeTokenizer（中文优化）
  if tokenizer is None:
    from reviewer.entity_extract.core import tokenizer as tokenizer_lib
    tokenizer = tokenizer_lib.UnicodeTokenizer()
  
  # 构建提示模板
  prompt_template = prompting.PromptTemplateStructured(
      description=prompt_description,
  )
  if examples:
    prompt_template.examples.extend(examples)
  
  # 设置语言模型
  language_model = model

  # 解析器参数
  resolver_params = resolver_params or {}
  format_handler, remaining_params = fh.FormatHandler.from_resolver_params(
      resolver_params=resolver_params,
      base_format_type=format_type,
      base_use_fences=language_model.requires_fence_output,
      base_attribute_suffix=data.ATTRIBUTE_SUFFIX,
      base_use_wrapper=True,
      base_wrapper_key=data.EXTRACTIONS_KEY,
  )

  # Pull alignment settings from normalized params
  alignment_kwargs = {}
  for key in resolver.ALIGNMENT_PARAM_KEYS:
    val = remaining_params.pop(key, None)
    if val is not None:
      alignment_kwargs[key] = val

  effective_params = {"format_handler": format_handler, **remaining_params}

  try:
    res = resolver.Resolver(**effective_params)
  except TypeError as e:
    msg = str(e)
    if (
        "unexpected keyword argument" in msg
        or "got an unexpected keyword argument" in msg
    ):
      raise TypeError(
          f"Unknown key in resolver_params; check spelling: {e}"
      ) from e
    raise

  annotator = annotation.Annotator(
      language_model=language_model,
      prompt_template=prompt_template,
      format_handler=format_handler,
  )

  if isinstance(text_or_documents, str):
    result = annotator.annotate_text(
        text=text_or_documents,
        resolver=res,
        max_char_buffer=max_char_buffer,
        batch_length=batch_length,
        additional_context=additional_context,
        debug=debug,
        extraction_passes=extraction_passes,
        show_progress=show_progress,
        max_workers=max_workers,
        tokenizer=tokenizer,
        **alignment_kwargs,
    )
    return result
  else:
    documents = cast(Iterable[data.Document], text_or_documents)
    result = annotator.annotate_documents(
        documents=documents,
        resolver=res,
        max_char_buffer=max_char_buffer,
        batch_length=batch_length,
        debug=debug,
        extraction_passes=extraction_passes,
        show_progress=show_progress,
        max_workers=max_workers,
        tokenizer=tokenizer,
        **alignment_kwargs,
    )
    return list(result)
