from reviewer.ext.url_loader import load_document_from_url
from reviewer.ext.multi_pass import multi_pass_extract, MultiPassStrategies
from reviewer.ext.config import LangExtractConfig, create_example_config, get_config, configure
from reviewer.ext.templates import (ExtractionTemplate, ExtractionField, DocumentType, TemplateManager, get_builtin_template)
from reviewer.ext.quality import (QualityAnnotation, AnnotationType, ConfidenceLevel, QualityScorer, ExtractionVerifier,
                                  ExtractionAnnotator)


__all__ = [
    # Original features
    'load_document_from_url', 'multi_pass_extract',
    'MultiPassStrategies', 'LangExtractConfig', 'create_example_config', 'get_config', 'configure',

    # Template system
    'ExtractionTemplate', 'ExtractionField', 'DocumentType', 'TemplateManager', 'get_builtin_template',

    # Annotation
    'QualityAnnotation', 'AnnotationType', 'ConfidenceLevel', 'QualityScorer', 'ExtractionVerifier', 'ExtractionAnnotator', ]

__version__ = '0.1.0'
