# Original features
from reviewer.entityextract_ext.url_loader import load_document_from_url
from reviewer.entityextract_ext.csv_loader import load_documents_from_csv
from reviewer.entityextract_ext.gif_export import export_to_gif, create_simple_gif
from reviewer.entityextract_ext.multi_pass import multi_pass_extract, MultiPassStrategies
from reviewer.entityextract_ext.config import LangExtractConfig, create_example_config, get_config, configure
from reviewer.entityextract_ext.custom_visualization import (HTMLTemplate, MinimalTemplate, DarkModeTemplate, CompactTemplate,
                                   visualize_with_template, create_custom_template, load_template_from_file)
from reviewer.entityextract_ext.templates import (ExtractionTemplate, ExtractionField, DocumentType, TemplateManager, get_builtin_template)

# New core features
from reviewer.entityextract_ext.providers import BaseProvider, ProviderCapabilities, GeminiProvider
from reviewer.entityextract_ext.registry import (register, get_provider_class, create_provider, list_providers, list_patterns)
from reviewer.entityextract_ext.factory import ProviderFactory, ExtractorFactory, PipelineFactory
from reviewer.entityextract_ext.extraction import extract, extract_with_provider, fetch_url_content
from reviewer.entityextract_ext.resolver import (ReferenceResolver, RelationshipResolver, Reference, Relationship)
from reviewer.entityextract_ext.quality import (Annotation, AnnotationType, ConfidenceLevel, QualityScorer, ExtractionVerifier,
                         ExtractionAnnotator)


__all__ = [# Original features
    'load_document_from_url', 'load_documents_from_csv', 'export_to_gif', 'create_simple_gif', 'multi_pass_extract',
    'MultiPassStrategies', 'LangExtractConfig', 'create_example_config', 'get_config', 'configure', 'HTMLTemplate',
    'MinimalTemplate', 'DarkModeTemplate', 'CompactTemplate', 'visualize_with_template', 'create_custom_template',
    'load_template_from_file',

    # Template system
    'ExtractionTemplate', 'ExtractionField', 'DocumentType', 'TemplateManager', 'get_builtin_template',

    # Provider system
    'BaseProvider', 'ProviderCapabilities', 'GeminiProvider',

    # Registry system
    'register', 'get_provider_class', 'create_provider', 'list_providers', 'list_patterns',

    # Factory pattern
    'ProviderFactory', 'ExtractorFactory', 'PipelineFactory',

    # Enhanced extraction
    'extract', 'extract_with_provider', 'fetch_url_content',

    # Resolver
    'ReferenceResolver', 'RelationshipResolver', 'Reference', 'Relationship',

    # Annotation
    'Annotation', 'AnnotationType', 'ConfidenceLevel', 'QualityScorer', 'ExtractionVerifier', 'ExtractionAnnotator', ]

__version__ = '0.1.0'  # Bumped version for new features