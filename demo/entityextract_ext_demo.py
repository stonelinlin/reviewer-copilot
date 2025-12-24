# Method 1: Using Templates (Recommended for consistency)
import PyPDF2
from reviewer.entity_extract import data
from reviewer.entityextract_ext.template_builder import extract_with_template

input = 'resume.pdf'
with open(input, 'rb') as f:
    reader = PyPDF2.PdfReader(f)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    text = '\n'.join(text_parts)
doc = data.Document(text=text, document_id=input)


# Use built-in template
result = extract_with_template(
    document=doc,
    template="resume"  # or "resume", "legal_document", "medical_record"
)
