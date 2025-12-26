"""
Command-line interface for LangExtract
"""

import click
import os
import sys
import json
import yaml
import mimetypes
import PyPDF2
import reviewer.entity_extract as lx
from reviewer.entity_extract.core import data
from . import (
    load_document_from_url, 
    multi_pass_extract,
    MultiPassStrategies,
    ExtractionAnnotator,
)
from .templates import (
    TemplateManager,
    get_builtin_template,
    list_builtin_templates
)
from .template_builder import (
    TemplateWizard,
    TemplateBuilder,
    extract_with_template
)


@click.group()
@click.version_option(version='0.1.0')
def cli():
    """LangExtract CLI - Extract structured information from documents with grounding."""
    pass


@cli.command()
@click.option('--input', '-i', required=True, help='Input file path, URL, or "-" for stdin')
@click.option('--output', '-o', default='output.html', help='Output file (default: output.html)')
@click.option('--format', '-f', type=click.Choice(['html', 'jsonl', 'csv', 'gif']), 
              default='html', help='Output format')
@click.option('--model', '-m', default='qwen-plus', help='Model ID to use (default: Qwen Plus)')
@click.option('--prompt', '-p', help='Extraction prompt description')
@click.option('--template', help='Template ID or path to use for extraction')
@click.option('--examples', '-e', help='Path to examples file (JSON/YAML)')
@click.option('--api-key', '-k', envvar='QWEN_API_KEY', help='API key (or use QWEN_API_KEY env var)')
@click.option('--temperature', '-t', default=0.3, type=float, help='Generation temperature (0.0-2.0)')
@click.option('--fetch-urls', is_flag=True, help='Fetch content from URLs automatically')
@click.option('--resolve-refs', is_flag=True, help='Resolve references and relationships')
@click.option('--annotate', is_flag=True, help='Add quality annotations')
@click.option('--debug', is_flag=True, help='Enable debug output')
def extract(input, output, format, prompt, template, examples, api_key, temperature, fetch_urls, annotate, debug):
    """
    Extract information from a document.
    
    Examples:
    
        # Extract from a local file
        entity_extract extract -i document.pdf -p "Extract names and amounts" -o results.html
        
        # Extract using a template
        entity_extract extract -i invoice.pdf --template invoice -o results.html
        
        # Extract from a URL
        entity_extract extract -i https://example.com/doc.pdf -p "Extract dates" -o dates.jsonl
        
        # Extract from stdin
        cat document.txt | entity_extract extract -i - -p "Extract addresses" -f csv
    """
    # Validate that either prompt or template is provided
    if not prompt and not template:
        click.echo("Error: Either --prompt or --template must be provided", err=True)
        sys.exit(1)
    
    # Set API key if provided
    if api_key:
        os.environ['QWEN_API_KEY'] = api_key
    
    # Load examples if provided
    example_data = []
    if examples:
        try:
            with open(examples, 'r') as f:
                if examples.endswith('.yaml') or examples.endswith('.yml'):
                    examples_dict = yaml.safe_load(f)
                else:
                    examples_dict = json.load(f)
            
            # Convert to ExampleData objects
            for ex in examples_dict.get('examples', []):
                extractions = []
                for ext in ex.get('extractions', []):
                    extractions.append(data.Extraction(
                        extraction_class=ext['class'],
                        extraction_text=ext['text'],
                        attributes=ext.get('attributes')
                    ))
                example_data.append(data.ExampleData(
                    text=ex['text'],
                    extractions=extractions
                ))
        except Exception as e:
            click.echo(f"Error loading examples: {e}", err=True)
            sys.exit(1)
    
    # Load document
    try:
        if input == '-':
            # Read from stdin
            text = sys.stdin.read()
            doc = data.Document(text=text, document_id='stdin')
        elif input.startswith('http://') or input.startswith('https://'):
            # Load from URL
            click.echo(f"Loading from URL: {input}")
            doc = load_document_from_url(input)
        else:
            # Load from file
            if not os.path.exists(input):
                click.echo(f"Error: File not found: {input}", err=True)
                sys.exit(1)
            mime_type, _ = mimetypes.guess_type(input)

            if (mime_type == 'application/pdf') or input.lower().endswith('.pdf'):
                try:
                    with open(input, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text_parts = []
                        for page in reader.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text_parts.append(page_text)
                        text = '\n'.join(text_parts)
                    doc = data.Document(text=text, document_id=input)
                except Exception as e:
                    click.echo(f"Error reading PDF: {e}", err=True)
                    sys.exit(1)
            else:
                with open(input, 'r', encoding='utf-8') as f:
                    text = f.read()
                doc = data.Document(text=text, document_id=input)
    
    except Exception as e:
        click.echo(f"Error loading input: {e}", err=True)
        sys.exit(1)
    
    # Perform extraction with new features
    click.echo(f"Extracting with model (temperature: {temperature})")
    try:
        # Use template if provided
        if template:
            click.echo(f"Using template: {template}")
            result = extract_with_template(
                document=doc,
                template=template,
                temperature=temperature,
                api_key=api_key,
                debug=debug
            )
        # Use enhanced extract if URL fetching or temperature is specified
        elif fetch_urls or temperature != 0.3:
            result = lx.extract(
                text_or_documents=doc,
                prompt_description=prompt,
                examples=example_data,
                temperature=temperature,
                fetch_urls=fetch_urls,
                api_key=api_key,
                debug=debug
            )
        else:
            result = lx.extract(
                text_or_documents=doc,
                prompt_description=prompt,
                examples=example_data,
                api_key=api_key,
                debug=debug
            )

        # Apply annotations if requested
        if annotate and result.extractions:
            click.echo("Adding annotations...")
            annotator = ExtractionAnnotator()
            for extraction in result.extractions:
                annotator.annotate_extraction(extraction, result.text, result.extractions)
            
            # Add annotations to result
            result.annotations = annotator.export_annotations()
            click.echo(f"Added {len(annotator.annotations)} annotations")
            
    except Exception as e:
        click.echo(f"Extraction error: {e}", err=True)
        sys.exit(1)
    
    # Save output
    try:
        if format == 'html':
            # Save JSONL first
            root, _ = os.path.splitext(output)
            jsonl_path = root + '.jsonl'
            lx.io.save_annotated_documents([result], output_name=jsonl_path)
            # Generate HTML
            html = lx.visualize(jsonl_path)
            html_path = root + '.html'
            with open(html_path, 'w') as f:
                f.write(html)
            click.echo(f"✓ Created HTML visualization: {html_path}")
            
        elif format == 'jsonl':
            lx.io.save_annotated_documents([result], output_name=output)
            click.echo(f"✓ Saved JSONL: {output}")
    except Exception as e:
        click.echo(f"Error saving output: {e}", err=True)
        sys.exit(1)
    
    # Print summary
    click.echo(f"\nExtracted {len(result.extractions)} items:")
    by_class = {}
    for ext in result.extractions:
        by_class[ext.extraction_class] = by_class.get(ext.extraction_class, 0) + 1
    
    for class_name, count in sorted(by_class.items()):
        click.echo(f"  {class_name}: {count}")



@cli.command()
@click.option('--input', '-i', required=True, help='Input text file or "-" for stdin')
@click.option('--strategy', '-s', 
              type=click.Choice(['legal', 'medical', 'financial', 'custom']),
              default='custom', help='Multi-pass strategy')
@click.option('--passes', '-p', help='Custom passes config (JSON/YAML)')
@click.option('--output', '-o', default='multipass_results.html', help='Output file')
@click.option('--model', '-m', default='qwen-plus', help='Model ID (default: Qwen Plus)')
@click.option('--debug', is_flag=True, help='Enable debug output')
def multipass(input, strategy, passes, output, model, debug):
    """
    Perform multi-pass extraction for improved recall.
    
    Examples:
        # Use preset strategy
        entity_extract multipass -i judgment.txt -s legal -o results.html
        
        # Use custom passes
        entity_extract multipass -i document.txt -p passes.yaml -o results.html
    """
    # Load document
    if input == '-':
        text = sys.stdin.read()
    else:
        with open(input, 'r') as f:
            text = f.read()
    
    # Get passes configuration
    if strategy == 'custom' and passes:
        with open(passes, 'r') as f:
            if passes.endswith('.yaml') or passes.endswith('.yml'):
                passes_config = yaml.safe_load(f)['passes']
            else:
                passes_config = json.load(f)['passes']
    else:
        # Use preset strategy
        if strategy == 'legal':
            passes_config = MultiPassStrategies.legal_document_strategy()
        elif strategy == 'medical':
            passes_config = MultiPassStrategies.medical_record_strategy()
        elif strategy == 'financial':
            passes_config = MultiPassStrategies.financial_document_strategy()
        else:
            click.echo("Error: Must specify --strategy or --passes", err=True)
            sys.exit(1)
    
    # Perform multi-pass extraction
    try:
        result = multi_pass_extract(
            text,
            passes_config,
            debug=debug
        )
        
        # Save results
        root, ext = os.path.splitext(output)
        if ext == '.html' or ext == '':
            jsonl_path = root + '.jsonl'
            lx.io.save_annotated_documents([result], output_name=jsonl_path)
            html = lx.visualize(jsonl_path)
            html_path = root + '.html'
            with open(html_path, 'w') as f:
                f.write(html)
            output_path = html_path
        else:
            lx.io.save_annotated_documents([result], output_name=output)
            output_path = output

        click.echo(f"✓ Saved results: {output_path}")
        click.echo(f"Total extractions: {len(result.extractions)}")
        
    except Exception as e:
        click.echo(f"Multi-pass error: {e}", err=True)
        sys.exit(1)



@cli.command()
def examples():
    """Show example usage and create example files."""
    click.echo("""
LangExtract Examples
===================

1. Basic extraction from PDF:
   entity_extract extract -i document.pdf -p "Extract names and dates" -o results.html

2. Extract using a template:
   entity_extract extract -i invoice.pdf --template invoice -o results.html
   entity_extract extract -i resume.pdf --template resume -o candidates.jsonl

3. Extract from URL with automatic fetching:
   entity_extract extract -i https://example.com/doc.pdf -p "Extract amounts" --fetch-urls -o amounts.jsonl

4. Extract with custom temperature:
   entity_extract extract -i doc.txt -p "Extract creative insights" -t 0.7 -o insights.html

5. Extract with reference resolution:
   entity_extract extract -i legal.txt -p "Extract entities" --resolve-refs -o entities.html

6. Extract with quality annotations:
   entity_extract extract -i report.pdf -p "Extract findings" --annotate -o annotated.jsonl

7. Batch process CSV:
   entity_extract batch -c documents.csv -t content -p "Extract entities" -o results.csv

8. Multi-pass extraction:
   entity_extract multipass -i legal.txt -s legal -o legal_results.html

9. Template management:
   entity_extract template list                                    # List all templates
   entity_extract template list -v                                 # List with details
   entity_extract template show invoice                            # Show template details
   entity_extract template create -i                               # Interactive wizard
   entity_extract template create -e examples.yaml -n "My Template"  # From examples
   entity_extract template export invoice -o invoice_template.yaml   # Export template
   entity_extract template import custom_template.yaml             # Import template
   entity_extract template delete my_template                      # Delete template

10. Create visualization:
    entity_extract visualize -j results.jsonl -o viz.html
    entity_extract visualize -j results.jsonl -o viz.html -t dark
    entity_extract visualize -j results.jsonl -f gif -o animation.gif

11. List available providers:
    entity_extract providers

Example files have been created:
- example_prompts.yaml
- example_passes.yaml
- example_template.yaml
""")
    
    # Create example files
    example_prompts = {
        'examples': [
            {
                'text': 'Invoice #12345 dated January 15, 2025',
                'extractions': [
                    {'class': 'invoice_number', 'text': '#12345'},
                    {'class': 'date', 'text': 'January 15, 2025'}
                ]
            }
        ]
    }
    
    example_passes = {
        'passes': [
            {
                'prompt_description': 'Extract all person names',
                'focus_on': ['person', 'name']
            },
            {
                'prompt_description': 'Extract all monetary amounts',
                'focus_on': ['amount', 'money', 'price']
            }
        ]
    }
    
    with open('example_prompts.yaml', 'w') as f:
        yaml.dump(example_prompts, f)
    
    with open('example_passes.yaml', 'w') as f:
        yaml.dump(example_passes, f)
    
    # Create example template file
    example_template = {
        'template_id': 'custom_contract',
        'name': 'Contract Analysis Template',
        'description': 'Extract key information from contracts',
        'document_type': 'legal',
        'fields': [
            {
                'name': 'party1',
                'extraction_class': 'organization',
                'description': 'First party in the contract',
                'required': True,
                'examples': ['ABC Corporation', 'John Smith Ltd.']
            },
            {
                'name': 'party2',
                'extraction_class': 'organization',
                'description': 'Second party in the contract',
                'required': True,
                'examples': ['XYZ Inc.', 'Jane Doe Enterprises']
            },
            {
                'name': 'effective_date',
                'extraction_class': 'date',
                'description': 'Contract effective date',
                'required': True,
                'validation_pattern': r'\d{4}-\d{2}-\d{2}'
            },
            {
                'name': 'term_length',
                'extraction_class': 'text',
                'description': 'Duration of the contract',
                'required': False,
                'examples': ['12 months', '3 years', 'perpetual']
            },
            {
                'name': 'total_value',
                'extraction_class': 'amount',
                'description': 'Total contract value',
                'required': False,
                'examples': ['$100,000', '€50,000']
            }
        ],
        'preferred_model': 'qwen-plus',
        'temperature': 0.2,
        'tags': ['legal', 'contract', 'analysis'],
        'author': 'example',
        'version': '1.0.0'
    }
    
    with open('example_template.yaml', 'w') as f:
        yaml.dump(example_template, f, default_flow_style=False)


@cli.group()
def template():
    """Manage extraction templates."""
    pass


@template.command('list')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed template information')
def template_list(verbose):
    """
    List available templates.
    
    Examples:
        entity_extract template list
        entity_extract template list -v
    """
    click.echo("Built-in Templates")
    click.echo("==================\n")
    
    builtin = list_builtin_templates()
    for template_id in builtin:
        template = get_builtin_template(template_id)
        click.echo(f"• {template_id}")
        if verbose:
            click.echo(f"  Name: {template.name}")
            click.echo(f"  Description: {template.description}")
            click.echo(f"  Fields: {', '.join(f.name for f in template.fields)}")
            click.echo()
    
    click.echo("\nCustom Templates")
    click.echo("================\n")
    
    manager = TemplateManager()
    custom = manager.list_templates()
    
    if custom:
        for template_id in custom:
            template = manager.load_template(template_id)
            if template:
                click.echo(f"• {template_id}")
                if verbose:
                    click.echo(f"  Name: {template.name}")
                    click.echo(f"  Description: {template.description}")
                    click.echo(f"  Fields: {', '.join(f.name for f in template.fields)}")
                    click.echo()
    else:
        click.echo("No custom templates found.")
    
    click.echo("\nTo use a template:")
    click.echo("entity_extract extract -i document.pdf --template <template_id>")


@template.command('show')
@click.argument('template_id')
@click.option('--format', '-f', type=click.Choice(['yaml', 'json']), default='yaml', 
              help='Output format')
def template_show(template_id, format):
    """
    Show details of a specific template.
    
    Examples:
        entity_extract template show invoice
        entity_extract template show invoice -f json
    """
    # Try built-in first
    template = get_builtin_template(template_id)
    
    # Then try custom
    if not template:
        manager = TemplateManager()
        template = manager.load_template(template_id)
    
    if not template:
        click.echo(f"Template not found: {template_id}", err=True)
        sys.exit(1)
    
    # Convert to dict
    template_dict = {
        'template_id': template.template_id,
        'name': template.name,
        'description': template.description,
        'document_type': template.document_type.value,
        'fields': [
            {
                'name': f.name,
                'extraction_class': f.extraction_class,
                'description': f.description,
                'required': f.required,
                'examples': f.examples,
                'validation_pattern': f.validation_pattern
            }
            for f in template.fields
        ],
        'preferred_model': template.preferred_model,
        'temperature': template.temperature,
        'tags': template.tags,
        'author': template.author,
        'version': template.version
    }
    
    if format == 'json':
        click.echo(json.dumps(template_dict, indent=2))
    else:
        click.echo(yaml.dump(template_dict, default_flow_style=False))


@template.command('create')
@click.option('--interactive', '-i', is_flag=True, help='Use interactive wizard')
@click.option('--from-examples', '-e', help='Create from examples file (JSON/YAML)')
@click.option('--name', '-n', help='Template name')
@click.option('--output', '-o', help='Save to file instead of template directory')
def template_create(interactive, from_examples, name, output):
    """
    Create a new extraction template.
    
    Examples:
        # Interactive wizard
        entity_extract template create -i
        
        # From examples
        entity_extract template create -e examples.yaml -n "My Template"
        
        # Save to file
        entity_extract template create -i -o my_template.yaml
    """
    if interactive:
        # Run interactive wizard
        template = TemplateWizard.run()
        
    elif from_examples:
        if not name:
            click.echo("Error: --name is required when using --from-examples", err=True)
            sys.exit(1)
        
        # Load examples
        try:
            with open(from_examples, 'r') as f:
                if from_examples.endswith('.yaml') or from_examples.endswith('.yml'):
                    examples_data = yaml.safe_load(f)
                else:
                    examples_data = json.load(f)
            
            documents = examples_data.get('documents', [])
            extractions = examples_data.get('extractions', [])
            
            if not documents or not extractions:
                click.echo("Error: Examples file must contain 'documents' and 'extractions'", err=True)
                sys.exit(1)
            
            # Build template from examples
            builder = TemplateBuilder()
            template = builder.build_from_examples(
                example_documents=documents,
                expected_extractions=extractions,
                template_name=name
            )
            
            click.echo(f"✓ Created template from {len(documents)} examples")
            
        except Exception as e:
            click.echo(f"Error creating template: {e}", err=True)
            sys.exit(1)
    
    else:
        click.echo("Error: Use --interactive or --from-examples", err=True)
        sys.exit(1)
    
    # Save template
    if output:
        # Save to file
        template_dict = template.to_dict()
        if output.endswith('.json'):
            with open(output, 'w') as f:
                json.dump(template_dict, f, indent=2)
        else:
            with open(output, 'w') as f:
                yaml.dump(template_dict, f, default_flow_style=False)
        click.echo(f"✓ Saved template to: {output}")
    else:
        # Save to template directory
        manager = TemplateManager()
        manager.save_template(template)
        click.echo(f"✓ Saved template: {template.template_id}")


@template.command('delete')
@click.argument('template_id')
@click.confirmation_option(prompt='Are you sure you want to delete this template?')
def template_delete(template_id):
    """
    Delete a custom template.
    
    Example:
        entity_extract template delete my_template
    """
    manager = TemplateManager()
    
    if manager.delete_template(template_id):
        click.echo(f"✓ Deleted template: {template_id}")
    else:
        click.echo(f"Error: Template not found or is built-in: {template_id}", err=True)
        sys.exit(1)


@template.command('export')
@click.argument('template_id')
@click.option('--output', '-o', required=True, help='Output file path')
@click.option('--format', '-f', type=click.Choice(['yaml', 'json']), default='yaml',
              help='Output format')
def template_export(template_id, output, format):
    """
    Export a template to a file.
    
    Examples:
        entity_extract template export invoice -o invoice_template.yaml
        entity_extract template export invoice -o invoice_template.json -f json
    """
    # Try built-in first
    template = get_builtin_template(template_id)
    
    # Then try custom
    if not template:
        manager = TemplateManager()
        template = manager.load_template(template_id)
    
    if not template:
        click.echo(f"Template not found: {template_id}", err=True)
        sys.exit(1)
    
    # Export
    template_dict = template.to_dict()
    
    if format == 'json':
        with open(output, 'w') as f:
            json.dump(template_dict, f, indent=2)
    else:
        with open(output, 'w') as f:
            yaml.dump(template_dict, f, default_flow_style=False)
    
    click.echo(f"✓ Exported template to: {output}")


@template.command('import')
@click.argument('file_path')
@click.option('--template-id', help='Override template ID')
def template_import(file_path, template_id):
    """
    Import a template from a file.
    
    Examples:
        entity_extract template import template.yaml
        entity_extract template import template.json --template-id custom_invoice
    """
    try:
        with open(file_path, 'r') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                template_dict = yaml.safe_load(f)
            else:
                template_dict = json.load(f)
        
        # Override template ID if provided
        if template_id:
            template_dict['template_id'] = template_id
        
        # Create template from dict
        from .templates import ExtractionTemplate, ExtractionField, DocumentType
        
        fields = []
        for field_dict in template_dict.get('fields', []):
            fields.append(ExtractionField(
                name=field_dict['name'],
                extraction_class=field_dict['extraction_class'],
                description=field_dict.get('description', ''),
                required=field_dict.get('required', True),
                examples=field_dict.get('examples', []),
                validation_pattern=field_dict.get('validation_pattern')
            ))
        
        template = ExtractionTemplate(
            template_id=template_dict['template_id'],
            name=template_dict['name'],
            description=template_dict.get('description', ''),
            document_type=DocumentType(template_dict.get('document_type', 'custom')),
            fields=fields,
            preferred_model=template_dict.get('preferred_model', 'qwen-plus'),
            temperature=template_dict.get('temperature', 0.3),
            tags=template_dict.get('tags', []),
            author=template_dict.get('author', 'imported'),
            version=template_dict.get('version', '1.0.0')
        )
        
        # Save template
        manager = TemplateManager()
        manager.save_template(template)
        
        click.echo(f"✓ Imported template: {template.template_id}")
        
    except Exception as e:
        click.echo(f"Error importing template: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()