"""
Documentation validation tool for LLM Karpathy Consortium.
Checks for common documentation issues and enforces style guidelines.
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Set, Optional
import yaml
import click
from rich.console import Console
from rich.table import Table

class DocValidator:
    def __init__(self, docs_path: str = "docs"):
        self.docs_path = Path(docs_path)
        self.console = Console()
        self.errors: List[Dict] = []
        self.warnings: List[Dict] = []

    def validate_all(self) -> bool:
        """Run all validation checks."""
        self.check_broken_links()
        self.check_style_consistency()
        self.check_code_blocks()
        self.check_required_sections()
        self.check_file_structure()
        self.validate_frontmatter()
        
        return len(self.errors) == 0

    def check_broken_links(self) -> None:
        """Check for broken internal links."""
        for file_path in self.docs_path.rglob("*.md"):
            content = file_path.read_text()
            links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
            
            for text, link in links:
                if not link.startswith(('http://', 'https://', '#')):
                    target_path = file_path.parent / link
                    if not target_path.exists():
                        self.errors.append({
                            'file': str(file_path),
                            'type': 'broken_link',
                            'message': f'Broken link to {link}'
                        })

    def check_style_consistency(self) -> None:
        """Check for style guide compliance."""
        style_checks = {
            'heading_spacing': r'#[^#\s]',  # Should have space after #
            'code_block_language': r'```\s*$',  # Should specify language
            'trailing_whitespace': r'[ \t]+$',  # No trailing whitespace
        }

        for file_path in self.docs_path.rglob("*.md"):
            content = file_path.read_text()
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                for check_name, pattern in style_checks.items():
                    if re.search(pattern, line):
                        self.warnings.append({
                            'file': str(file_path),
                            'line': i,
                            'type': check_name,
                            'message': f'Style issue: {check_name}'
                        })

    def check_code_blocks(self) -> None:
        """Validate code blocks."""
        for file_path in self.docs_path.rglob("*.md"):
            content = file_path.read_text()
            code_blocks = re.finditer(r'```(\w+)?\n(.*?)```', content, re.DOTALL)
            
            for block in code_blocks:
                if not block.group(1):  # No language specified
                    self.warnings.append({
                        'file': str(file_path),
                        'type': 'code_block',
                        'message': 'Code block missing language specification'
                    })

    def check_required_sections(self) -> None:
        """Check for required documentation sections."""
        required_sections = {
            'README.md': ['Installation', 'Usage', 'Contributing'],
            'CONTRIBUTING.md': ['Code of Conduct', 'Pull Request Process'],
            'API.md': ['Reference', 'Examples']
        }

        for file_name, sections in required_sections.items():
            file_path = self.docs_path.parent / file_name
            if file_path.exists():
                content = file_path.read_text()
                for section in sections:
                    if not re.search(f'^#+\s+{section}', content, re.MULTILINE):
                        self.errors.append({
                            'file': str(file_path),
                            'type': 'missing_section',
                            'message': f'Missing required section: {section}'
                        })

    def check_file_structure(self) -> None:
        """Validate documentation file structure."""
        required_files = {
            'README.md',
            'CONTRIBUTING.md',
            'API.md',
            'CHANGELOG.md'
        }
        
        found_files = {f.name for f in self.docs_path.parent.glob('*.md')}
        missing_files = required_files - found_files
        
        for file_name in missing_files:
            self.errors.append({
                'type': 'missing_file',
                'message': f'Missing required file: {file_name}'
            })

    def validate_frontmatter(self) -> None:
        """Validate YAML frontmatter in documentation files."""
        for file_path in self.docs_path.rglob("*.md"):
            content = file_path.read_text()
            if content.startswith('---'):
                try:
                    end = content.index('---', 3)
                    frontmatter = content[3:end]
                    yaml.safe_load(frontmatter)
                except Exception as e:
                    self.errors.append({
                        'file': str(file_path),
                        'type': 'frontmatter',
                        'message': f'Invalid frontmatter: {str(e)}'
                    })

    def print_report(self) -> None:
        """Print validation report."""
        error_table = Table(title="Validation Errors")
        error_table.add_column("File")
        error_table.add_column("Type")
        error_table.add_column("Message")
        error_table.add_column("Line", justify="right")

        for error in self.errors:
            error_table.add_row(
                error.get('file', 'N/A'),
                error['type'],
                error['message'],
                str(error.get('line', ''))
            )

        warning_table = Table(title="Validation Warnings")
        warning_table.add_column("File")
        warning_table.add_column("Type")
        warning_table.add_column("Message")
        warning_table.add_column("Line", justify="right")

        for warning in self.warnings:
            warning_table.add_row(
                warning.get('file', 'N/A'),
                warning['type'],
                warning['message'],
                str(warning.get('line', ''))
            )

        self.console.print(error_table)
        self.console.print(warning_table)

@click.command()
@click.option('--docs-path', default='docs', help='Path to documentation directory')
@click.option('--fix', is_flag=True, help='Attempt to fix common issues')
@click.option('--report', type=click.Path(), help='Save report to file')
def main(docs_path: str, fix: bool, report: Optional[str]) -> None:
    """Validate documentation files."""
    validator = DocValidator(docs_path)
    is_valid = validator.validate_all()
    
    if report:
        report_data = {
            'errors': validator.errors,
            'warnings': validator.warnings
        }
        with open(report, 'w') as f:
            yaml.dump(report_data, f)
    else:
        validator.print_report()
    
    if not is_valid:
        click.exit(1)

if __name__ == "__main__":
    main()
