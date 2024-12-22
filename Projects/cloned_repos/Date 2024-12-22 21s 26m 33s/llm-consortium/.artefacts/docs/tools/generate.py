"""
Documentation generation tool for LLM Karpathy Consortium.
Generates documentation from source code and templates.
"""
import ast
import inspect
import re
from pathlib import Path
from typing import Dict, List, Optional
import click
from jinja2 import Environment, FileSystemLoader
import black
import yaml

class DocGenerator:
    def __init__(self, src_path: str = "llm_consortium", docs_path: str = "docs"):
        self.src_path = Path(src_path)
        self.docs_path = Path(docs_path)
        self.env = Environment(loader=FileSystemLoader("docs/templates"))
        self.parsed_modules: Dict[str, ast.Module] = {}

    def generate_all(self) -> None:
        """Generate all documentation."""
        self.generate_api_reference()
        self.generate_examples()
        self.update_readme()
        self.generate_module_docs()

    def parse_module(self, path: Path) -> ast.Module:
        """Parse Python module and extract documentation."""
        content = path.read_text()
        return ast.parse(content)

    def extract_docstring(self, node: ast.AST) -> Optional[str]:
        """Extract and clean docstring from AST node."""
        docstring = ast.get_docstring(node)
        if docstring:
            return inspect.cleandoc(docstring)
        return None

    def generate_api_reference(self) -> None:
        """Generate API reference documentation."""
        template = self.env.get_template("api_reference.md.j2")
        api_docs = []

        for py_file in self.src_path.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue

            module = self.parse_module(py_file)
            module_docs = {
                "name": py_file.stem,
                "docstring": self.extract_docstring(module),
                "classes": [],
                "functions": []
            }

            for node in ast.walk(module):
                if isinstance(node, ast.ClassDef):
                    class_doc = {
                        "name": node.name,
                        "docstring": self.extract_docstring(node),
                        "methods": []
                    }
                    for method in node.body:
                        if isinstance(method, ast.FunctionDef):
                            class_doc["methods"].append({
                                "name": method.name,
                                "docstring": self.extract_docstring(method),
                                "signature": self._get_signature(method)
                            })
                    module_docs["classes"].append(class_doc)
                elif isinstance(node, ast.FunctionDef):
                    module_docs["functions"].append({
                        "name": node.name,
                        "docstring": self.extract_docstring(node),
                        "signature": self._get_signature(node)
                    })

            api_docs.append(module_docs)

        output = template.render(modules=api_docs)
        (self.docs_path / "API_REFERENCE.md").write_text(output)

    def _get_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature."""
        args = []
        for arg in node.args.args:
            arg_name = arg.arg
            arg_type = ""
            if arg.annotation:
                arg_type = ": " + ast.unparse(arg.annotation)
            args.append(f"{arg_name}{arg_type}")
        
        returns = ""
        if node.returns:
            returns = f" -> {ast.unparse(node.returns)}"
        
        return f"def {node.name}({', '.join(args)}){returns}"

    def generate_examples(self) -> None:
        """Generate example documentation."""
        template = self.env.get_template("examples.md.j2")
        examples = []

        for example_file in (self.src_path.parent / "examples").glob("*.py"):
            module = self.parse_module(example_file)
            example_doc = {
                "name": example_file.stem,
                "docstring": self.extract_docstring(module),
                "code": black.format_str(
                    example_file.read_text(),
                    mode=black.FileMode()
                )
            }
            examples.append(example_doc)

        output = template.render(examples=examples)
        (self.docs_path / "EXAMPLES.md").write_text(output)

    def update_readme(self) -> None:
        """Update README with current information."""
        template = self.env.get_template("readme.md.j2")
        
        # Extract version
        init_file = self.src_path / "__init__.py"
        version_match = re.search(
            r'__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
            init_file.read_text()
        )
        version = version_match.group(1) if version_match else "Unknown"

        # Get example count
        example_count = len(list((self.src_path.parent / "examples").glob("*.py")))

        output = template.render(
            version=version,
            example_count=example_count
        )
        (self.src_path.parent / "README.md").write_text(output)

    def generate_module_docs(self) -> None:
        """Generate individual module documentation."""
        template = self.env.get_template("module.md.j2")

        for py_file in self.src_path.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue

            module = self.parse_module(py_file)
            module_doc = {
                "name": py_file.stem,
                "docstring": self.extract_docstring(module),
                "classes": [],
                "functions": [],
                "examples": self._extract_examples(module)
            }

            output = template.render(module=module_doc)
            doc_path = self.docs_path / "modules" / f"{py_file.stem}.md"
            doc_path.parent.mkdir(exist_ok=True)
            doc_path.write_text(output)

    def _extract_examples(self, module: ast.Module) -> List[Dict]:
        """Extract examples from docstrings."""
        examples = []
        for node in ast.walk(module):
            docstring = self.extract_docstring(node)
            if docstring and "Examples:" in docstring:
                example_text = docstring.split("Examples:")[1].strip()
                examples.append({
                    "context": node.name if isinstance(node, (ast.ClassDef, ast.FunctionDef)) else "module",
                    "code": example_text
                })
        return examples

@click.command()
@click.option('--src-path', default='llm_consortium', help='Source code path')
@click.option('--docs-path', default='docs', help='Documentation path')
@click.option('--check', is_flag=True, help='Check if docs are up to date')
def main(src_path: str, docs_path: str, check: bool) -> None:
    """Generate documentation from source code."""
    generator = DocGenerator(src_path, docs_path)
    generator.generate_all()

if __name__ == "__main__":
    main()
