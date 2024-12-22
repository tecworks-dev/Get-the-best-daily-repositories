"""
Test suite for documentation validation and consistency.
"""
import pytest
from pathlib import Path
from validate import DocValidator
from search import DocSearcher

@pytest.fixture
def validator():
    return DocValidator()

@pytest.fixture
def searcher():
    return DocSearcher()

def test_required_files_exist(validator):
    """Test that all required documentation files exist."""
    validator.check_file_structure()
    assert len(validator.errors) == 0, "Missing required files"

def test_valid_links(validator):
    """Test that all internal links are valid."""
    validator.check_broken_links()
    assert len(validator.errors) == 0, "Found broken links"

def test_style_consistency(validator):
    """Test documentation style consistency."""
    validator.check_style_consistency()
    assert len(validator.warnings) == 0, "Style inconsistencies found"

def test_code_blocks(validator):
    """Test code block formatting."""
    validator.check_code_blocks()
    assert len(validator.warnings) == 0, "Invalid code blocks found"

def test_search_functionality(searcher):
    """Test documentation search functionality."""
    results = searcher.search("installation")
    assert len(results) > 0, "Search returned no results"

def test_required_sections(validator):
    """Test presence of required documentation sections."""
    validator.check_required_sections()
    assert len(validator.errors) == 0, "Missing required sections"

def test_frontmatter_validity(validator):
    """Test YAML frontmatter validity."""
    validator.validate_frontmatter()
    assert len(validator.errors) == 0, "Invalid frontmatter found"

def test_search_relevance(searcher):
    """Test search result relevance."""
    results = searcher.search("API reference")
    assert any('API' in r['file'] for r in results), "Search results not relevant"

def test_style_guide_compliance():
    """Test compliance with style guide."""
    style_guide = Path("docs/DOCUMENTATION_STYLE.md")
    assert style_guide.exists(), "Style guide not found"
    
    content = style_guide.read_text()
    required_sections = ['Formatting Standards', 'Content Guidelines']
    for section in required_sections:
        assert section in content, f"Missing section: {section}"
