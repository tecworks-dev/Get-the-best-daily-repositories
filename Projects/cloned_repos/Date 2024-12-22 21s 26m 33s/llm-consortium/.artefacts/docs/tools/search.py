"""
Documentation search tool for LLM Karpathy Consortium.
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Optional
import argparse
import json

class DocSearcher:
    def __init__(self, docs_path: str = "docs"):
        self.docs_path = Path(docs_path)
        self.index: Dict[str, Dict] = {}
        self._build_index()

    def _build_index(self) -> None:
        """Build search index from documentation files."""
        for file_path in self.docs_path.rglob("*.md"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                self.index[str(file_path)] = {
                    "content": content,
                    "headings": self._extract_headings(content),
                    "keywords": self._extract_keywords(content)
                }

    def _extract_headings(self, content: str) -> List[str]:
        """Extract markdown headings from content."""
        return re.findall(r"^#+\s+(.+)$", content, re.MULTILINE)

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content."""
        # Extract words from headings and code blocks
        keywords = set()
        # Add heading words
        for heading in self._extract_headings(content):
            keywords.update(word.lower() for word in re.findall(r"\w+", heading))
        # Add words from code blocks
        code_blocks = re.findall(r"```[\w]*\n(.*?)```", content, re.DOTALL)
        for block in code_blocks:
            keywords.update(word.lower() for word in re.findall(r"\w+", block))
        return list(keywords)

    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search documentation for query."""
        results = []
        query_terms = query.lower().split()
        
        for file_path, data in self.index.items():
            score = 0
            content = data["content"].lower()
            
            # Score exact matches in headings
            for heading in data["headings"]:
                if query.lower() in heading.lower():
                    score += 3
            
            # Score keyword matches
            for term in query_terms:
                if term in data["keywords"]:
                    score += 2
            
            # Score content matches
            for term in query_terms:
                score += content.count(term)
            
            if score > 0:
                results.append({
                    "file": file_path,
                    "score": score,
                    "headings": data["headings"][:3],  # Show first 3 headings
                    "preview": self._get_preview(content, query)
                })
        
        return sorted(results, key=lambda x: x["score"], reverse=True)[:max_results]

    def _get_preview(self, content: str, query: str, context: int = 100) -> str:
        """Get content preview around match."""
        index = content.lower().find(query.lower())
        if index == -1:
            return content[:200] + "..."
        
        start = max(0, index - context)
        end = min(len(content), index + len(query) + context)
        return f"...{content[start:end]}..."

def main():
    parser = argparse.ArgumentParser(description="Search documentation")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--max-results", type=int, default=10, help="Maximum results to show")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    args = parser.parse_args()

    searcher = DocSearcher()
    results = searcher.search(args.query, args.max_results)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            print(f"\nFile: {result['file']}")
            print(f"Score: {result['score']}")
            print("Headings:", ", ".join(result['headings']))
            print("Preview:", result['preview'])
            print("-" * 80)

if __name__ == "__main__":
    main()
