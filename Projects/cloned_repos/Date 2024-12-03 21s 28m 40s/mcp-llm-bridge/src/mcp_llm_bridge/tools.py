from typing import Dict, List, Any
from dataclasses import dataclass
import sqlite3
import logging

@dataclass
class DatabaseSchema:
    """Represents the schema of a database table"""
    table_name: str
    columns: Dict[str, str]
    description: str

class DatabaseQueryTool:
    """Tool for executing database queries with schema validation"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.schemas: Dict[str, DatabaseSchema] = {}
        
        # Register default product schema
        self.register_schema(DatabaseSchema(
            table_name="products",
            columns={
                "id": "INTEGER",
                "title": "TEXT",
                "description": "TEXT",
                "price": "REAL",
                "category": "TEXT",
                "stock": "INTEGER",
                "created_at": "DATETIME"
            },
            description="Product catalog with items for sale"
        ))
    
    def register_schema(self, schema: DatabaseSchema):
        """Register a database schema"""
        self.schemas[schema.table_name] = schema
    
    def get_tool_spec(self) -> Dict[str, Any]:
        """Get the tool specification in MCP format"""
        schema_desc = "\n".join([
            f"Table {schema.table_name}: {schema.description}\n"
            f"Columns: {', '.join(f'{name} ({type_})' for name, type_ in schema.columns.items())}"
            for schema in self.schemas.values()
        ])
        
        return {
            "name": "query_database",
            "description": f"Execute SQL queries against the database. Available schemas:\n{schema_desc}",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "SQL query to execute"
                    }
                },
                "required": ["query"]
            }
        }
    
    def get_schema_description(self) -> str:
        """Get a formatted description of all registered schemas"""
        schema_parts = []
        for schema in self.schemas.values():
            column_info = []
            for name, type_ in schema.columns.items():
                column_info.append(f"  - {name} ({type_})")
            schema_parts.append(f"Table {schema.table_name}: {schema.description}\n" + "\n".join(column_info))
            
        return "\n\n".join(schema_parts)
    
    def validate_query(self, query: str) -> bool:
        """Validate a query against registered schemas"""
        query = query.lower()
        for schema in self.schemas.values():
            if schema.table_name in query:
                # Check if query references any non-existent columns
                for word in query.split():
                    if '.' in word:
                        table, column = word.split('.')
                        if table == schema.table_name and column not in schema.columns:
                            return False
        return True
    
    async def execute(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a SQL query and return results"""
        query = params.get("query")
        if not query:
            raise ValueError("Query parameter is required")
            
        if not self.validate_query(query):
            raise ValueError("Query references invalid columns")
            
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [description[0] for description in cursor.description]
            results = []
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
            return results
        finally:
            conn.close()