"""
GraphRAG Query Engine with Local Ollama

This module demonstrates how to load an existing GraphRAG database
and query it using a local Ollama instance.

Prerequisites:
- Ollama installed and running locally (ollama serve)
- Python 3.10+
- graphrag package installed
- A model pulled in Ollama (e.g., ollama pull llama2)

Configuration:
Set environment variables or update config:
- LLM_MODEL=ollama/llama2
- LLM_API_BASE=http://localhost:11434
- EMBEDDING_MODEL=ollama/nomic-embed-text
- EMBEDDING_API_BASE=http://localhost:11434
"""

import os
import logging
from pathlib import Path
from typing import Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphRAGQueryEngine:
    """Load and query an existing GraphRAG database with local Ollama."""
    
    def __init__(
        self,
        llm_model: str = os.getenv("LLM_MODEL", "ollama/llama2"),
        llm_api_base: str = os.getenv("LLM_API_BASE", "http://localhost:11434"),
        embedding_model: str = os.getenv("EMBEDDING_MODEL", "ollama/nomic-embed-text"),
        embedding_api_base: str = os.getenv("EMBEDDING_API_BASE", "http://localhost:11434"),
        graph_db_path: str = "./graph_db",
    ):
        """
        Initialize GraphRAG query engine.
        
        Args:
            llm_model: LLM model identifier
            llm_api_base: Base URL for Ollama LLM API
            embedding_model: Embedding model identifier
            embedding_api_base: Base URL for Ollama embedding API
            graph_db_path: Path to graph database persistence directory
        """
        self.llm_model = llm_model
        self.llm_api_base = llm_api_base
        self.embedding_model = embedding_model
        self.embedding_api_base = embedding_api_base
        self.graph_db_path = Path(graph_db_path)
        
        self.config = self._create_config()
        self.graph_store = self._load_graph_db()
        logger.info("GraphRAG query engine initialized")
    
    def _create_config(self) -> dict:
        """Create GraphRAG configuration dictionary."""
        return {
            "llm_model": self.llm_model,
            "llm_api_base": self.llm_api_base,
            "embedding_model": self.embedding_model,
            "embedding_api_base": self.embedding_api_base,
            "graph_db_path": str(self.graph_db_path),
        }
    
    def _load_graph_db(self) -> dict:
        """Load graph database from disk."""
        db_file = self.graph_db_path / "graph_store.json"
        
        if not db_file.exists():
            logger.error(f"Graph database not found at {db_file}")
            raise FileNotFoundError(f"Graph database not found at {db_file}")
        
        try:
            with open(db_file, 'r', encoding='utf-8') as f:
                graph_store = json.load(f)
            logger.info(f"Loaded graph database from {db_file}")
            logger.info(f"Database contains {len(graph_store.get('nodes', []))} nodes and {len(graph_store.get('edges', []))} edges")
            return graph_store
        except Exception as e:
            logger.error(f"Error loading graph database: {e}")
            raise
    
    def query(self, query_text: str, top_k: int = 10) -> Optional[dict]:
        """
        Run a query against the loaded graph database.
        
        Args:
            query_text: Query string
            top_k: Number of top results to return
            
        Returns:
            Query results dictionary or None on error
        """
        try:
            from graphrag.query.structured_search import run_structured_query
            
            results = run_structured_query(
                query=query_text,
                config=self.config,
                top_k=top_k,
            )
            
            logger.info(f"Query executed successfully: '{query_text}'")
            return results
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return None


def main():
    """Example usage of GraphRAG query engine with local Ollama."""
    
    # Initialize query engine with existing graph database
    try:
        engine = GraphRAGQueryEngine(
            llm_model=os.getenv("LLM_MODEL", "ollama/llama2"),
            llm_api_base=os.getenv("LLM_API_BASE", "http://localhost:11434"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "ollama/nomic-embed-text"),
            embedding_api_base=os.getenv("EMBEDDING_API_BASE", "http://localhost:11434"),
            graph_db_path=os.getenv("GRAPH_DB_PATH", "./graph_db"),
        )
    except FileNotFoundError as e:
        logger.error(f"Failed to initialize query engine: {e}")
        return
    
    # Example: Run queries
    queries = [
        "What is GraphRAG?",
        "How does local Ollama work?",
        "What are the key features of this system?",
    ]
    
    logger.info("Running sample queries against graph database...")
    for query in queries:
        logger.info(f"\nQuery: {query}")
        results = engine.query(query)
        if results:
            logger.info(f"Results: {results}")


if __name__ == "__main__":
    main()
