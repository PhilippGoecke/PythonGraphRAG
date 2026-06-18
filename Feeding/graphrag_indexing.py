"""
GraphRAG Indexing with Local Ollama Example

This module demonstrates how to index documents and set up GraphRAG
with a local Ollama instance for LLM and embedding operations.

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


class GraphRAGIndexer:
    """Handle indexing and initialization of GraphRAG with local Ollama."""
    
    def __init__(
        self,
        llm_model: str = os.getenv("LLM_MODEL", "ollama/llama2"),
        llm_api_base: str = os.getenv("LLM_API_BASE", "http://localhost:11434"),
        embedding_model: str = os.getenv("EMBEDDING_MODEL", "ollama/nomic-embed-text"),
        embedding_api_base: str = os.getenv("EMBEDDING_API_BASE", "http://localhost:11434"),
        data_dir: str = "./data",
        graph_db_path: str = "./graph_db",
    ):
        """
        Initialize GraphRAG configuration for local Ollama.
        
        Args:
            llm_model: LLM model identifier
            llm_api_base: Base URL for Ollama LLM API
            embedding_model: Embedding model identifier
            embedding_api_base: Base URL for Ollama embedding API
            data_dir: Directory containing documents to index
            graph_db_path: Path to graph database persistence directory
        """
        self.llm_model = llm_model
        self.llm_api_base = llm_api_base
        self.embedding_model = embedding_model
        self.embedding_api_base = embedding_api_base
        self.data_dir = Path(data_dir)
        self.graph_db_path = Path(graph_db_path)
        
        # Create data and graph database directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.graph_db_path.mkdir(parents=True, exist_ok=True)
        
        self.config = self._create_config()
        self.graph_store = self._init_graph_db()
        logger.info("GraphRAG indexer initialized with Ollama backend and graph database persistence")
    
    def _create_config(self) -> dict:
        """Create GraphRAG configuration dictionary."""
        return {
            "llm_model": self.llm_model,
            "llm_api_base": self.llm_api_base,
            "embedding_model": self.embedding_model,
            "embedding_api_base": self.embedding_api_base,
            "data_dir": str(self.data_dir),
            "graph_db_path": str(self.graph_db_path),
        }
    
    def _init_graph_db(self) -> dict:
        """Initialize or load graph database for persistence."""
        db_file = self.graph_db_path / "graph_store.json"
        
        if db_file.exists():
            try:
                with open(db_file, 'r', encoding='utf-8') as f:
                    graph_store = json.load(f)
                logger.info("Loaded existing graph database")
                return graph_store
            except Exception as e:
                logger.error(f"Error loading graph database: {e}")
                return {"nodes": [], "edges": []}
        else:
            logger.info("Initializing new graph database")
            return {"nodes": [], "edges": []}
    
    def _persist_graph_db(self) -> bool:
        """Persist graph database to disk."""
        try:
            db_file = self.graph_db_path / "graph_store.json"
            with open(db_file, 'w', encoding='utf-8') as f:
                json.dump(self.graph_store, f, indent=2)
            logger.info("Graph database persisted to disk")
            return True
        except Exception as e:
            logger.error(f"Error persisting graph database: {e}")
            return False
    
    def index_documents(self, documents: list[str]) -> bool:
        """
        Index a list of documents using GraphRAG.
        
        Args:
            documents: List of document texts to index
            
        Returns:
            True if indexing successful, False otherwise
        """
        try:
            logger.info(f"Starting indexing of {len(documents)} documents...")
            
            # Import here to avoid hard dependency
            try:
                from graphrag.index import build_index
            except ImportError:
                logger.error("graphrag package not installed. Install with: pip install graphrag")
                return False
            
            # Build index with configuration
            index = build_index(
                documents=documents,
                config=self.config,
            )
            
            # Store index in graph database
            self.graph_store["index"] = str(index)
            self.graph_store["document_count"] = len(documents)
            self._persist_graph_db()
            
            logger.info("Indexing completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            return False
    
    def index_from_files(self, file_pattern: str = "*.txt") -> bool:
        """
        Index documents from files in data directory.
        
        Args:
            file_pattern: File pattern to match (default: *.txt)
            
        Returns:
            True if indexing successful, False otherwise
        """
        try:
            # Find all matching files
            files = list(self.data_dir.glob(file_pattern))
            
            if not files:
                logger.warning(f"No files matching '{file_pattern}' found in {self.data_dir}")
                return False
            
            logger.info(f"Found {len(files)} files to index")
            
            # Read documents from files
            documents = []
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        documents.append(content)
                        logger.info(f"Loaded: {file_path}")
                except Exception as e:
                    logger.error(f"Error reading {file_path}: {e}")
            
            # Index documents
            return self.index_documents(documents)
            
        except Exception as e:
            logger.error(f"Error indexing from files: {e}")
            return False
    
    def query(self, query_text: str, top_k: int = 10) -> Optional[dict]:
        """
        Run a query against the indexed documents.
        
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
            
            logger.info(f"Query executed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return None


def main():
    """Example usage of GraphRAG indexer with local Ollama."""
    
    # Initialize indexer with graph database persistence
    indexer = GraphRAGIndexer(
        llm_model=os.getenv("LLM_MODEL", "ollama/llama2"),
        llm_api_base=os.getenv("LLM_API_BASE", "http://localhost:11434"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "ollama/nomic-embed-text"),
        embedding_api_base=os.getenv("EMBEDDING_API_BASE", "http://localhost:11434"),
        data_dir=os.getenv("DATA_DIR", "./data"),
        graph_db_path=os.getenv("GRAPH_DB_PATH", "./graph_db"),
    )
    
    # Example 1: Index from sample documents
    sample_docs = [
        "GraphRAG is a framework for building knowledge graphs from unstructured text.",
        "Local Ollama provides LLM and embedding capabilities without cloud dependencies.",
        "The system supports entity extraction, relationship detection, and semantic search.",
    ]
    
    logger.info("Example 1: Indexing sample documents...")
    success = indexer.index_documents(sample_docs)
    
    if success:
        # Example 2: Query indexed documents
        logger.info("\nExample 2: Running sample queries...")
        
        query = "What is GraphRAG?"
        results = indexer.query(query)
        if results:
            logger.info(f"Query results: {results}")
    
    # Example 3: Index from files (uncomment to use)
    # logger.info("\nExample 3: Indexing from files...")
    # indexer.index_from_files("*.txt")


if __name__ == "__main__":
    main()
