"""
Vector database utilities for document management and retrieval.

This module handles all vector database operations including:
- Vector store initialization
- Document indexing
- Similarity search
- Document filtering and processing
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

try:
    from .logger import LoggerSetup
    from .models import VectorSearchResult
except ImportError:
    from logger import LoggerSetup
    from models import VectorSearchResult

logger = LoggerSetup.setup_logger(__name__)


class DocumentLoader:
    """Handle loading and processing of documents from various sources."""

    @staticmethod
    def load_pdf_documents(directory_path: str) -> List[Document]:
        """
        Load all PDF documents from a directory.
        
        Args:
            directory_path: Path to directory containing PDF files
            
        Returns:
            List of loaded documents
            
        TODO: Add support for other document formats (DOCX, RTF, HTML)
        TODO: Add document metadata extraction
        """
        logger.info(f"Loading PDF documents from: {directory_path}")
        try:
            loader = DirectoryLoader(
                directory_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            documents = loader.load()
            logger.info(f"Successfully loaded {len(documents)} PDF documents")
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF documents: {e}")
            raise

    @staticmethod
    def load_text_documents(directory_path: str) -> List[Document]:
        """
        Load all text documents from a directory.
        
        Args:
            directory_path: Path to directory containing text files
            
        Returns:
            List of loaded documents
        """
        logger.info(f"Loading text documents from: {directory_path}")
        try:
            loader = DirectoryLoader(
                directory_path,
                glob="**/*.txt",
                loader_cls=TextLoader
            )
            documents = loader.load()
            logger.info(f"Successfully loaded {len(documents)} text documents")
            return documents
        except Exception as e:
            logger.error(f"Error loading text documents: {e}")
            raise

    @staticmethod
    def filter_documents(documents: List[Document]) -> List[Document]:
        """
        Filter and clean documents, preserving important metadata.
        
        Args:
            documents: List of documents to filter
            
        Returns:
            List of filtered documents with cleaned metadata
        """
        logger.info(f"Filtering {len(documents)} documents")
        filtered_docs = []
        
        for doc in documents:
            source = doc.metadata.get('source', 'Unknown')
            filtered_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata={
                        'source': source,
                        'author': doc.metadata.get('author', 'Unknown'),
                        'page': doc.metadata.get('page', None)
                    }
                )
            )
        
        logger.info(f"Filtered to {len(filtered_docs)} documents")
        return filtered_docs


class DocumentSplitter:
    """Handle document splitting and chunking strategies."""

    @staticmethod
    def split_documents(
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Document]:
        """
        Split documents into smaller chunks for better embedding and retrieval.
        
        Args:
            documents: Documents to split
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of split documents
            
        TODO: Implement semantic splitting based on content structure
        TODO: Add support for custom separators per document type
        """
        logger.info(
            f"Splitting {len(documents)} documents with "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}"
        )
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            split_docs = text_splitter.split_documents(documents)
            logger.info(f"Split into {len(split_docs)} chunks")
            return split_docs
        except Exception as e:
            logger.error(f"Error splitting documents: {e}")
            raise


class EmbeddingManager:
    """Manage embedding operations and models."""

    @staticmethod
    def get_embeddings(model_name: str = "nomic-embed-text:latest") -> OllamaEmbeddings:
        """
        Get embeddings model instance.
        
        Args:
            model_name: Name of the embedding model to use
            
        Returns:
            OllamaEmbeddings instance
        """
        logger.info(f"Initializing embeddings with model: {model_name}")
        try:
            embeddings = OllamaEmbeddings(model=model_name)
            logger.info("Embeddings initialized successfully")
            return embeddings
        except Exception as e:
            logger.error(f"Error initializing embeddings: {e}")
            raise


class VectorStore:
    """Manage vector database operations with Pinecone."""

    @staticmethod
    def initialize_pinecone(api_key: str) -> Pinecone:
        """
        Initialize Pinecone client.
        
        Args:
            api_key: Pinecone API key
            
        Returns:
            Initialized Pinecone client
        """
        logger.info("Initializing Pinecone client")
        try:
            pc = Pinecone(api_key=api_key)
            logger.info("Pinecone client initialized successfully")
            return pc
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {e}")
            raise

    @staticmethod
    def create_index_if_not_exists(
        pc: Pinecone,
        index_name: str,
        dimension: int = 768,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1"
    ) -> None:
        """
        Create Pinecone index if it doesn't exist.
        
        Args:
            pc: Pinecone client
            index_name: Name of the index
            dimension: Vector dimension (must match embedding model)
            metric: Distance metric ('cosine', 'euclidean', or 'dotproduct')
            cloud: Cloud provider
            region: Cloud region
            
        TODO: Add support for different index types (sparse, hybrid)
        TODO: Add index configuration validation
        """
        logger.info(f"Checking/Creating index: {index_name}")
        try:
            if not pc.has_index(index_name):
                logger.info(f"Creating new index: {index_name}")
                pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=ServerlessSpec(cloud=cloud, region=region)
                )
                logger.info(f"Index {index_name} created successfully")
            else:
                logger.info(f"Index {index_name} already exists")
        except Exception as e:
            logger.error(f"Error creating index: {e}")
            raise

    @staticmethod
    def create_vectorstore(
        documents: List[Document],
        embeddings: OllamaEmbeddings,
        index_name: str
    ) -> PineconeVectorStore:
        """
        Create vector store from documents.
        
        Args:
            documents: Documents to index
            embeddings: Embeddings model
            index_name: Pinecone index name
            
        Returns:
            PineconeVectorStore instance
        """
        logger.info(f"Creating vector store with {len(documents)} documents for index: {index_name}")
        try:
            vectorstore = PineconeVectorStore.from_documents(
                documents=documents,
                embedding=embeddings,
                index_name=index_name
            )
            logger.info("Vector store created successfully")
            return vectorstore
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise

    @staticmethod
    def load_vectorstore(
        embeddings: OllamaEmbeddings,
        index_name: str
    ) -> PineconeVectorStore:
        """
        Load existing vector store.
        
        Args:
            embeddings: Embeddings model
            index_name: Pinecone index name
            
        Returns:
            PineconeVectorStore instance
        """
        logger.info(f"Loading vector store from index: {index_name}")
        try:
            vectorstore = PineconeVectorStore.from_existing_index(
                embedding=embeddings,
                index_name=index_name
            )
            logger.info("Vector store loaded successfully")
            return vectorstore
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise

    @staticmethod
    def add_documents_to_vectorstore(
        vectorstore: PineconeVectorStore,
        documents: List[Document]
    ) -> None:
        """
        Add new documents to existing vector store.
        
        Args:
            vectorstore: Existing PineconeVectorStore
            documents: Documents to add
        """
        logger.info(f"Adding {len(documents)} documents to vector store")
        try:
            vectorstore.add_documents(documents)
            logger.info("Documents added successfully")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise


class VectorSearch:
    """Perform searches on vector store."""

    @staticmethod
    def search_similar_documents(
        vectorstore: PineconeVectorStore,
        query: str,
        k: int = 3,
        search_type: str = "similarity"
    ) -> List[VectorSearchResult]:
        """
        Search for documents similar to query.
        
        Args:
            vectorstore: PineconeVectorStore instance
            query: Search query
            k: Number of results to return
            search_type: Type of search ('similarity' or 'mmr')
            
        Returns:
            List of VectorSearchResult objects
            
        TODO: Add MMR (Maximum Marginal Relevance) search option
        TODO: Add metadata filtering
        TODO: Add hybrid search combining keyword and semantic search
        """
        logger.info(f"Performing {search_type} search with query: {query}")
        try:
            retriever = vectorstore.as_retriever(
                search_type=search_type,
                search_kwargs={"k": k}
            )
            results = retriever.invoke(query)
            
            # Convert to VectorSearchResult objects
            search_results = [
                VectorSearchResult(
                    content=doc.page_content,
                    source=doc.metadata.get('source', 'Unknown'),
                    relevance_score=0.8,  # Placeholder - actual score from Pinecone
                    metadata=doc.metadata
                )
                for doc in results
            ]
            
            logger.info(f"Found {len(search_results)} similar documents")
            return search_results
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise

    @staticmethod
    def has_sufficient_context(
        search_results: List[VectorSearchResult],
        relevance_threshold: float = 0.5
    ) -> bool:
        """
        Determine if search results provide sufficient context.
        
        Args:
            search_results: Results from vector search
            relevance_threshold: Minimum relevance score
            
        Returns:
            Boolean indicating sufficient context
            
        TODO: Implement more sophisticated context sufficiency heuristics
        TODO: Add query complexity-based thresholds
        """
        if not search_results:
            return False
        
        has_relevant = any(
            result.relevance_score >= relevance_threshold
            for result in search_results
        )
        
        return has_relevant
