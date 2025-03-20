import logging
import os
import stat
import uuid
import tempfile
from typing import List, Dict, Optional, Union
from pathlib import Path
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
from .config import (
    QDRANT_PATH, QDRANT_COLLECTION,
    VECTOR_SIZE, EMBEDDING_MODEL
)

logger = logging.getLogger(__name__)

class VectorStore:
    _instance = None

    def __new__(cls, storage_path=None):
        if cls._instance is None or storage_path:  # Create new instance if specific path requested
            instance = super(VectorStore, cls).__new__(cls)
            instance.initialized = False
            if storage_path:
                # For testing: Don't use singleton with custom path
                return instance
            cls._instance = instance
        return cls._instance

    def __init__(self, storage_path=None):
        if hasattr(self, 'initialized') and not self.initialized or storage_path:
            self.storage_path = storage_path or QDRANT_PATH
            
            # Ensure directory exists with proper permissions
            logger.debug(f"Setting up vector store at {self.storage_path}")
            os.makedirs(self.storage_path, exist_ok=True)
            
            # Set full permissions for testing
            try:
                for root, dirs, files in os.walk(self.storage_path):
                    for d in dirs:
                        os.chmod(os.path.join(root, d), 0o777)
                    for f in files:
                        os.chmod(os.path.join(root, f), 0o666)
                
                # Set directory permissions
                os.chmod(self.storage_path, 0o777)
            except Exception as e:
                logger.warning(f"Could not set permissions: {e}")
            
            try:
                self.client = QdrantClient(path=str(self.storage_path))
                self.model = SentenceTransformer(EMBEDDING_MODEL)
                self._ensure_collection()
                self.initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize vector store: {e}")
                raise

    def _ensure_collection(self):
        """Ensure the vector collection exists"""
        try:
            self.client.get_collection(QDRANT_COLLECTION)
        except Exception:
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created collection: {QDRANT_COLLECTION}")

    def add_texts(self, texts: List[str], metadata: Optional[List[Dict]] = None) -> List[str]:
        """Add texts to the vector store"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        points = []
        
        if metadata is None:
            metadata = [{} for _ in texts]

        for i, (text, embedding, meta) in enumerate(zip(texts, embeddings, metadata)):
            point_id = uuid.uuid4().int % (2**63)
            points.append(models.PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "text": text,
                    **meta
                }
            ))

        self.client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=points
        )
        return [str(p.id) for p in points]

    def similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar texts using vector similarity"""
        vector = self.model.encode(query, convert_to_numpy=True)
        
        # Convert filter dictionary to proper Qdrant filter format
        query_filter = None
        if filter:
            # Create proper filter condition
            filter_conditions = []
            for key, value in filter.items():
                filter_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            
            if filter_conditions:
                query_filter = models.Filter(
                    must=filter_conditions
                )
        
        # NOTE: Using deprecated 'search' method because it works consistently.
        # The recommended 'query_points' method had compatibility issues.
        search_result = self.client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=vector.tolist(),
            limit=k,
            query_filter=query_filter
        )
        
        return [{
            "text": hit.payload["text"],
            "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
            "score": hit.score
        } for hit in search_result]

    def delete_texts(self, ids: List[str]):
        """Delete texts by their IDs"""
        self.client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=models.PointIdsList(
                points=list(map(int, ids))
            )
        )

    def close(self):
        """Close the Qdrant client connection"""
        if hasattr(self, 'client'):
            self.client.close()
            delattr(self, 'client')
        self.initialized = False

    def __del__(self):
        """Cleanup when instance is deleted"""
        try:
            self.close()
        except:
            pass

# Don't create instance on import
vector_store = None

def get_vector_store(storage_path=None):
    """Get or create vector store instance"""
    global vector_store
    if vector_store is None:
        vector_store = VectorStore(storage_path)
    return vector_store
