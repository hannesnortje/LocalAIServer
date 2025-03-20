import logging
import time
import uuid
import os
from pathlib import Path  # Add this import
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union, Any

from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

from .config import (
    QDRANT_PATH, VECTOR_SIZE, EMBEDDING_MODEL,
    RESPONSE_HISTORY_COLLECTION, ENABLE_RESPONSE_HISTORY,
    MAX_HISTORY_ITEMS, HISTORY_RETENTION_DAYS
)

logger = logging.getLogger(__name__)

class ResponseHistoryManager:
    """Manager for storing and retrieving response history in the vector store.
    
    This class handles saving responses, retrieving historical responses,
    and cleanup of old entries.
    """
    _instance = None
    
    def __new__(cls, storage_path=None):
        if cls._instance is None or storage_path:  # Allow custom path for testing
            instance = super(ResponseHistoryManager, cls).__new__(cls)
            instance.initialized = False
            if storage_path:
                # For testing: Don't use singleton with custom path
                return instance
            cls._instance = instance
        return cls._instance
    
    def __init__(self, storage_path=None):
        if not hasattr(self, 'initialized') or not self.initialized or storage_path:
            self.storage_path = storage_path or QDRANT_PATH
            self.enabled = ENABLE_RESPONSE_HISTORY
            self.initialized = False  # Set to False until fully initialized
            logger.debug(f"Preparing response history manager with storage at {self.storage_path}")
    
    def _initialize(self):
        """Delayed initialization to avoid lock conflicts."""
        if self.initialized:
            return
        
        try:
            # Log additional debugging info
            logger.debug(f"Initializing history manager with path: {self.storage_path}")
            logger.debug(f"Directory exists: {Path(self.storage_path).exists()}")
            
            # Ensure the directory exists with proper permissions
            os.makedirs(self.storage_path, exist_ok=True)
            try:
                os.chmod(self.storage_path, 0o777)
            except Exception as e:
                logger.warning(f"Could not set permissions: {e}")
            
            # Initialize vector store client
            self.client = QdrantClient(path=str(self.storage_path))
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            
            # Force enabled for tests
            if self.storage_path != QDRANT_PATH:  # This is a test instance
                logger.debug("Forcing history enabled for test instance")
                self.enabled = True
            
            # Create collection if it doesn't exist
            self._ensure_collection()
            
            self.initialized = True
            logger.info(f"Response history manager initialized (enabled: {self.enabled})")
        except Exception as e:
            logger.error(f"Failed to initialize history manager: {e}")
            raise
    
    def _ensure_collection(self):
        """Ensure the response history collection exists"""
        if not hasattr(self, 'client'):
            self._initialize()
            
        try:
            self.client.get_collection(RESPONSE_HISTORY_COLLECTION)
            logger.debug(f"Found existing collection: {RESPONSE_HISTORY_COLLECTION}")
        except Exception:
            self.client.create_collection(
                collection_name=RESPONSE_HISTORY_COLLECTION,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created collection: {RESPONSE_HISTORY_COLLECTION}")
    
    def save_response(self, query: str, response: Any, metadata: Optional[Dict] = None) -> Optional[str]:
        """Save a response to the history.
        
        Args:
            query: The user's query
            response: The system's response
            metadata: Additional metadata about the response
            
        Returns:
            ID of the saved entry or None if history is disabled
        """
        if not self.initialized:
            logger.debug("Initializing history manager during save_response call")
            self._initialize()
            
        # Additional debug logging
        logger.debug(f"History enabled: {self.enabled}")
        
        if not self.enabled:
            logger.debug("History is disabled, not saving response")
            return None
        
        try:
            # Create embedding for the query
            query_embedding = self.model.encode(query, convert_to_numpy=True)
            
            # Generate a unique ID
            point_id = uuid.uuid4().int % (2**63)
            
            # Create metadata if not provided
            if metadata is None:
                metadata = {}
            
            # Ensure we have a timestamp
            if 'timestamp' not in metadata:
                metadata['timestamp'] = time.time()
                
            # Add the point to the collection
            self.client.upsert(
                collection_name=RESPONSE_HISTORY_COLLECTION,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=query_embedding.tolist(),
                        payload={
                            "query": query,
                            "response": response,
                            "metadata": metadata
                        }
                    )
                ]
            )
            
            logger.debug(f"Saved response to history with ID: {point_id}")
            return str(point_id)
            
        except Exception as e:
            logger.error(f"Error saving response to history: {e}")
            return None
    
    def find_similar_responses(
        self, 
        query: str, 
        limit: int = MAX_HISTORY_ITEMS,
        min_score: float = 0.7,
        filter_params: Optional[Dict] = None
    ) -> List[Dict]:
        """Find similar previous responses based on query similarity.
        
        Args:
            query: The current query
            limit: Maximum number of responses to return
            min_score: Minimum similarity score (0-1)
            filter_params: Additional filters
            
        Returns:
            List of similar responses with metadata
        """
        if not self.enabled:
            return []
            
        if not self.initialized:
            self._initialize()
            
        try:
            # Generate embedding for query
            query_embedding = self.model.encode(query, convert_to_numpy=True)
            
            # Set up filter 
            query_filter = None
            if filter_params:
                filter_conditions = []
                for key, value in filter_params.items():
                    filter_conditions.append(
                        models.FieldCondition(
                            key=f"metadata.{key}",
                            match=models.MatchValue(value=value)
                        )
                    )
                
                if filter_conditions:
                    query_filter = models.Filter(
                        must=filter_conditions
                    )
            
            # Search for similar responses
            results = self.client.search(
                collection_name=RESPONSE_HISTORY_COLLECTION,
                query_vector=query_embedding.tolist(),
                limit=limit,
                query_filter=query_filter,
                score_threshold=min_score
            )
            
            # Format results
            responses = []
            for hit in results:
                responses.append({
                    "query": hit.payload["query"],
                    "response": hit.payload["response"],
                    "metadata": hit.payload["metadata"],
                    "similarity": hit.score
                })
                
            logger.debug(f"Found {len(responses)} similar historical responses")
            return responses
            
        except Exception as e:
            logger.error(f"Error finding similar responses: {e}")
            return []
    
    def clean_old_entries(self, days: int = HISTORY_RETENTION_DAYS) -> int:
        """Remove entries older than specified days.
        
        Args:
            days: Number of days to keep
            
        Returns:
            Number of entries removed
        """
        if not self.enabled:
            return 0
            
        if not self.initialized:
            self._initialize()
            
        try:
            # Calculate cutoff timestamp
            cutoff_time = time.time() - (days * 24 * 60 * 60)
            
            # Create filter to find old entries
            old_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="metadata.timestamp",
                        range=models.Range(
                            lt=cutoff_time
                        )
                    )
                ]
            )
            
            # Search for old entries to get count (with limit=0, we just get count)
            count_result = self.client.count(
                collection_name=RESPONSE_HISTORY_COLLECTION,
                count_filter=old_filter
            )
            
            if count_result.count == 0:
                logger.debug("No old entries to clean up")
                return 0
                
            # Delete old entries
            self.client.delete(
                collection_name=RESPONSE_HISTORY_COLLECTION,
                points_selector=models.FilterSelector(
                    filter=old_filter
                )
            )
            
            logger.info(f"Cleaned up {count_result.count} old history entries")
            return count_result.count
            
        except Exception as e:
            logger.error(f"Error cleaning old entries: {e}")
            return 0
    
    def delete_all_history(self) -> bool:
        """Delete all history entries.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False
            
        if not self.initialized:
            self._initialize()
            
        try:
            # Recreate the collection
            self.client.delete_collection(RESPONSE_HISTORY_COLLECTION)
            self._ensure_collection()
            
            logger.info("Deleted all response history")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting all history: {e}")
            return False

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

# Create a lazy-loading global instance instead of initializing immediately
response_history = ResponseHistoryManager()

def get_response_history(storage_path=None):
    """Get or create response history manager instance."""
    global response_history
    if storage_path:
        # For testing - create a new instance with custom path
        # Close existing instance if any
        if response_history and hasattr(response_history, 'client'):
            try:
                response_history.close()
            except:
                pass
        return ResponseHistoryManager(storage_path=storage_path)
    
    # Initialize if needed
    if not response_history.initialized:
        response_history._initialize()
    return response_history
