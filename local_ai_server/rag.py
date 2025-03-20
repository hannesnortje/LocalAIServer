import logging
from typing import List, Dict, Optional, Union, Any
import time

from .vector_store import get_vector_store
from .model_manager import model_manager
from .history_manager import get_response_history  # Use the factory function instead
from .config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, ENABLE_RESPONSE_HISTORY

logger = logging.getLogger(__name__)

class RAG:
    """Retrieval-Augmented Generation utility."""
    
    vector_store = None  # Add class variable to store vector store instance
    
    @staticmethod
    def format_retrieved_documents(docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents for inclusion in the prompt.
        
        Args:
            docs: List of document dictionaries with 'text' and 'metadata' keys
            
        Returns:
            Formatted string with document contents
        """
        if not docs:
            return "No relevant documents found."
        
        formatted_docs = []
        for i, doc in enumerate(docs, 1):
            source = ""
            if doc.get("metadata") and doc["metadata"].get("source"):
                source = f" (Source: {doc['metadata']['source']})"
                
            formatted_docs.append(f"Document {i}{source}:\n{doc['text']}\n")
            
        return "\n".join(formatted_docs)
    
    @staticmethod
    def format_history_responses(history_items: List[Dict]) -> str:
        """Format historical responses for inclusion in the prompt.
        
        Args:
            history_items: List of historical response dictionaries
            
        Returns:
            Formatted string with historical responses
        """
        if not history_items:
            return ""
        
        history_text = ["PREVIOUS QUESTIONS AND ANSWERS:"]
        
        for i, item in enumerate(history_items, 1):
            timestamp = item.get("metadata", {}).get("timestamp", "")
            if timestamp:
                date_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(timestamp))
                history_text.append(f"Date: {date_str}")
                
            history_text.append(f"User: {item['query']}")
            history_text.append(f"Assistant: {item['response']}")
            history_text.append("")  # Empty line between entries
            
        return "\n".join(history_text)
    
    @staticmethod
    def generate_rag_response(
        query: str,
        model_name: str,
        search_params: Optional[Dict] = None,
        generation_params: Optional[Dict] = None,
        use_history: Optional[bool] = None
    ) -> Dict:
        """Generate a response using retrieved documents as context.
        
        Args:
            query: User query
            model_name: Name of the language model to use
            search_params: Parameters for document search
            generation_params: Parameters for LLM generation
            use_history: Whether to include response history (defaults to global setting)
            
        Returns:
            Dict containing response text and metadata
        """
        start_time = time.time()
        
        # Set default parameters
        search_params = search_params or {}
        generation_params = generation_params or {}
        
        # Determine if we should use history
        if use_history is None:
            use_history = ENABLE_RESPONSE_HISTORY
        
        # Use class vector store if set, otherwise get new instance
        vector_store = RAG.vector_store or get_vector_store()
        
        # Set search parameters
        k = search_params.get('limit', 4)
        filter_params = search_params.get('filter')
        
        try:
            # Get historical responses if enabled
            history_items = []
            if use_history:
                history_manager = get_response_history()  # Use factory function
                history_limit = search_params.get('history_limit', 3)
                history_items = history_manager.find_similar_responses(
                    query=query,
                    limit=history_limit,
                    filter_params=search_params.get('history_filter')
                )
            
            # Retrieve relevant documents
            results = vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter_params
            )
            
            retrieved_docs = RAG.format_retrieved_documents(results)
            history_context = RAG.format_history_responses(history_items)
            
            logger.debug(f"Retrieved {len(results)} documents and {len(history_items)} history items")
            
            # Load the specified model
            if model_manager.model is None or model_manager.current_model_name != model_name:
                model_manager.load_model(model_name)
            
            # Construct prompt with retrieved documents and history
            rag_prompt = f"""You are an AI assistant that answers questions based on the provided documents and conversation history.

{history_context}

DOCUMENTS:
{retrieved_docs}

USER QUESTION: {query}

Please provide a comprehensive answer based on the information in the documents and previous conversations. If the documents don't contain relevant information, state that you don't have enough information to answer properly.

ANSWER:"""

            # Set generation parameters
            gen_params = {
                "temperature": generation_params.get("temperature", DEFAULT_TEMPERATURE),
                "max_tokens": generation_params.get("max_tokens", DEFAULT_MAX_TOKENS)
            }
            
            # Add other generation parameters if provided
            for param in ['top_p', 'frequency_penalty', 'presence_penalty', 'stop', 'stream']:
                if param in generation_params:
                    gen_params[param] = generation_params[param]
            
            # Generate the response
            response_text = model_manager.generate(rag_prompt, **gen_params)
            
            # Create response
            response = {
                "answer": response_text,
                "model": model_name,
                "retrieved_documents": results,
                "history_items": history_items if use_history else [],
                "metadata": {
                    "query": query,
                    "document_count": len(results),
                    "history_count": len(history_items),
                    "response_time": time.time() - start_time
                }
            }
            
            # Save to history if enabled
            if use_history:
                history_manager = get_response_history()  # Use factory function
                history_manager.save_response(
                    query=query,
                    response=response_text,
                    metadata={
                        "timestamp": time.time(),
                        "model": model_name,
                        "document_count": len(results)
                    }
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG response generation: {str(e)}", exc_info=True)
            raise
