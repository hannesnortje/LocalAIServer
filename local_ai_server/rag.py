import logging
from typing import List, Dict, Optional, Union, Any
import time

from .vector_store import get_vector_store
from .model_manager import model_manager
from .config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE

logger = logging.getLogger(__name__)

class RAG:
    """Retrieval-Augmented Generation utility.
    
    This class combines document retrieval from the vector store with
    language model generation to create context-enhanced responses.
    """
    
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
    def generate_rag_response(
        query: str,
        model_name: str,
        search_params: Optional[Dict] = None,
        generation_params: Optional[Dict] = None
    ) -> Dict:
        """Generate a response using retrieved documents as context.
        
        Args:
            query: User query
            model_name: Name of the language model to use
            search_params: Parameters for document search
            generation_params: Parameters for LLM generation
            
        Returns:
            Dict containing response text and metadata
        """
        start_time = time.time()
        
        # Default parameters
        search_params = search_params or {}
        generation_params = generation_params or {}
        
        # Get vector store
        vector_store = get_vector_store()
        
        # Set search parameters
        k = search_params.get('limit', 4)
        filter_params = search_params.get('filter')
        
        # Retrieve relevant documents
        try:
            results = vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter_params
            )
            
            retrieved_docs = RAG.format_retrieved_documents(results)
            logger.debug(f"Retrieved {len(results)} documents")
            
            # Load the specified model
            if model_manager.model is None or model_manager.current_model_name != model_name:
                model_manager.load_model(model_name)
            
            # Construct prompt with retrieved documents
            rag_prompt = f"""You are an AI assistant that answers questions based on the provided documents.
            
DOCUMENTS:
{retrieved_docs}

USER QUESTION: {query}

Please provide a comprehensive answer based solely on the information in the documents. If the documents don't contain relevant information, state that you don't have enough information to answer properly.

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
                "metadata": {
                    "query": query,
                    "document_count": len(results),
                    "response_time": time.time() - start_time
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG response generation: {str(e)}", exc_info=True)
            raise
