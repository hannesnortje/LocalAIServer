import logging
from typing import List, Dict, Optional, Union, Any
import time

from .vector_store_factory import get_vector_store
from .model_manager import model_manager
from .history_manager import get_response_history  # Use the factory function instead
from .config import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, ENABLE_RESPONSE_HISTORY

# Import the global vector_store and history_manager from server.py
from .app_state import vector_store as app_vector_store, history_manager as app_history_manager

logger = logging.getLogger(__name__)

class RAG:
    """Retrieval-Augmented Generation utility."""
    
    # We'll use the application-level shared instances by default
    vector_store = None  # Will fall back to app_vector_store if None
    
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
        
        # Use global instances from the application
        try:
            # Use class vector store if set, otherwise use application instance
            vector_store = RAG.vector_store or app_vector_store
            
            # Set search parameters
            k = search_params.get('limit', 4)
            filter_params = search_params.get('filter')
            
            # Get historical responses if enabled
            history_items = []
            if use_history and app_history_manager is not None:  # Check if history manager exists
                try:
                    history_manager = app_history_manager  # Use application instance
                    history_limit = search_params.get('history_limit', 3)
                    history_items = history_manager.find_similar_responses(
                        query=query,
                        limit=history_limit,
                        filter_params=search_params.get('history_filter')
                    )
                except Exception as history_error:
                    logger.warning(f"Failed to fetch history: {history_error}")
                    # Continue without history
            
            # Check if vector store exists
            if vector_store is None:
                logger.warning("No vector store available. Proceeding without document retrieval.")
                results = []
            else:
                # Use prior conversation for search context when the query is a followup
                search_query = query
                if history_items and len(history_items) > 0:
                    # Use a combined query of the most recent relevant conversation + current query
                    # This helps with retrieving documents for follow-up questions
                    recent_item = history_items[0]
                    # Check if it's likely a follow-up by checking if it's short or has pronouns
                    follow_up_indicators = ["it", "this", "that", "they", "their", "these", "those"]
                    is_followup = len(query.split()) < 8 or any(word in query.lower().split() for word in follow_up_indicators)
                    
                    if is_followup:
                        # Create a more comprehensive search query using prior context
                        search_query = f"{recent_item['query']} {recent_item['response']} {query}"
                        logger.debug(f"Using enhanced search query for follow-up: {search_query[:100]}...")
                
                # Retrieve relevant documents using the potentially enhanced search query
                results = vector_store.similarity_search(
                    query=search_query,  # Use enhanced query for search
                    k=k,
                    filter=filter_params
                )
            
            retrieved_docs = RAG.format_retrieved_documents(results)
            history_context = RAG.format_history_responses(history_items)
            
            logger.debug(f"Retrieved {len(results)} documents and {len(history_items)} history items")
            
            # Load the specified model
            if model_manager.model is None or model_manager.current_model_name != model_name:
                model_manager.load_model(model_name)
            
            # Create a more contextual prompt for follow-up questions
            system_instruction = "You are an AI assistant that answers questions based on the provided documents and conversation history."
            
            # Add context awareness for follow-up questions
            if history_items and len(history_items) > 0:
                system_instruction += " For follow-up questions, remember to consider the context from previous exchanges, even if documents don't directly address the follow-up."
            
            rag_prompt = f"""{system_instruction}

{history_context}

DOCUMENTS:
{retrieved_docs}

USER QUESTION: {query}

Please provide a comprehensive answer based on the information in the documents and previous conversations. If the documents don't contain relevant information but your previous conversation does, use that context to inform your response. If you truly don't have enough information on the topic, state that clearly.

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
            
            # Save to history if enabled and history manager exists
            if use_history and app_history_manager is not None:
                try:
                    app_history_manager.save_response(
                        query=query,
                        response=response_text,
                        metadata={
                            "timestamp": time.time(),
                            "model": model_name,
                            "document_count": len(results)
                        }
                    )
                except Exception as history_save_error:
                    logger.warning(f"Failed to save to history: {history_save_error}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG response generation: {str(e)}", exc_info=True)
            raise
