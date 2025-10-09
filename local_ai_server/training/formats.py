"""
Format Conversion Utilities
===========================

This module provides utilities for converting between different
training data formats and representations.

Key Features:
- Convert between ExtractedContent and various standard formats
- Support for popular training data formats (Alpaca, ShareGPT, etc.)
- Bidirectional conversion capabilities
- Format validation and normalization
- Export to common ML training formats
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import asdict

from .document_analyzer import ExtractedContent

logger = logging.getLogger(__name__)

class FormatConverter:
    """
    Converts training data between different formats.
    
    Supported conversions:
    - ExtractedContent ↔ Alpaca format
    - ExtractedContent ↔ ShareGPT format
    - ExtractedContent ↔ OpenAI format
    - ExtractedContent ↔ HuggingFace datasets format
    """
    
    def __init__(self):
        """Initialize the format converter."""
        self.supported_formats = [
            'alpaca', 'sharegpt', 'openai', 'huggingface', 'jsonl', 'csv'
        ]
        
        logger.info("FormatConverter initialized")
    
    def to_alpaca_format(self, data: List[ExtractedContent]) -> List[Dict[str, Any]]:
        """
        Convert to Alpaca instruction format.
        
        Args:
            data: List of ExtractedContent items
            
        Returns:
            List of Alpaca format dictionaries
        """
        alpaca_data = []
        
        for item in data:
            alpaca_item = {
                "instruction": item.instruction,
                "input": item.context or "",
                "output": item.output
            }
            
            # Add metadata as additional fields
            if item.category:
                alpaca_item["category"] = item.category
            if item.tags:
                alpaca_item["tags"] = item.tags
            if item.difficulty:
                alpaca_item["difficulty"] = item.difficulty
            
            alpaca_data.append(alpaca_item)
        
        logger.info(f"Converted {len(data)} items to Alpaca format")
        return alpaca_data
    
    def from_alpaca_format(self, alpaca_data: List[Dict[str, Any]]) -> List[ExtractedContent]:
        """
        Convert from Alpaca format to ExtractedContent.
        
        Args:
            alpaca_data: List of Alpaca format dictionaries
            
        Returns:
            List of ExtractedContent items
        """
        extracted_data = []
        
        for item in alpaca_data:
            extracted_item = ExtractedContent(
                instruction=item.get("instruction", ""),
                context=item.get("input", ""),
                output=item.get("output", ""),
                category=item.get("category", "general"),
                tags=item.get("tags", []),
                difficulty=item.get("difficulty", "intermediate"),
                source_section=item.get("source_section", ""),
                metadata=item.get("metadata", {})
            )
            
            extracted_data.append(extracted_item)
        
        logger.info(f"Converted {len(alpaca_data)} items from Alpaca format")
        return extracted_data
    
    def to_sharegpt_format(self, data: List[ExtractedContent]) -> List[Dict[str, Any]]:
        """
        Convert to ShareGPT conversation format.
        
        Args:
            data: List of ExtractedContent items
            
        Returns:
            List of ShareGPT format dictionaries
        """
        sharegpt_data = []
        
        for item in data:
            conversation = {
                "conversations": [
                    {
                        "from": "human",
                        "value": item.instruction
                    },
                    {
                        "from": "gpt",
                        "value": item.output
                    }
                ]
            }
            
            # Add system message if context exists
            if item.context:
                conversation["conversations"].insert(0, {
                    "from": "system",
                    "value": item.context
                })
            
            # Add metadata
            conversation["metadata"] = {
                "category": item.category,
                "tags": item.tags,
                "difficulty": item.difficulty,
                "source_section": item.source_section
            }
            
            sharegpt_data.append(conversation)
        
        logger.info(f"Converted {len(data)} items to ShareGPT format")
        return sharegpt_data
    
    def from_sharegpt_format(self, sharegpt_data: List[Dict[str, Any]]) -> List[ExtractedContent]:
        """
        Convert from ShareGPT format to ExtractedContent.
        
        Args:
            sharegpt_data: List of ShareGPT format dictionaries
            
        Returns:
            List of ExtractedContent items
        """
        extracted_data = []
        
        for item in sharegpt_data:
            conversations = item.get("conversations", [])
            
            instruction = ""
            context = ""
            output = ""
            
            # Parse conversation messages
            for conv in conversations:
                if conv.get("from") == "system":
                    context = conv.get("value", "")
                elif conv.get("from") == "human":
                    instruction = conv.get("value", "")
                elif conv.get("from") in ["gpt", "assistant"]:
                    output = conv.get("value", "")
            
            if instruction and output:
                metadata = item.get("metadata", {})
                
                extracted_item = ExtractedContent(
                    instruction=instruction,
                    context=context,
                    output=output,
                    category=metadata.get("category", "conversation"),
                    tags=metadata.get("tags", []),
                    difficulty=metadata.get("difficulty", "intermediate"),
                    source_section=metadata.get("source_section", ""),
                    metadata=metadata
                )
                
                extracted_data.append(extracted_item)
        
        logger.info(f"Converted {len(sharegpt_data)} items from ShareGPT format")
        return extracted_data
    
    def to_openai_format(self, data: List[ExtractedContent]) -> List[Dict[str, Any]]:
        """
        Convert to OpenAI fine-tuning format.
        
        Args:
            data: List of ExtractedContent items
            
        Returns:
            List of OpenAI format dictionaries
        """
        openai_data = []
        
        for item in data:
            messages = []
            
            # Add system message based on category
            system_message = self._generate_system_message(item)
            if system_message:
                messages.append({
                    "role": "system",
                    "content": system_message
                })
            
            # Add user message
            messages.append({
                "role": "user", 
                "content": item.instruction
            })
            
            # Add assistant response
            messages.append({
                "role": "assistant",
                "content": item.output
            })
            
            openai_item = {
                "messages": messages,
                "metadata": {
                    "category": item.category,
                    "tags": item.tags,
                    "difficulty": item.difficulty
                }
            }
            
            openai_data.append(openai_item)
        
        logger.info(f"Converted {len(data)} items to OpenAI format")
        return openai_data
    
    def to_huggingface_format(self, data: List[ExtractedContent]) -> Dict[str, List[Any]]:
        """
        Convert to HuggingFace datasets format.
        
        Args:
            data: List of ExtractedContent items
            
        Returns:
            Dictionary with lists for each field
        """
        hf_data = {
            "instruction": [],
            "context": [],
            "output": [],
            "category": [],
            "tags": [],
            "difficulty": [],
            "source_section": []
        }
        
        for item in data:
            hf_data["instruction"].append(item.instruction)
            hf_data["context"].append(item.context or "")
            hf_data["output"].append(item.output)
            hf_data["category"].append(item.category)
            hf_data["tags"].append(item.tags)
            hf_data["difficulty"].append(item.difficulty)
            hf_data["source_section"].append(item.source_section or "")
        
        logger.info(f"Converted {len(data)} items to HuggingFace format")
        return hf_data
    
    def to_csv_format(self, data: List[ExtractedContent]) -> List[Dict[str, str]]:
        """
        Convert to CSV-compatible format.
        
        Args:
            data: List of ExtractedContent items
            
        Returns:
            List of dictionaries suitable for CSV export
        """
        csv_data = []
        
        for item in data:
            csv_item = {
                "instruction": item.instruction,
                "context": item.context or "",
                "output": item.output,
                "category": item.category,
                "tags": ",".join(item.tags) if item.tags else "",
                "difficulty": item.difficulty,
                "source_section": item.source_section or ""
            }
            
            csv_data.append(csv_item)
        
        logger.info(f"Converted {len(data)} items to CSV format")
        return csv_data
    
    def export_format(self, 
                     data: List[ExtractedContent], 
                     format_type: str, 
                     output_path: str) -> str:
        """
        Export data to specified format and save to file.
        
        Args:
            data: List of ExtractedContent items
            format_type: Target format ('alpaca', 'sharegpt', 'openai', etc.)
            output_path: Path to save the exported data
            
        Returns:
            Path to the saved file
        """
        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format_type}. "
                           f"Supported formats: {self.supported_formats}")
        
        try:
            if format_type == "alpaca":
                converted_data = self.to_alpaca_format(data)
            elif format_type == "sharegpt":
                converted_data = self.to_sharegpt_format(data)
            elif format_type == "openai":
                converted_data = self.to_openai_format(data)
            elif format_type == "huggingface":
                converted_data = self.to_huggingface_format(data)
            elif format_type == "csv":
                converted_data = self.to_csv_format(data)
                self._save_csv(converted_data, output_path)
                return output_path
            else:
                # Default to JSON for unknown formats
                converted_data = [asdict(item) for item in data]
            
            # Save as JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(converted_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported {len(data)} items to {format_type} format at {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting to {format_type} format: {e}")
            raise
    
    def import_format(self, 
                     file_path: str, 
                     format_type: str) -> List[ExtractedContent]:
        """
        Import data from specified format file.
        
        Args:
            file_path: Path to the file to import
            format_type: Format of the source file
            
        Returns:
            List of ExtractedContent items
        """
        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format_type}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if format_type == "csv":
                    import pandas as pd
                    df = pd.read_csv(file_path)
                    raw_data = df.to_dict('records')
                    return self._from_csv_format(raw_data)
                else:
                    raw_data = json.load(f)
            
            if format_type == "alpaca":
                return self.from_alpaca_format(raw_data)
            elif format_type == "sharegpt":
                return self.from_sharegpt_format(raw_data)
            elif format_type == "openai":
                return self._from_openai_format(raw_data)
            elif format_type == "huggingface":
                return self._from_huggingface_format(raw_data)
            else:
                # Try to parse as ExtractedContent directly
                return self._from_generic_format(raw_data)
                
        except Exception as e:
            logger.error(f"Error importing from {format_type} format: {e}")
            raise
    
    def validate_format(self, data: Any, format_type: str) -> bool:
        """
        Validate that data conforms to specified format.
        
        Args:
            data: Data to validate
            format_type: Expected format
            
        Returns:
            True if data is valid for the format
        """
        try:
            if format_type == "alpaca":
                return self._validate_alpaca_format(data)
            elif format_type == "sharegpt":
                return self._validate_sharegpt_format(data)
            elif format_type == "openai":
                return self._validate_openai_format(data)
            else:
                return True  # Default to valid for unknown formats
                
        except Exception:
            return False
    
    # Private helper methods
    
    def _generate_system_message(self, item: ExtractedContent) -> str:
        """Generate appropriate system message based on item category."""
        system_messages = {
            'philosophy': "You are a collaborative intelligence assistant who understands systematic methodologies.",
            'methodology': "You are a systematic methodology expert who provides structured guidance.",
            'process': "You are a process guidance specialist who helps with systematic procedures.",
            'crisis_management': "You are a crisis prevention assistant using collaborative approaches.",
            'conversation': "You are a collaborative intelligence assistant with systematic communication style.",
            'example': "You are a practical implementation guide providing concrete examples.",
            'implementation': "You are an implementation specialist for systematic methodologies."
        }
        
        return system_messages.get(item.category, 
                                 "You are a helpful assistant trained in systematic methodologies.")
    
    def _from_openai_format(self, data: List[Dict[str, Any]]) -> List[ExtractedContent]:
        """Convert from OpenAI format."""
        extracted_data = []
        
        for item in data:
            messages = item.get("messages", [])
            
            instruction = ""
            context = ""
            output = ""
            
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content", "")
                
                if role == "system":
                    context = content
                elif role == "user":
                    instruction = content
                elif role == "assistant":
                    output = content
            
            if instruction and output:
                metadata = item.get("metadata", {})
                
                extracted_item = ExtractedContent(
                    instruction=instruction,
                    context=context,
                    output=output,
                    category=metadata.get("category", "general"),
                    tags=metadata.get("tags", []),
                    difficulty=metadata.get("difficulty", "intermediate")
                )
                
                extracted_data.append(extracted_item)
        
        return extracted_data
    
    def _from_huggingface_format(self, data: Dict[str, List[Any]]) -> List[ExtractedContent]:
        """Convert from HuggingFace datasets format."""
        extracted_data = []
        
        length = len(data.get("instruction", []))
        
        for i in range(length):
            extracted_item = ExtractedContent(
                instruction=data["instruction"][i],
                context=data.get("context", [""] * length)[i],
                output=data["output"][i],
                category=data.get("category", ["general"] * length)[i],
                tags=data.get("tags", [[] for _ in range(length)])[i],
                difficulty=data.get("difficulty", ["intermediate"] * length)[i],
                source_section=data.get("source_section", [""] * length)[i]
            )
            
            extracted_data.append(extracted_item)
        
        return extracted_data
    
    def _from_csv_format(self, data: List[Dict[str, str]]) -> List[ExtractedContent]:
        """Convert from CSV format."""
        extracted_data = []
        
        for item in data:
            tags = item.get("tags", "").split(",") if item.get("tags") else []
            tags = [tag.strip() for tag in tags if tag.strip()]
            
            extracted_item = ExtractedContent(
                instruction=item.get("instruction", ""),
                context=item.get("context", ""),
                output=item.get("output", ""),
                category=item.get("category", "general"),
                tags=tags,
                difficulty=item.get("difficulty", "intermediate"),
                source_section=item.get("source_section", "")
            )
            
            extracted_data.append(extracted_item)
        
        return extracted_data
    
    def _from_generic_format(self, data: List[Dict[str, Any]]) -> List[ExtractedContent]:
        """Convert from generic format."""
        extracted_data = []
        
        for item in data:
            extracted_item = ExtractedContent(
                instruction=item.get("instruction", ""),
                context=item.get("context", ""),
                output=item.get("output", ""),
                category=item.get("category", "general"),
                tags=item.get("tags", []),
                difficulty=item.get("difficulty", "intermediate"),
                source_section=item.get("source_section", ""),
                metadata=item.get("metadata", {})
            )
            
            extracted_data.append(extracted_item)
        
        return extracted_data
    
    def _save_csv(self, data: List[Dict[str, str]], output_path: str) -> None:
        """Save data as CSV file."""
        import pandas as pd
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8')
    
    def _validate_alpaca_format(self, data: List[Dict[str, Any]]) -> bool:
        """Validate Alpaca format."""
        if not isinstance(data, list):
            return False
        
        for item in data:
            if not isinstance(item, dict):
                return False
            
            required_fields = ["instruction", "output"]
            if not all(field in item for field in required_fields):
                return False
        
        return True
    
    def _validate_sharegpt_format(self, data: List[Dict[str, Any]]) -> bool:
        """Validate ShareGPT format."""
        if not isinstance(data, list):
            return False
        
        for item in data:
            if not isinstance(item, dict):
                return False
            
            if "conversations" not in item:
                return False
            
            conversations = item["conversations"]
            if not isinstance(conversations, list):
                return False
        
        return True
    
    def _validate_openai_format(self, data: List[Dict[str, Any]]) -> bool:
        """Validate OpenAI format."""
        if not isinstance(data, list):
            return False
        
        for item in data:
            if not isinstance(item, dict):
                return False
            
            if "messages" not in item:
                return False
            
            messages = item["messages"]
            if not isinstance(messages, list):
                return False
        
        return True