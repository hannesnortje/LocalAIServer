"""
Data Preprocessing for CodeLlama Training Format
================================================

This module converts training data into optimal formats for CodeLlama
and other instruction-following models, with support for various
prompt templates and conversation formats.

Key Features:
- Convert ExtractedContent to ChatML format
- Apply CodeLlama-specific prompt templates
- Handle conversation formatting and special tokens
- Support multiple model architectures and formats
- Optimize for training efficiency and quality

Supported Formats:
- ChatML (recommended for CodeLlama)
- Alpaca format
- Vicuna conversation format
- Custom instruction templates
"""

import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

from .document_analyzer import ExtractedContent

logger = logging.getLogger(__name__)

class PromptFormat(Enum):
    """Supported prompt formats for different models."""
    CHATML = "chatml"
    ALPACA = "alpaca"
    VICUNA = "vicuna"
    CODELLAMA = "codellama"
    CUSTOM = "custom"

@dataclass
class ProcessedTrainingItem:
    """Processed training item ready for model training."""
    text: str
    labels: Optional[str] = None
    input_ids: Optional[List[int]] = None
    attention_mask: Optional[List[int]] = None
    metadata: Dict[str, Any] = None

class DataPreprocessor:
    """
    Preprocesses training data for optimal CodeLlama training.
    
    Handles:
    - Prompt format conversion and optimization
    - Special token insertion and management
    - Text length normalization and truncation
    - Conversation threading and context management
    - Training-specific formatting requirements
    """
    
    # Token limits for different contexts
    MAX_SEQUENCE_LENGTH = 4096
    RECOMMENDED_LENGTH = 2048
    MIN_TRAINING_LENGTH = 50
    
    # Special tokens
    SYSTEM_TOKEN = "<|system|>"
    USER_TOKEN = "<|user|>"
    ASSISTANT_TOKEN = "<|assistant|>"
    END_TOKEN = "<|end|>"
    
    def __init__(self, 
                 prompt_format: PromptFormat = PromptFormat.CHATML,
                 max_length: int = 2048,
                 include_system_prompt: bool = True):
        """
        Initialize preprocessor with format and settings.
        
        Args:
            prompt_format: Format to use for prompts
            max_length: Maximum sequence length
            include_system_prompt: Whether to include system prompts
        """
        self.prompt_format = prompt_format
        self.max_length = max_length
        self.include_system_prompt = include_system_prompt
        
        # Load prompt templates
        self.templates = self._load_prompt_templates()
        
        logger.info(f"DataPreprocessor initialized: format={prompt_format.value}, "
                   f"max_length={max_length}")
    
    def process_dataset(self, data: List[ExtractedContent]) -> List[ProcessedTrainingItem]:
        """
        Process entire dataset for training.
        
        Args:
            data: List of ExtractedContent items
            
        Returns:
            List of ProcessedTrainingItem objects ready for training
        """
        if not data:
            logger.warning("Empty dataset provided for processing")
            return []
        
        logger.info(f"Processing {len(data)} items for training format: {self.prompt_format.value}")
        
        processed_items = []
        
        for i, item in enumerate(data):
            try:
                processed = self.process_item(item)
                if processed:
                    processed_items.append(processed)
                else:
                    logger.warning(f"Failed to process item {i}: {item.instruction[:50]}...")
                    
            except Exception as e:
                logger.error(f"Error processing item {i}: {e}")
                continue
        
        logger.info(f"Successfully processed {len(processed_items)}/{len(data)} items")
        return processed_items
    
    def process_item(self, item: ExtractedContent) -> Optional[ProcessedTrainingItem]:
        """
        Process a single training item.
        
        Args:
            item: ExtractedContent to process
            
        Returns:
            ProcessedTrainingItem or None if processing fails
        """
        if not item.instruction or not item.output:
            logger.warning("Item missing instruction or output")
            return None
        
        try:
            # Generate system prompt based on category and context
            system_prompt = self._generate_system_prompt(item)
            
            # Format according to selected template
            if self.prompt_format == PromptFormat.CHATML:
                text = self._format_chatml(item, system_prompt)
            elif self.prompt_format == PromptFormat.ALPACA:
                text = self._format_alpaca(item, system_prompt)
            elif self.prompt_format == PromptFormat.VICUNA:
                text = self._format_vicuna(item, system_prompt)
            elif self.prompt_format == PromptFormat.CODELLAMA:
                text = self._format_codellama(item, system_prompt)
            else:
                text = self._format_custom(item, system_prompt)
            
            # Validate and truncate if necessary
            text = self._validate_and_truncate(text)
            
            if not text:
                return None
            
            # Create processed item
            processed = ProcessedTrainingItem(
                text=text,
                metadata={
                    'category': item.category,
                    'difficulty': item.difficulty,
                    'tags': item.tags,
                    'source_section': item.source_section,
                    'original_length': len(item.instruction) + len(item.output),
                    'processed_length': len(text),
                    'format': self.prompt_format.value
                }
            )
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing item: {e}")
            return None
    
    def _generate_system_prompt(self, item: ExtractedContent) -> str:
        """Generate appropriate system prompt based on item category and context."""
        if not self.include_system_prompt:
            return ""
        
        # Category-specific system prompts
        category_prompts = {
            'philosophy': "You are a collaborative intelligence assistant who understands the '42 = FOR TWO' philosophy. You provide guidance on systematic methodologies and collaborative approaches to complex problems.",
            
            'methodology': "You are a systematic methodology expert who helps implement collaborative intelligence protocols. You provide clear, structured guidance based on proven systematic approaches.",
            
            'process': "You are a process guidance specialist who helps users apply systematic procedures and crisis prevention protocols. You emphasize collaborative approaches and structured problem-solving.",
            
            'crisis_management': "You are a crisis prevention and management assistant. You help users apply the '42 = FOR TWO' collaborative approach when facing overwhelming problems or complex decisions.",
            
            'conversation': "You are a collaborative intelligence assistant who communicates using the systematic yet enthusiastic style of the '42 = FOR TWO' methodology. You provide guidance while maintaining the collaborative spirit.",
            
            'example': "You are a practical implementation guide who provides concrete examples of systematic methodologies and collaborative intelligence principles in action.",
            
            'core_principle': "You are a foundational philosophy guide who explains core principles of collaborative intelligence and the '42 = FOR TWO' methodology with clarity and practical application.",
            
            'implementation': "You are an implementation specialist who helps translate methodologies and principles into practical, actionable steps using systematic approaches."
        }
        
        # Use category-specific prompt or default
        base_prompt = category_prompts.get(
            item.category, 
            "You are a helpful coding assistant trained in systematic methodologies and collaborative intelligence. You provide clear, structured guidance following the '42 = FOR TWO' principle."
        )
        
        # Add context-specific enhancement if available
        if item.context:
            enhanced_prompt = f"{base_prompt} Context: {item.context}"
            return enhanced_prompt
        
        return base_prompt
    
    def _format_chatml(self, item: ExtractedContent, system_prompt: str) -> str:
        """Format using ChatML format (recommended for CodeLlama)."""
        parts = []
        
        if system_prompt:
            parts.append(f"{self.SYSTEM_TOKEN}\n{system_prompt}{self.END_TOKEN}")
        
        parts.append(f"{self.USER_TOKEN}\n{item.instruction}{self.END_TOKEN}")
        parts.append(f"{self.ASSISTANT_TOKEN}\n{item.output}{self.END_TOKEN}")
        
        return "\n".join(parts)
    
    def _format_alpaca(self, item: ExtractedContent, system_prompt: str) -> str:
        """Format using Alpaca instruction format."""
        template = self.templates['alpaca']
        
        instruction = item.instruction
        input_text = item.context if item.context else ""
        output = item.output
        
        if system_prompt and input_text:
            input_text = f"System: {system_prompt}\n\nContext: {input_text}"
        elif system_prompt:
            input_text = f"System: {system_prompt}"
        
        formatted = template.format(
            instruction=instruction,
            input=input_text,
            output=output
        )
        
        return formatted
    
    def _format_vicuna(self, item: ExtractedContent, system_prompt: str) -> str:
        """Format using Vicuna conversation format."""
        parts = []
        
        if system_prompt:
            parts.append(f"SYSTEM: {system_prompt}")
        
        parts.append(f"USER: {item.instruction}")
        parts.append(f"ASSISTANT: {item.output}")
        
        return "\n\n".join(parts)
    
    def _format_codellama(self, item: ExtractedContent, system_prompt: str) -> str:
        """Format specifically optimized for CodeLlama."""
        # CodeLlama works well with structured prompts
        parts = []
        
        if system_prompt:
            parts.append(f"# System\n{system_prompt}\n")
        
        # Add category and tags as metadata for better training
        if item.category:
            parts.append(f"# Category: {item.category}")
        
        if item.tags:
            parts.append(f"# Tags: {', '.join(item.tags)}")
        
        if item.difficulty:
            parts.append(f"# Difficulty: {item.difficulty}")
        
        parts.append(f"\n# Instruction\n{item.instruction}\n")
        
        if item.context:
            parts.append(f"# Context\n{item.context}\n")
        
        parts.append(f"# Response\n{item.output}")
        
        return "\n".join(parts)
    
    def _format_custom(self, item: ExtractedContent, system_prompt: str) -> str:
        """Format using custom template."""
        template = self.templates.get('custom', self.templates['chatml'])
        
        return template.format(
            system=system_prompt,
            instruction=item.instruction,
            context=item.context or "",
            output=item.output,
            category=item.category or "",
            tags=", ".join(item.tags) if item.tags else "",
            difficulty=item.difficulty or ""
        )
    
    def _validate_and_truncate(self, text: str) -> Optional[str]:
        """Validate text and truncate if necessary."""
        if not text or len(text.strip()) < self.MIN_TRAINING_LENGTH:
            logger.warning(f"Text too short: {len(text)} chars")
            return None
        
        # Truncate if too long
        if len(text) > self.max_length:
            logger.warning(f"Truncating text from {len(text)} to {self.max_length} chars")
            
            # Try to truncate intelligently at sentence boundaries
            truncated = self._intelligent_truncate(text, self.max_length)
            
            if len(truncated.strip()) < self.MIN_TRAINING_LENGTH:
                logger.warning("Text too short after truncation")
                return None
            
            return truncated
        
        return text
    
    def _intelligent_truncate(self, text: str, max_length: int) -> str:
        """Truncate text intelligently at sentence or paragraph boundaries."""
        if len(text) <= max_length:
            return text
        
        # Try to find good breaking points
        break_points = ['. ', '\n\n', '\n', '. ']
        
        for break_point in break_points:
            # Find the last occurrence of break point within limit
            truncate_pos = text.rfind(break_point, 0, max_length - 100)  # Leave some buffer
            
            if truncate_pos > max_length // 2:  # Don't truncate too aggressively
                return text[:truncate_pos + len(break_point)].strip()
        
        # If no good break point found, truncate at word boundary
        words = text[:max_length].split()
        return ' '.join(words[:-1])  # Remove last potentially incomplete word
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates for different formats."""
        templates = {
            'chatml': "{system}\n{user_token}\n{instruction}{end_token}\n{assistant_token}\n{output}{end_token}",
            
            'alpaca': """Below is an instruction that describes a task{input_desc}. Write a response that appropriately completes the request.

### Instruction:
{instruction}
{input_section}
### Response:
{output}""",
            
            'vicuna': """A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: {instruction}
ASSISTANT: {output}""",
            
            'codellama': """# System
{system}

# Instruction
{instruction}

# Response
{output}""",
            
            'custom': """System: {system}

Category: {category}
Tags: {tags}
Difficulty: {difficulty}

Instruction: {instruction}
Context: {context}

Response: {output}"""
        }
        
        return templates
    
    def save_processed_dataset(self, 
                              processed_data: List[ProcessedTrainingItem],
                              output_path: str,
                              format_type: str = 'jsonl') -> str:
        """
        Save processed dataset to file.
        
        Args:
            processed_data: List of processed training items
            output_path: Path to save the data
            format_type: Format to save ('jsonl', 'json', 'txt')
            
        Returns:
            Path to saved file
        """
        try:
            if format_type == 'jsonl':
                self._save_jsonl(processed_data, output_path)
            elif format_type == 'json':
                self._save_json(processed_data, output_path)
            elif format_type == 'txt':
                self._save_txt(processed_data, output_path)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            logger.info(f"Saved {len(processed_data)} processed items to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving processed dataset: {e}")
            raise
    
    def _save_jsonl(self, data: List[ProcessedTrainingItem], path: str) -> None:
        """Save data in JSONL format (one JSON object per line)."""
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                json_obj = {
                    'text': item.text,
                    'metadata': item.metadata
                }
                json.dump(json_obj, f, ensure_ascii=False)
                f.write('\n')
    
    def _save_json(self, data: List[ProcessedTrainingItem], path: str) -> None:
        """Save data in JSON format."""
        json_data = []
        for item in data:
            json_data.append({
                'text': item.text,
                'metadata': item.metadata
            })
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    def _save_txt(self, data: List[ProcessedTrainingItem], path: str) -> None:
        """Save data as plain text with separators."""
        with open(path, 'w', encoding='utf-8') as f:
            for i, item in enumerate(data):
                f.write(f"=== Training Example {i+1} ===\n")
                f.write(item.text)
                f.write("\n\n" + "="*50 + "\n\n")
    
    def get_statistics(self, processed_data: List[ProcessedTrainingItem]) -> Dict[str, Any]:
        """Generate statistics for processed dataset."""
        if not processed_data:
            return {}
        
        lengths = [len(item.text) for item in processed_data]
        categories = [item.metadata.get('category', 'unknown') for item in processed_data]
        difficulties = [item.metadata.get('difficulty', 'unknown') for item in processed_data]
        
        from collections import Counter
        
        stats = {
            'total_items': len(processed_data),
            'format': self.prompt_format.value,
            'text_length': {
                'min': min(lengths),
                'max': max(lengths),
                'avg': sum(lengths) / len(lengths),
                'median': sorted(lengths)[len(lengths)//2]
            },
            'category_distribution': dict(Counter(categories)),
            'difficulty_distribution': dict(Counter(difficulties)),
            'avg_compression_ratio': self._calculate_compression_ratio(processed_data)
        }
        
        return stats
    
    def _calculate_compression_ratio(self, processed_data: List[ProcessedTrainingItem]) -> float:
        """Calculate average compression ratio from original to processed text."""
        ratios = []
        
        for item in processed_data:
            original_length = item.metadata.get('original_length', 0)
            processed_length = item.metadata.get('processed_length', len(item.text))
            
            if original_length > 0:
                ratio = processed_length / original_length
                ratios.append(ratio)
        
        return sum(ratios) / len(ratios) if ratios else 1.0