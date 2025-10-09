"""
Data Manager for Training Dataset Handling
==========================================

This module manages training datasets from multiple sources and formats,
providing a unified interface for data loading, validation, and preparation.

Key Features:
- Support multiple input formats (JSON, JSONL, markdown, CSV)
- Handle both raw documents and prepared training data
- Validate data quality and completeness
- Create train/validation splits with stratification
- Generate dataset statistics and quality reports

Supported Data Sources:
- Raw methodology documents (markdown files)
- Prepared JSON instruction datasets
- Conversational logs (text files)
- CSV structured data
- Future: ChromaDB semantic chunks
"""

import json
import csv
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import pandas as pd

from .document_analyzer import DocumentAnalyzer, ExtractedContent
from .validation import DataValidator

logger = logging.getLogger(__name__)

@dataclass
class DatasetInfo:
    """Information about a training dataset."""
    name: str
    source_path: str
    format_type: str
    total_items: int
    train_items: int
    validation_items: int
    categories: List[str]
    avg_instruction_length: float
    avg_output_length: float
    quality_score: float
    created_at: str
    metadata: Dict[str, Any] = None

class DataManager:
    """
    Manages training datasets from multiple sources and formats.
    
    Provides unified interface for:
    - Loading data from various file formats
    - Processing raw documents with DocumentAnalyzer
    - Validating data quality and completeness
    - Creating train/validation splits
    - Generating comprehensive dataset statistics
    """
    
    def __init__(self, storage_dir: str = "training_data"):
        """
        Initialize DataManager with storage directory.
        
        Args:
            storage_dir: Directory to store processed training data
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.document_analyzer = DocumentAnalyzer()
        self.validator = DataValidator()
        
        # Create subdirectories
        (self.storage_dir / "raw").mkdir(exist_ok=True)
        (self.storage_dir / "processed").mkdir(exist_ok=True)
        (self.storage_dir / "splits").mkdir(exist_ok=True)
        (self.storage_dir / "metadata").mkdir(exist_ok=True)
        
        logger.info(f"DataManager initialized with storage: {self.storage_dir}")
    
    def load_dataset(self, 
                    source_path: str, 
                    dataset_name: str,
                    format_type: Optional[str] = None) -> List[ExtractedContent]:
        """
        Load dataset from file and convert to standard format.
        
        Args:
            source_path: Path to source file or directory
            dataset_name: Name for this dataset
            format_type: Format type ('json', 'markdown', 'csv', 'auto')
            
        Returns:
            List of ExtractedContent items
        """
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")
        
        # Auto-detect format if not specified
        if format_type is None or format_type == 'auto':
            format_type = self._detect_format(path)
        
        logger.info(f"Loading dataset '{dataset_name}' from {source_path} (format: {format_type})")
        
        try:
            if format_type == 'markdown':
                return self._load_markdown(path)
            elif format_type == 'json':
                return self._load_json(path)
            elif format_type == 'jsonl':
                return self._load_jsonl(path)
            elif format_type == 'csv':
                return self._load_csv(path)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
                
        except Exception as e:
            logger.error(f"Error loading dataset from {source_path}: {e}")
            raise
    
    def save_dataset(self, 
                    data: List[ExtractedContent], 
                    dataset_name: str,
                    format_type: str = 'json') -> str:
        """
        Save processed dataset to storage.
        
        Args:
            data: List of ExtractedContent items
            dataset_name: Name for this dataset
            format_type: Output format ('json', 'jsonl')
            
        Returns:
            Path to saved file
        """
        output_path = self.storage_dir / "processed" / f"{dataset_name}.{format_type}"
        
        try:
            if format_type == 'json':
                self._save_json(data, output_path)
            elif format_type == 'jsonl':
                self._save_jsonl(data, output_path)
            else:
                raise ValueError(f"Unsupported output format: {format_type}")
            
            logger.info(f"Saved dataset '{dataset_name}' to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving dataset {dataset_name}: {e}")
            raise
    
    def create_train_validation_split(self, 
                                     data: List[ExtractedContent],
                                     dataset_name: str,
                                     train_ratio: float = 0.8,
                                     stratify_by: str = 'category',
                                     random_seed: int = 42) -> Tuple[List[ExtractedContent], List[ExtractedContent]]:
        """
        Create train/validation split with optional stratification.
        
        Args:
            data: List of ExtractedContent items
            dataset_name: Name for this dataset
            train_ratio: Ratio of data for training (0.0-1.0)
            stratify_by: Field to stratify by ('category', 'difficulty', None)
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_data, validation_data)
        """
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be between 0.0 and 1.0")
        
        random.seed(random_seed)
        
        if stratify_by and len(data) > 10:  # Only stratify for larger datasets
            train_data, val_data = self._stratified_split(data, train_ratio, stratify_by)
        else:
            train_data, val_data = self._random_split(data, train_ratio)
        
        # Save splits
        train_path = self.storage_dir / "splits" / f"{dataset_name}_train.json"
        val_path = self.storage_dir / "splits" / f"{dataset_name}_validation.json"
        
        self._save_json(train_data, train_path)
        self._save_json(val_data, val_path)
        
        logger.info(f"Created train/validation split for '{dataset_name}': "
                   f"{len(train_data)} train, {len(val_data)} validation")
        
        return train_data, val_data
    
    def generate_dataset_info(self, 
                             data: List[ExtractedContent],
                             dataset_name: str,
                             source_path: str,
                             format_type: str) -> DatasetInfo:
        """
        Generate comprehensive dataset information and statistics.
        
        Args:
            data: List of ExtractedContent items
            dataset_name: Name of the dataset
            source_path: Original source path
            format_type: Source format type
            
        Returns:
            DatasetInfo object with comprehensive statistics
        """
        from datetime import datetime
        
        if not data:
            raise ValueError("Cannot generate info for empty dataset")
        
        # Basic statistics
        total_items = len(data)
        categories = list(set(item.category for item in data))
        
        # Calculate average lengths
        instruction_lengths = [len(item.instruction) for item in data]
        output_lengths = [len(item.output) for item in data]
        
        avg_instruction_length = sum(instruction_lengths) / len(instruction_lengths)
        avg_output_length = sum(output_lengths) / len(output_lengths)
        
        # Quality score (placeholder - can be enhanced)
        quality_score = self.validator.calculate_quality_score(data)
        
        # Create dataset info
        dataset_info = DatasetInfo(
            name=dataset_name,
            source_path=source_path,
            format_type=format_type,
            total_items=total_items,
            train_items=int(total_items * 0.8),  # Default split
            validation_items=int(total_items * 0.2),
            categories=categories,
            avg_instruction_length=avg_instruction_length,
            avg_output_length=avg_output_length,
            quality_score=quality_score,
            created_at=datetime.now().isoformat(),
            metadata={
                'instruction_length_range': (min(instruction_lengths), max(instruction_lengths)),
                'output_length_range': (min(output_lengths), max(output_lengths)),
                'category_distribution': {cat: sum(1 for item in data if item.category == cat) 
                                        for cat in categories},
                'difficulty_distribution': {diff: sum(1 for item in data if item.difficulty == diff)
                                          for diff in set(item.difficulty for item in data)},
                'tags_frequency': self._calculate_tag_frequency(data)
            }
        )
        
        # Save dataset info
        info_path = self.storage_dir / "metadata" / f"{dataset_name}_info.json"
        with open(info_path, 'w') as f:
            json.dump(asdict(dataset_info), f, indent=2, default=str)
        
        logger.info(f"Generated dataset info for '{dataset_name}': "
                   f"{total_items} items, {len(categories)} categories, "
                   f"quality score: {quality_score:.2f}")
        
        return dataset_info
    
    def validate_dataset(self, data: List[ExtractedContent]) -> Dict[str, Any]:
        """
        Validate dataset quality and completeness.
        
        Args:
            data: List of ExtractedContent items
            
        Returns:
            Validation report dictionary
        """
        return self.validator.validate_dataset(data)
    
    def list_datasets(self) -> List[str]:
        """List all processed datasets."""
        processed_dir = self.storage_dir / "processed"
        if not processed_dir.exists():
            return []
        
        datasets = []
        for file_path in processed_dir.glob("*.json"):
            datasets.append(file_path.stem)
        
        return sorted(datasets)
    
    def load_processed_dataset(self, dataset_name: str) -> List[ExtractedContent]:
        """Load previously processed dataset."""
        dataset_path = self.storage_dir / "processed" / f"{dataset_name}.json"
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Processed dataset not found: {dataset_name}")
        
        return self._load_json(dataset_path)
    
    def get_dataset_info(self, dataset_name: str) -> Optional[DatasetInfo]:
        """Get information about a processed dataset."""
        info_path = self.storage_dir / "metadata" / f"{dataset_name}_info.json"
        
        if not info_path.exists():
            return None
        
        with open(info_path, 'r') as f:
            info_dict = json.load(f)
        
        return DatasetInfo(**info_dict)
    
    # Private methods for format handling
    
    def _detect_format(self, path: Path) -> str:
        """Auto-detect file format based on extension."""
        suffix = path.suffix.lower()
        
        if suffix == '.md':
            return 'markdown'
        elif suffix == '.json':
            return 'json'
        elif suffix == '.jsonl':
            return 'jsonl'
        elif suffix == '.csv':
            return 'csv'
        elif suffix == '.txt':
            # Try to determine if it's structured data or raw text
            with open(path, 'r') as f:
                first_line = f.readline().strip()
                if first_line.startswith('{'):
                    return 'jsonl'
                else:
                    return 'markdown'  # Treat as raw text for now
        else:
            raise ValueError(f"Unknown file format: {suffix}")
    
    def _load_markdown(self, path: Path) -> List[ExtractedContent]:
        """Load data from markdown document using DocumentAnalyzer."""
        return self.document_analyzer.analyze_document(str(path))
    
    def _load_json(self, path: Path) -> List[ExtractedContent]:
        """Load data from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to ExtractedContent objects
        items = []
        for item in data:
            if isinstance(item, dict):
                # Handle different JSON formats
                if 'instruction' in item and 'output' in item:
                    # Direct format
                    items.append(ExtractedContent(
                        instruction=item['instruction'],
                        context=item.get('context', ''),
                        output=item['output'],
                        category=item.get('category', 'general'),
                        tags=item.get('tags', []),
                        difficulty=item.get('difficulty', 'intermediate'),
                        source_section=item.get('source_section', ''),
                        metadata=item.get('metadata', {})
                    ))
                elif 'conversations' in item:
                    # Conversational format
                    conversations = item['conversations']
                    human_msg = ""
                    for conv in conversations:
                        if conv.get('from') == 'human':
                            human_msg = conv['value']
                        elif conv.get('from') == 'assistant' and human_msg:
                            items.append(ExtractedContent(
                                instruction=human_msg,
                                context="Conversational interaction",
                                output=conv['value'],
                                category="conversation",
                                tags=["conversation"],
                                difficulty="intermediate"
                            ))
                            human_msg = ""
        
        return items
    
    def _load_jsonl(self, path: Path) -> List[ExtractedContent]:
        """Load data from JSONL file."""
        items = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item_data = json.loads(line)
                    # Similar processing as JSON
                    if 'instruction' in item_data and 'output' in item_data:
                        items.append(ExtractedContent(
                            instruction=item_data['instruction'],
                            context=item_data.get('context', ''),
                            output=item_data['output'],
                            category=item_data.get('category', 'general'),
                            tags=item_data.get('tags', []),
                            difficulty=item_data.get('difficulty', 'intermediate')
                        ))
        
        return items
    
    def _load_csv(self, path: Path) -> List[ExtractedContent]:
        """Load data from CSV file."""
        items = []
        
        df = pd.read_csv(path)
        required_columns = ['instruction', 'output']
        
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        
        for _, row in df.iterrows():
            items.append(ExtractedContent(
                instruction=str(row['instruction']),
                context=str(row.get('context', '')),
                output=str(row['output']),
                category=str(row.get('category', 'general')),
                tags=str(row.get('tags', '')).split(',') if row.get('tags') else [],
                difficulty=str(row.get('difficulty', 'intermediate'))
            ))
        
        return items
    
    def _save_json(self, data: List[ExtractedContent], path: Path) -> None:
        """Save data to JSON file."""
        json_data = [asdict(item) for item in data]
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    def _save_jsonl(self, data: List[ExtractedContent], path: Path) -> None:
        """Save data to JSONL file."""
        with open(path, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(asdict(item), f, ensure_ascii=False)
                f.write('\n')
    
    def _stratified_split(self, 
                         data: List[ExtractedContent], 
                         train_ratio: float, 
                         stratify_by: str) -> Tuple[List[ExtractedContent], List[ExtractedContent]]:
        """Create stratified split by specified field."""
        # Group by stratification field
        groups = {}
        for item in data:
            key = getattr(item, stratify_by, 'unknown')
            if key not in groups:
                groups[key] = []
            groups[key].append(item)
        
        train_data = []
        val_data = []
        
        # Split each group proportionally
        for group_items in groups.values():
            random.shuffle(group_items)
            split_idx = int(len(group_items) * train_ratio)
            train_data.extend(group_items[:split_idx])
            val_data.extend(group_items[split_idx:])
        
        # Shuffle the final datasets
        random.shuffle(train_data)
        random.shuffle(val_data)
        
        return train_data, val_data
    
    def _random_split(self, 
                     data: List[ExtractedContent], 
                     train_ratio: float) -> Tuple[List[ExtractedContent], List[ExtractedContent]]:
        """Create random split."""
        shuffled_data = data.copy()
        random.shuffle(shuffled_data)
        
        split_idx = int(len(shuffled_data) * train_ratio)
        train_data = shuffled_data[:split_idx]
        val_data = shuffled_data[split_idx:]
        
        return train_data, val_data
    
    def _calculate_tag_frequency(self, data: List[ExtractedContent]) -> Dict[str, int]:
        """Calculate frequency of tags across dataset."""
        tag_freq = {}
        
        for item in data:
            for tag in item.tags:
                tag_freq[tag] = tag_freq.get(tag, 0) + 1
        
        # Sort by frequency
        return dict(sorted(tag_freq.items(), key=lambda x: x[1], reverse=True))