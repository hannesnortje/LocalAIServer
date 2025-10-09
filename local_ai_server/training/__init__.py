"""
Training Data Pipeline for QLoRA Fine-tuning
============================================

This module provides comprehensive tools for processing and preparing training data
for QLoRA fine-tuning of coding models.

Key Features:
- Multiple data format support (instruction, conversational, principles)
- Document analysis and knowledge extraction from methodology documents
- Automatic train/validation splitting with quality validation
- Data preprocessing for CodeLlama format optimization
- ChromaDB integration framework for dynamic training data

Philosophy: "42 = FOR TWO" - Collaborative intelligence between
strategic guidance (TRON) and systematic execution (AI) produces
superior training data and methodology extraction.

Components:
- DataManager: Handle multiple input formats and dataset management
- DocumentAnalyzer: Extract training data from methodology documents
- DataPreprocessor: Convert data to optimal CodeLlama training format
- DataValidator: Ensure quality and completeness of training datasets
- FormatConverter: Transform between different data representation formats
- ChromaAdapter: Framework for future ChromaDB integration
"""

from .data_manager import DataManager
from .document_analyzer import DocumentAnalyzer
from .preprocessing import DataPreprocessor, PromptFormat
from .validation import DataValidator
from .formats import FormatConverter
from .chroma_adapter import ChromaAdapter
from .qlora import QLoRATrainer, LoRAConfig, TrainingConfig

__version__ = "0.1.0"
__all__ = [
    'DataManager',
    'DocumentAnalyzer',
    'DataPreprocessor', 
    'PromptFormat',
    'DataValidator',
    'FormatConverter',
    'ChromaAdapter',
    'QLoRATrainer',
    'LoRAConfig', 
    'TrainingConfig'
]

# Training data format constants
INSTRUCTION_FORMAT = "instruction"
CONVERSATIONAL_FORMAT = "conversational"
COMPLETION_FORMAT = "completion"
METHODOLOGY_FORMAT = "methodology"

# Supported input file types
SUPPORTED_EXTENSIONS = ['.md', '.json', '.jsonl', '.txt', '.csv']

# Quality thresholds
MIN_INSTRUCTION_LENGTH = 10
MIN_OUTPUT_LENGTH = 20
MAX_SEQUENCE_LENGTH = 4096
MIN_DATASET_SIZE = 5