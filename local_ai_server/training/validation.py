"""
Data Validation for Training Datasets
=====================================

This module provides comprehensive validation for training datasets,
ensuring data quality, completeness, and suitability for QLoRA training.

Key Features:
- Validate instruction-response pair quality
- Check data completeness and format consistency
- Identify potential issues and provide recommendations
- Calculate quality scores and metrics
- Generate detailed validation reports

Quality Checks:
- Instruction and output length validation
- Content quality assessment
- Category and tag consistency
- Duplicate detection
- Format compliance verification
"""

import re
import logging
from typing import Dict, List, Any, Set, Tuple
from collections import Counter
from dataclasses import dataclass

from .document_analyzer import ExtractedContent

logger = logging.getLogger(__name__)

@dataclass
class ValidationIssue:
    """Represents a validation issue found in the dataset."""
    item_index: int
    issue_type: str
    severity: str  # 'error', 'warning', 'info'
    message: str
    suggestion: str = ""

@dataclass
class ValidationReport:
    """Comprehensive validation report for a dataset."""
    total_items: int
    valid_items: int
    issues: List[ValidationIssue]
    quality_score: float
    statistics: Dict[str, Any]
    recommendations: List[str]

class DataValidator:
    """
    Validates training datasets for quality and completeness.
    
    Performs comprehensive checks on:
    - Data format and structure
    - Content quality and appropriateness
    - Length constraints and requirements
    - Category consistency and distribution
    - Duplicate detection and similarity
    """
    
    # Quality thresholds
    MIN_INSTRUCTION_LENGTH = 10
    MAX_INSTRUCTION_LENGTH = 2000
    MIN_OUTPUT_LENGTH = 20
    MAX_OUTPUT_LENGTH = 8000
    MIN_CONTEXT_LENGTH = 0
    MAX_CONTEXT_LENGTH = 1000
    
    # Content quality patterns
    POOR_INSTRUCTION_PATTERNS = [
        r'^(what|how|why|when|where)\?*$',  # Single word questions
        r'^(yes|no|ok|sure)\.?$',  # Single word responses
        r'^.{1,5}$',  # Too short
    ]
    
    POOR_OUTPUT_PATTERNS = [
        r'^(yes|no|ok|sure)\.?$',  # Single word responses
        r'^.{1,10}$',  # Too short
        r'^(i don\'t know|not sure|maybe)\.?$',  # Unhelpful responses
    ]
    
    def __init__(self):
        """Initialize the data validator with default settings."""
        self.validation_rules = {
            'required_fields': ['instruction', 'output', 'category'],
            'optional_fields': ['context', 'tags', 'difficulty', 'source_section'],
            'valid_difficulties': ['foundational', 'intermediate', 'advanced'],
            'valid_categories': [
                'philosophy', 'methodology', 'process', 'example', 
                'conversation', 'crisis_management', 'implementation',
                'core_principle', 'general'
            ]
        }
    
    def validate_dataset(self, data: List[ExtractedContent]) -> ValidationReport:
        """
        Perform comprehensive validation of a training dataset.
        
        Args:
            data: List of ExtractedContent items to validate
            
        Returns:
            ValidationReport with detailed findings and recommendations
        """
        if not data:
            return ValidationReport(
                total_items=0,
                valid_items=0,
                issues=[ValidationIssue(0, "empty_dataset", "error", 
                                       "Dataset is empty", 
                                       "Add training data items")],
                quality_score=0.0,
                statistics={},
                recommendations=["Add training data to the dataset"]
            )
        
        logger.info(f"Validating dataset with {len(data)} items")
        
        issues = []
        valid_items = 0
        
        # Validate each item
        for i, item in enumerate(data):
            item_issues = self._validate_item(item, i)
            issues.extend(item_issues)
            
            # Count as valid if no errors
            if not any(issue.severity == 'error' for issue in item_issues):
                valid_items += 1
        
        # Perform dataset-level validations
        dataset_issues = self._validate_dataset_level(data)
        issues.extend(dataset_issues)
        
        # Calculate quality score
        quality_score = self.calculate_quality_score(data)
        
        # Generate statistics
        statistics = self._generate_statistics(data)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(data, issues)
        
        report = ValidationReport(
            total_items=len(data),
            valid_items=valid_items,
            issues=issues,
            quality_score=quality_score,
            statistics=statistics,
            recommendations=recommendations
        )
        
        logger.info(f"Validation complete: {valid_items}/{len(data)} valid items, "
                   f"quality score: {quality_score:.2f}")
        
        return report
    
    def calculate_quality_score(self, data: List[ExtractedContent]) -> float:
        """
        Calculate overall quality score for the dataset.
        
        Args:
            data: List of ExtractedContent items
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not data:
            return 0.0
        
        scores = []
        
        for item in data:
            item_score = self._calculate_item_quality_score(item)
            scores.append(item_score)
        
        # Calculate weighted average
        base_score = sum(scores) / len(scores)
        
        # Apply dataset-level bonuses/penalties
        diversity_bonus = self._calculate_diversity_bonus(data)
        completeness_bonus = self._calculate_completeness_bonus(data)
        
        final_score = min(1.0, base_score + diversity_bonus + completeness_bonus)
        
        return round(final_score, 3)
    
    def _validate_item(self, item: ExtractedContent, index: int) -> List[ValidationIssue]:
        """Validate a single training item."""
        issues = []
        
        # Check required fields
        if not item.instruction or not item.instruction.strip():
            issues.append(ValidationIssue(
                index, "missing_instruction", "error",
                "Instruction is empty or missing",
                "Provide a clear instruction or question"
            ))
        
        if not item.output or not item.output.strip():
            issues.append(ValidationIssue(
                index, "missing_output", "error",
                "Output is empty or missing",
                "Provide a comprehensive response"
            ))
        
        if not item.category or not item.category.strip():
            issues.append(ValidationIssue(
                index, "missing_category", "warning",
                "Category is empty or missing",
                "Assign an appropriate category"
            ))
        
        # Length validations
        if item.instruction:
            inst_len = len(item.instruction)
            if inst_len < self.MIN_INSTRUCTION_LENGTH:
                issues.append(ValidationIssue(
                    index, "instruction_too_short", "warning",
                    f"Instruction too short ({inst_len} chars, min: {self.MIN_INSTRUCTION_LENGTH})",
                    "Provide more detailed instruction or context"
                ))
            elif inst_len > self.MAX_INSTRUCTION_LENGTH:
                issues.append(ValidationIssue(
                    index, "instruction_too_long", "warning",
                    f"Instruction too long ({inst_len} chars, max: {self.MAX_INSTRUCTION_LENGTH})",
                    "Break down into smaller, focused instructions"
                ))
        
        if item.output:
            out_len = len(item.output)
            if out_len < self.MIN_OUTPUT_LENGTH:
                issues.append(ValidationIssue(
                    index, "output_too_short", "warning",
                    f"Output too short ({out_len} chars, min: {self.MIN_OUTPUT_LENGTH})",
                    "Provide more comprehensive response"
                ))
            elif out_len > self.MAX_OUTPUT_LENGTH:
                issues.append(ValidationIssue(
                    index, "output_too_long", "warning",
                    f"Output too long ({out_len} chars, max: {self.MAX_OUTPUT_LENGTH})",
                    "Consider breaking into multiple training examples"
                ))
        
        # Content quality checks
        if item.instruction:
            for pattern in self.POOR_INSTRUCTION_PATTERNS:
                if re.match(pattern, item.instruction.strip(), re.IGNORECASE):
                    issues.append(ValidationIssue(
                        index, "poor_instruction_quality", "warning",
                        "Instruction appears too simple or generic",
                        "Provide more specific and detailed instructions"
                    ))
                    break
        
        if item.output:
            for pattern in self.POOR_OUTPUT_PATTERNS:
                if re.match(pattern, item.output.strip(), re.IGNORECASE):
                    issues.append(ValidationIssue(
                        index, "poor_output_quality", "warning",
                        "Output appears too simple or unhelpful",
                        "Provide more comprehensive and detailed responses"
                    ))
                    break
        
        # Category validation
        if item.category and item.category not in self.validation_rules['valid_categories']:
            issues.append(ValidationIssue(
                index, "invalid_category", "info",
                f"Unknown category: {item.category}",
                f"Use one of: {', '.join(self.validation_rules['valid_categories'])}"
            ))
        
        # Difficulty validation
        if item.difficulty and item.difficulty not in self.validation_rules['valid_difficulties']:
            issues.append(ValidationIssue(
                index, "invalid_difficulty", "info",
                f"Unknown difficulty: {item.difficulty}",
                f"Use one of: {', '.join(self.validation_rules['valid_difficulties'])}"
            ))
        
        # Tags validation
        if item.tags:
            if len(item.tags) > 10:
                issues.append(ValidationIssue(
                    index, "too_many_tags", "info",
                    f"Too many tags ({len(item.tags)}), consider reducing",
                    "Keep tags focused and relevant (max 5-7 recommended)"
                ))
            
            # Check for empty or very short tags
            short_tags = [tag for tag in item.tags if len(tag.strip()) < 3]
            if short_tags:
                issues.append(ValidationIssue(
                    index, "short_tags", "info",
                    f"Very short tags found: {short_tags}",
                    "Use descriptive tags with at least 3 characters"
                ))
        
        return issues
    
    def _validate_dataset_level(self, data: List[ExtractedContent]) -> List[ValidationIssue]:
        """Perform dataset-level validations."""
        issues = []
        
        # Check dataset size
        if len(data) < 5:
            issues.append(ValidationIssue(
                -1, "dataset_too_small", "warning",
                f"Dataset is very small ({len(data)} items)",
                "Consider adding more training examples for better model performance"
            ))
        
        # Check for duplicates
        duplicates = self._find_duplicates(data)
        if duplicates:
            issues.append(ValidationIssue(
                -1, "duplicate_instructions", "warning",
                f"Found {len(duplicates)} duplicate instructions",
                "Review and remove or modify duplicate entries"
            ))
        
        # Check category distribution
        categories = [item.category for item in data if item.category]
        category_counts = Counter(categories)
        
        if len(category_counts) == 1:
            issues.append(ValidationIssue(
                -1, "single_category", "info",
                "All items belong to the same category",
                "Consider adding variety in training categories"
            ))
        
        # Check for heavily imbalanced categories
        if category_counts:
            max_count = max(category_counts.values())
            min_count = min(category_counts.values())
            
            if max_count > min_count * 10:  # 10:1 ratio threshold
                issues.append(ValidationIssue(
                    -1, "imbalanced_categories", "info",
                    "Dataset has heavily imbalanced categories",
                    "Consider balancing category distribution for better training"
                ))
        
        # Check for missing context in conversation items
        conversation_items = [item for item in data if item.category == 'conversation']
        missing_context = [item for item in conversation_items if not item.context]
        
        if missing_context and len(missing_context) > len(conversation_items) * 0.5:
            issues.append(ValidationIssue(
                -1, "missing_conversation_context", "info",
                f"{len(missing_context)} conversation items missing context",
                "Add context to conversation examples for better training"
            ))
        
        return issues
    
    def _calculate_item_quality_score(self, item: ExtractedContent) -> float:
        """Calculate quality score for a single item."""
        score = 1.0
        
        # Length scoring
        if item.instruction:
            inst_len = len(item.instruction)
            if inst_len < self.MIN_INSTRUCTION_LENGTH:
                score -= 0.2
            elif inst_len > self.MAX_INSTRUCTION_LENGTH:
                score -= 0.1
        else:
            score -= 0.5
        
        if item.output:
            out_len = len(item.output)
            if out_len < self.MIN_OUTPUT_LENGTH:
                score -= 0.3
            elif out_len > self.MAX_OUTPUT_LENGTH:
                score -= 0.1
        else:
            score -= 0.5
        
        # Content quality scoring
        if item.instruction:
            for pattern in self.POOR_INSTRUCTION_PATTERNS:
                if re.match(pattern, item.instruction.strip(), re.IGNORECASE):
                    score -= 0.2
                    break
        
        if item.output:
            for pattern in self.POOR_OUTPUT_PATTERNS:
                if re.match(pattern, item.output.strip(), re.IGNORECASE):
                    score -= 0.3
                    break
        
        # Completeness scoring
        if item.category:
            score += 0.05
        if item.tags:
            score += 0.05
        if item.context:
            score += 0.05
        if item.source_section:
            score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def _calculate_diversity_bonus(self, data: List[ExtractedContent]) -> float:
        """Calculate bonus for dataset diversity."""
        if not data:
            return 0.0
        
        # Category diversity
        categories = set(item.category for item in data if item.category)
        category_bonus = min(0.1, len(categories) * 0.02)
        
        # Tag diversity
        all_tags = set()
        for item in data:
            all_tags.update(item.tags)
        tag_bonus = min(0.05, len(all_tags) * 0.005)
        
        # Difficulty diversity
        difficulties = set(item.difficulty for item in data if item.difficulty)
        difficulty_bonus = min(0.05, len(difficulties) * 0.02)
        
        return category_bonus + tag_bonus + difficulty_bonus
    
    def _calculate_completeness_bonus(self, data: List[ExtractedContent]) -> float:
        """Calculate bonus for dataset completeness."""
        if not data:
            return 0.0
        
        # Calculate completion rates for optional fields
        context_rate = sum(1 for item in data if item.context) / len(data)
        tags_rate = sum(1 for item in data if item.tags) / len(data)
        source_rate = sum(1 for item in data if item.source_section) / len(data)
        
        avg_completeness = (context_rate + tags_rate + source_rate) / 3
        
        return avg_completeness * 0.1  # Max 0.1 bonus
    
    def _find_duplicates(self, data: List[ExtractedContent]) -> List[Tuple[int, int]]:
        """Find duplicate instructions in the dataset."""
        duplicates = []
        instructions = {}
        
        for i, item in enumerate(data):
            if item.instruction:
                instruction_clean = item.instruction.strip().lower()
                if instruction_clean in instructions:
                    duplicates.append((instructions[instruction_clean], i))
                else:
                    instructions[instruction_clean] = i
        
        return duplicates
    
    def _generate_statistics(self, data: List[ExtractedContent]) -> Dict[str, Any]:
        """Generate comprehensive dataset statistics."""
        if not data:
            return {}
        
        # Basic statistics
        instruction_lengths = [len(item.instruction) for item in data if item.instruction]
        output_lengths = [len(item.output) for item in data if item.output]
        
        # Category distribution
        categories = [item.category for item in data if item.category]
        category_dist = dict(Counter(categories))
        
        # Difficulty distribution
        difficulties = [item.difficulty for item in data if item.difficulty]
        difficulty_dist = dict(Counter(difficulties))
        
        # Tag analysis
        all_tags = []
        for item in data:
            all_tags.extend(item.tags)
        tag_dist = dict(Counter(all_tags))
        
        return {
            'total_items': len(data),
            'instruction_length': {
                'min': min(instruction_lengths) if instruction_lengths else 0,
                'max': max(instruction_lengths) if instruction_lengths else 0,
                'avg': sum(instruction_lengths) / len(instruction_lengths) if instruction_lengths else 0,
                'median': sorted(instruction_lengths)[len(instruction_lengths)//2] if instruction_lengths else 0
            },
            'output_length': {
                'min': min(output_lengths) if output_lengths else 0,
                'max': max(output_lengths) if output_lengths else 0,
                'avg': sum(output_lengths) / len(output_lengths) if output_lengths else 0,
                'median': sorted(output_lengths)[len(output_lengths)//2] if output_lengths else 0
            },
            'category_distribution': category_dist,
            'difficulty_distribution': difficulty_dist,
            'tag_distribution': dict(list(tag_dist.items())[:10]),  # Top 10 tags
            'completeness': {
                'has_context': sum(1 for item in data if item.context) / len(data),
                'has_tags': sum(1 for item in data if item.tags) / len(data),
                'has_source': sum(1 for item in data if item.source_section) / len(data)
            }
        }
    
    def _generate_recommendations(self, 
                                 data: List[ExtractedContent], 
                                 issues: List[ValidationIssue]) -> List[str]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []
        
        # Analyze issue patterns
        error_count = sum(1 for issue in issues if issue.severity == 'error')
        warning_count = sum(1 for issue in issues if issue.severity == 'warning')
        
        if error_count > 0:
            recommendations.append(
                f"Fix {error_count} critical errors before using this dataset for training"
            )
        
        if warning_count > len(data) * 0.3:  # More than 30% of items have warnings
            recommendations.append(
                "Consider improving data quality - many items have quality issues"
            )
        
        # Dataset size recommendations
        if len(data) < 20:
            recommendations.append(
                "Dataset is quite small. Consider adding more examples for better training results"
            )
        elif len(data) < 100:
            recommendations.append(
                "Dataset size is moderate. Adding more examples could improve model performance"
            )
        
        # Balance recommendations
        categories = [item.category for item in data if item.category]
        if categories:
            category_counts = Counter(categories)
            max_count = max(category_counts.values())
            min_count = min(category_counts.values())
            
            if max_count > min_count * 5:
                recommendations.append(
                    "Consider balancing category distribution for more robust training"
                )
        
        # Quality recommendations
        quality_score = self.calculate_quality_score(data)
        if quality_score < 0.7:
            recommendations.append(
                "Overall quality score is low. Focus on improving instruction and output quality"
            )
        elif quality_score < 0.8:
            recommendations.append(
                "Good quality dataset. Small improvements could enhance training effectiveness"
            )
        else:
            recommendations.append(
                "High quality dataset ready for training!"
            )
        
        return recommendations