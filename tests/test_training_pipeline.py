#!/usr/bin/env python3
"""
Training Pipeline Test Script
============================

This script tests the Step 5 training data pipeline by processing
the 42 document and generating training data ready for QLoRA training.

Test Workflow:
1. Load and analyze the 42 document using DocumentAnalyzer
2. Process and validate the extracted data using DataManager
3. Convert to CodeLlama training format using DataPreprocessor
4. Generate comprehensive validation and quality reports
5. Save processed data in multiple formats for training

Usage:
    python tests/test_training_pipeline.py [path_to_42_document]
    # Or from within tests directory:
    cd tests && python test_training_pipeline.py [path_to_42_document]
"""

import sys
import json
import logging
from pathlib import Path

# Add the local_ai_server to Python path
sys.path.append(str(Path(__file__).parent.parent))

from local_ai_server.training import (
    DocumentAnalyzer, 
    DataManager, 
    DataPreprocessor, 
    DataValidator,
    FormatConverter,
    PromptFormat
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_training_pipeline(document_path: str = None):
    """
    Test the complete training data pipeline with the 42 document.
    
    Args:
        document_path: Path to the 42 document (optional)
    """
    logger.info("🚀 Starting Training Pipeline Test - Step 5 Validation")
    logger.info("=" * 60)
    
    # Default to attached 42 document if no path provided
    if not document_path:
        # Look for the 42 document in common locations
        possible_paths = [
            "42-comprehensive-analysis-report.md",
            "../42-comprehensive-analysis-report.md", 
            "docs/42-comprehensive-analysis-report.md"
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                document_path = path
                break
        
        if not document_path:
            logger.error("❌ 42 document not found. Please provide path as argument.")
            logger.info("Usage: python tests/test_training_pipeline.py <path_to_42_document>")
            return False
    
    if not Path(document_path).exists():
        logger.error(f"❌ Document not found: {document_path}")
        return False
    
    logger.info(f"📄 Processing document: {document_path}")
    
    try:
        # Step 1: Initialize components
        logger.info("\n📋 Step 1: Initializing Training Pipeline Components")
        
        analyzer = DocumentAnalyzer()
        data_manager = DataManager("test_training_data")
        preprocessor = DataPreprocessor(
            prompt_format=PromptFormat.CHATML,
            max_length=2048,
            include_system_prompt=True
        )
        validator = DataValidator()
        converter = FormatConverter()
        
        logger.info("✅ All components initialized successfully")
        
        # Step 2: Analyze document and extract training data
        logger.info("\n🔍 Step 2: Analyzing 42 Document and Extracting Training Data")
        
        extracted_data = analyzer.analyze_document(document_path)
        
        if not extracted_data:
            logger.error("❌ No training data extracted from document")
            return False
        
        logger.info(f"✅ Extracted {len(extracted_data)} training items")
        
        # Show sample extracted data
        logger.info("\n📋 Sample Extracted Data:")
        for i, item in enumerate(extracted_data[:3]):  # Show first 3 items
            logger.info(f"  Item {i+1}:")
            logger.info(f"    Category: {item.category}")
            logger.info(f"    Instruction: {item.instruction[:80]}...")
            logger.info(f"    Output Length: {len(item.output)} chars")
            logger.info(f"    Tags: {item.tags}")
        
        # Step 3: Validate extracted data
        logger.info("\n✅ Step 3: Validating Training Data Quality")
        
        validation_report = validator.validate_dataset(extracted_data)
        
        logger.info(f"📊 Validation Results:")
        logger.info(f"  Total Items: {validation_report.total_items}")
        logger.info(f"  Valid Items: {validation_report.valid_items}")
        logger.info(f"  Quality Score: {validation_report.quality_score:.3f}")
        logger.info(f"  Issues Found: {len(validation_report.issues)}")
        
        # Show critical issues
        errors = [issue for issue in validation_report.issues if issue.severity == 'error']
        if errors:
            logger.warning(f"⚠️ Critical Issues Found: {len(errors)}")
            for error in errors[:3]:  # Show first 3 errors
                logger.warning(f"  - {error.message}")
        else:
            logger.info("✅ No critical issues found")
        
        # Show recommendations
        if validation_report.recommendations:
            logger.info("\n💡 Recommendations:")
            for rec in validation_report.recommendations[:3]:
                logger.info(f"  • {rec}")
        
        # Step 4: Save dataset and create splits
        logger.info("\n💾 Step 4: Saving Dataset and Creating Train/Validation Splits")
        
        # Save original extracted data
        dataset_path = data_manager.save_dataset(extracted_data, "42_methodology")
        logger.info(f"✅ Saved dataset: {dataset_path}")
        
        # Create train/validation splits
        train_data, val_data = data_manager.create_train_validation_split(
            extracted_data, 
            "42_methodology",
            train_ratio=0.8,
            stratify_by='category'
        )
        
        logger.info(f"✅ Created splits: {len(train_data)} train, {len(val_data)} validation")
        
        # Generate dataset info
        dataset_info = data_manager.generate_dataset_info(
            extracted_data,
            "42_methodology", 
            document_path,
            "markdown"
        )
        
        logger.info(f"📊 Dataset Statistics:")
        logger.info(f"  Categories: {len(dataset_info.categories)}")
        logger.info(f"  Avg Instruction Length: {dataset_info.avg_instruction_length:.0f} chars")
        logger.info(f"  Avg Output Length: {dataset_info.avg_output_length:.0f} chars")
        
        # Step 5: Process for training format
        logger.info("\n🔄 Step 5: Converting to CodeLlama Training Format")
        
        processed_train = preprocessor.process_dataset(train_data)
        processed_val = preprocessor.process_dataset(val_data)
        
        if not processed_train:
            logger.error("❌ Failed to process training data")
            return False
        
        logger.info(f"✅ Processed {len(processed_train)} training items")
        logger.info(f"✅ Processed {len(processed_val)} validation items")
        
        # Show sample processed data
        logger.info("\n📋 Sample Processed Training Data (ChatML Format):")
        sample_item = processed_train[0]
        logger.info("=" * 50)
        logger.info(sample_item.text[:500] + "..." if len(sample_item.text) > 500 else sample_item.text)
        logger.info("=" * 50)
        
        # Save processed data
        train_path = preprocessor.save_processed_dataset(
            processed_train, 
            "test_training_data/42_methodology_train_chatml.jsonl",
            "jsonl"
        )
        val_path = preprocessor.save_processed_dataset(
            processed_val,
            "test_training_data/42_methodology_val_chatml.jsonl", 
            "jsonl"
        )
        
        logger.info(f"✅ Saved training data: {train_path}")
        logger.info(f"✅ Saved validation data: {val_path}")
        
        # Step 6: Export to multiple formats
        logger.info("\n📤 Step 6: Exporting to Multiple Training Formats")
        
        # Export to Alpaca format
        alpaca_path = converter.export_format(
            extracted_data, 
            "alpaca",
            "test_training_data/42_methodology_alpaca.json"
        )
        logger.info(f"✅ Exported Alpaca format: {alpaca_path}")
        
        # Export to ShareGPT format
        sharegpt_path = converter.export_format(
            extracted_data,
            "sharegpt", 
            "test_training_data/42_methodology_sharegpt.json"
        )
        logger.info(f"✅ Exported ShareGPT format: {sharegpt_path}")
        
        # Step 7: Generate comprehensive statistics
        logger.info("\n📈 Step 7: Generating Comprehensive Statistics")
        
        preprocessing_stats = preprocessor.get_statistics(processed_train + processed_val)
        
        logger.info(f"📊 Processing Statistics:")
        logger.info(f"  Format: {preprocessing_stats['format']}")
        logger.info(f"  Avg Text Length: {preprocessing_stats['text_length']['avg']:.0f} chars")
        logger.info(f"  Length Range: {preprocessing_stats['text_length']['min']}-{preprocessing_stats['text_length']['max']}")
        logger.info(f"  Compression Ratio: {preprocessing_stats['avg_compression_ratio']:.2f}")
        
        # Category distribution
        cat_dist = preprocessing_stats['category_distribution']
        logger.info(f"  Category Distribution:")
        for category, count in cat_dist.items():
            logger.info(f"    {category}: {count} items")
        
        # Step 8: Final validation and summary
        logger.info("\n🎯 Step 8: Final Validation and Summary")
        
        # Validate we can load the saved data
        loaded_data = data_manager.load_processed_dataset("42_methodology")
        if len(loaded_data) == len(extracted_data):
            logger.info("✅ Data persistence validation passed")
        else:
            logger.warning("⚠️ Data persistence validation failed")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("🎉 TRAINING PIPELINE TEST COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"✅ Document Analyzed: {Path(document_path).name}")
        logger.info(f"✅ Training Items Generated: {len(extracted_data)}")
        logger.info(f"✅ Quality Score: {validation_report.quality_score:.3f}")
        logger.info(f"✅ Train/Val Split: {len(train_data)}/{len(val_data)}")
        logger.info(f"✅ Processed for CodeLlama: {len(processed_train + processed_val)} items")
        logger.info(f"✅ Multiple Export Formats: Alpaca, ShareGPT, ChatML")
        logger.info("✅ Ready for Step 6: QLoRA Training Engine")
        
        # Next steps guidance
        logger.info("\n📋 Next Steps:")
        logger.info("1. Review generated training data quality")
        logger.info("2. Proceed to Step 6: QLoRA Training Engine implementation")
        logger.info("3. Use the processed ChatML data for CodeLlama training")
        logger.info("4. Test trained model with '42 = FOR TWO' methodology queries")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Training pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the training pipeline test."""
    document_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    success = test_training_pipeline(document_path)
    
    if success:
        logger.info("\n🎯 Step 5 Training Data Pipeline: IMPLEMENTATION COMPLETE!")
        sys.exit(0)
    else:
        logger.error("\n❌ Step 5 Training Data Pipeline: TEST FAILED!")
        sys.exit(1)

if __name__ == "__main__":
    main()