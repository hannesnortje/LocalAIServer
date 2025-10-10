# Data Directory

This directory contains all training data organized by purpose and lifecycle stage.

## Structure

- **`training/`** - Training data files
  - **`current/`** - Active training data
    - `42_training_data.json` - Core 42 philosophy training data
    - `tron_collaborative_intelligence_training.json` - Comprehensive training set
    - `tron_collaborative_intelligence_train_texts.json` - Text format training data
  - **`components/`** - Component-specific training data
    - `*_42_philosophy.json` - 42 philosophy components
    - `*_collaborative_methodologies.json` - Collaboration methodology components
    - `*_technical_excellence.json` - Technical excellence components
  - **`archive/`** - Historical/legacy training data
    - `42_intensive_training.json` - Pre-surgical intensive approach
    - `42_ultra_aggressive_training.json` - Nuclear approach data

- **`processed/`** - Cleaned and preprocessed data
- **`raw/`** - Source data before processing

## Usage

Current surgical LoRA training uses files from `training/current/`.
Component files in `training/components/` are used for multi-concept training.
Archived files are preserved for historical reference but superseded by surgical approach.