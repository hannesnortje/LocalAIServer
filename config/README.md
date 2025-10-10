# Configuration Directory

This directory contains all LocalAIServer configuration files organized by purpose.

## Structure

- **`training/`** - Training configuration files
  - `surgical_training_config_full.json` - Surgical LoRA configuration (ACTIVE)
  - `surgical_*.json` - Various surgical LoRA configs
  - `enhanced_training*.json` - Legacy nuclear approach configs
  - `intensive_training_config.json` - Pre-surgical approach config
  - `ultra_aggressive_training_config.json` - Nuclear approach config

- **`active/`** - Symlinks to currently active configurations
  - `current_training_config.json` -> `../training/surgical_training_config_full.json`

## Usage

The active configuration is accessed via symlink to maintain stable paths while allowing easy config switching.

Current active training config:
```bash
cat config/active/current_training_config.json
```

To switch configurations, update the symlink:
```bash
ln -sf ../training/new_config.json config/active/current_training_config.json
```