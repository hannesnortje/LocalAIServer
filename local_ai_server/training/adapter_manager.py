"""
Adapter Management System for QLoRA Training

This module provides management capabilities for trained LoRA adapters,
including listing, loading, and deleting adapters.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import shutil

logger = logging.getLogger(__name__)


@dataclass
class AdapterInfo:
    """Information about a trained LoRA adapter"""
    name: str
    path: str
    model_name: str
    created_at: datetime
    size_mb: float
    description: Optional[str] = None
    training_config: Optional[Dict] = None
    metrics: Optional[Dict] = None
    is_loaded: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert adapter info to dictionary for JSON serialization"""
        result = asdict(self)
        result['created_at'] = self.created_at.isoformat()
        return result


class AdapterManager:
    """
    Manages trained LoRA adapters for the QLoRA training system.
    
    Features:
    - Adapter discovery and listing
    - Adapter metadata management
    - Adapter loading/unloading
    - Adapter deletion and cleanup
    """
    
    def __init__(self, adapters_dir: str = "./adapters"):
        """
        Initialize the adapter manager.
        
        Args:
            adapters_dir: Directory where adapters are stored
        """
        self.adapters_dir = Path(adapters_dir)
        self.adapters_dir.mkdir(parents=True, exist_ok=True)
        
        # Current loaded adapter
        self.current_adapter: Optional[str] = None
        
        logger.info(f"AdapterManager initialized with directory: {self.adapters_dir}")
    
    def discover_adapters(self) -> List[AdapterInfo]:
        """
        Discover all available adapters in the adapters directory.
        
        Returns:
            List of AdapterInfo objects
        """
        adapters = []
        
        try:
            # Look for adapter directories
            for adapter_path in self.adapters_dir.iterdir():
                if not adapter_path.is_dir():
                    continue
                
                # Check if it's a valid adapter (has adapter_config.json)
                config_file = adapter_path / "adapter_config.json"
                if not config_file.exists():
                    continue
                
                try:
                    # Load adapter information
                    adapter_info = self._load_adapter_info(adapter_path)
                    if adapter_info:
                        adapters.append(adapter_info)
                except Exception as e:
                    logger.warning(f"Failed to load adapter info from {adapter_path}: {e}")
            
            logger.info(f"Discovered {len(adapters)} adapters")
            return adapters
            
        except Exception as e:
            logger.error(f"Failed to discover adapters: {e}")
            return []
    
    def _load_adapter_info(self, adapter_path: Path) -> Optional[AdapterInfo]:
        """Load adapter information from directory"""
        try:
            # Load adapter config
            config_file = adapter_path / "adapter_config.json"
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Load training metadata if available
            metadata_file = adapter_path / "training_metadata.json"
            metadata = {}
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            # Calculate size
            size_mb = self._calculate_directory_size(adapter_path)
            
            # Get creation time
            created_at = datetime.fromtimestamp(adapter_path.stat().st_ctime)
            
            # Extract information
            adapter_info = AdapterInfo(
                name=adapter_path.name,
                path=str(adapter_path),
                model_name=metadata.get('model_name', 'unknown'),
                created_at=created_at,
                size_mb=size_mb,
                description=metadata.get('description'),
                training_config=metadata.get('training_config'),
                metrics=metadata.get('metrics'),
                is_loaded=(self.current_adapter == adapter_path.name)
            )
            
            return adapter_info
            
        except Exception as e:
            logger.error(f"Failed to load adapter info from {adapter_path}: {e}")
            return None
    
    def _calculate_directory_size(self, path: Path) -> float:
        """Calculate directory size in MB"""
        total_size = 0
        try:
            for file_path in path.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size / (1024 * 1024)  # Convert to MB
        except Exception as e:
            logger.warning(f"Failed to calculate size for {path}: {e}")
            return 0.0
    
    def get_adapter(self, name: str) -> Optional[AdapterInfo]:
        """
        Get information about a specific adapter.
        
        Args:
            name: Adapter name
            
        Returns:
            AdapterInfo object or None if not found
        """
        adapter_path = self.adapters_dir / name
        if not adapter_path.exists() or not adapter_path.is_dir():
            return None
        
        return self._load_adapter_info(adapter_path)
    
    def delete_adapter(self, name: str) -> bool:
        """
        Delete an adapter.
        
        Args:
            name: Adapter name to delete
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        try:
            adapter_path = self.adapters_dir / name
            
            if not adapter_path.exists():
                logger.warning(f"Adapter {name} not found")
                return False
            
            # Unload if currently loaded
            if self.current_adapter == name:
                self.unload_adapter()
            
            # Delete the directory
            shutil.rmtree(adapter_path)
            
            logger.info(f"Deleted adapter: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete adapter {name}: {e}")
            return False
    
    def load_adapter(self, name: str) -> bool:
        """
        Load an adapter for inference.
        
        This integrates with the model manager to actually load the adapter
        onto the base model for inference use.
        
        Args:
            name: Adapter name to load
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            adapter_path = self.adapters_dir / name
            
            if not adapter_path.exists():
                logger.error(f"Adapter {name} not found")
                return False
            
            # Validate adapter
            config_file = adapter_path / "adapter_config.json"
            if not config_file.exists():
                logger.error(f"Adapter {name} is missing configuration file")
                return False
            
            # Import here to avoid circular imports
            from ..model_manager import model_manager
            
            # Load adapter through model manager
            success = model_manager.load_adapter(name)
            
            if success:
                self.current_adapter = name
                logger.info(f"Adapter {name} loaded successfully")
            else:
                logger.error(f"Failed to load adapter {name} through model manager")
            
            return success
            
        except Exception as e:
            logger.error(f"Error loading adapter {name}: {e}")
            return False
            
            logger.info(f"Loaded adapter: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load adapter {name}: {e}")
            return False
    
    def unload_adapter(self) -> bool:
        """
        Unload the currently loaded adapter.
        
        Returns:
            bool: True if unloaded successfully, False otherwise
        """
        try:
            if self.current_adapter:
                # Import here to avoid circular imports
                from ..model_manager import model_manager
                
                # Unload adapter through model manager
                success = model_manager.unload_adapter()
                
                if success:
                    logger.info(f"Unloaded adapter: {self.current_adapter}")
                    self.current_adapter = None
                else:
                    logger.error("Failed to unload adapter through model manager")
                
                return success
            else:
                logger.warning("No adapter currently loaded")
                return True  # Nothing to unload is considered success
                
        except Exception as e:
            logger.error(f"Failed to unload adapter: {e}")
            return False
    
    def get_current_adapter(self) -> Optional[str]:
        """Get the name of the currently loaded adapter"""
        return self.current_adapter
    
    def save_adapter_metadata(
        self, 
        adapter_path: Path, 
        model_name: str,
        training_config: Optional[Dict] = None,
        metrics: Optional[Dict] = None,
        description: Optional[str] = None
    ):
        """
        Save metadata for a trained adapter.
        
        Args:
            adapter_path: Path to the adapter directory
            model_name: Name of the base model
            training_config: Training configuration used
            metrics: Training metrics
            description: Optional description
        """
        try:
            metadata = {
                'model_name': model_name,
                'created_at': datetime.now().isoformat(),
                'training_config': training_config,
                'metrics': metrics,
                'description': description
            }
            
            metadata_file = adapter_path / "training_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved metadata for adapter at {adapter_path}")
            
        except Exception as e:
            logger.error(f"Failed to save adapter metadata: {e}")
    
    def import_adapter(
        self, 
        source_path: str, 
        adapter_name: str,
        model_name: str,
        description: Optional[str] = None
    ) -> bool:
        """
        Import an adapter from an external location.
        
        Args:
            source_path: Source path of the adapter
            adapter_name: Name to give the adapter
            model_name: Name of the base model
            description: Optional description
            
        Returns:
            bool: True if imported successfully, False otherwise
        """
        try:
            source = Path(source_path)
            if not source.exists():
                logger.error(f"Source path does not exist: {source_path}")
                return False
            
            # Create destination
            dest_path = self.adapters_dir / adapter_name
            if dest_path.exists():
                logger.error(f"Adapter {adapter_name} already exists")
                return False
            
            # Copy adapter
            if source.is_dir():
                shutil.copytree(source, dest_path)
            else:
                logger.error(f"Source must be a directory: {source_path}")
                return False
            
            # Save metadata
            self.save_adapter_metadata(
                dest_path,
                model_name=model_name,
                description=description
            )
            
            logger.info(f"Imported adapter {adapter_name} from {source_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import adapter: {e}")
            return False
    
    def export_adapter(self, adapter_name: str, dest_path: str) -> bool:
        """
        Export an adapter to an external location.
        
        Args:
            adapter_name: Name of the adapter to export
            dest_path: Destination path
            
        Returns:
            bool: True if exported successfully, False otherwise
        """
        try:
            source_path = self.adapters_dir / adapter_name
            if not source_path.exists():
                logger.error(f"Adapter {adapter_name} not found")
                return False
            
            dest = Path(dest_path)
            if dest.exists():
                logger.error(f"Destination already exists: {dest_path}")
                return False
            
            # Copy adapter
            shutil.copytree(source_path, dest)
            
            logger.info(f"Exported adapter {adapter_name} to {dest_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export adapter: {e}")
            return False


# Global adapter manager instance
adapter_manager: Optional[AdapterManager] = None


def get_adapter_manager() -> AdapterManager:
    """Get the global adapter manager instance"""
    global adapter_manager
    if adapter_manager is None:
        adapter_manager = AdapterManager()
    return adapter_manager


def initialize_adapter_manager(**kwargs) -> AdapterManager:
    """Initialize the global adapter manager with custom settings"""
    global adapter_manager
    adapter_manager = AdapterManager(**kwargs)
    return adapter_manager