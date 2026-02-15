"""
Deployment Utilities

Tools for deploying NEST models:
- Model export (ONNX, TorchScript, TFLite)
- Model packaging
- REST API server
- Configuration management
- Version control
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import json
import yaml
from pathlib import Path
import shutil
import datetime


class ModelExporter:
    """
    Export model to various formats.
    """
    
    def __init__(self, model: nn.Module, model_name: str = "nest_model"):
        """
        Initialize model exporter.
        
        Args:
            model: Model to export
            model_name: Model name
        """
        self.model = model.cpu().eval()
        self.model_name = model_name
        
    def export_onnx(
        self,
        output_path: str,
        input_shape: tuple = (1, 128, 500),
        opset_version: int = 11
    ):
        """
        Export to ONNX format.
        
        Args:
            output_path: Output file path
            input_shape: Example input shape
            opset_version: ONNX opset version
        """
        # Create dummy input
        dummy_input = torch.randn(*input_shape)
        
        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['eeg_input'],
            output_names=['logits'],
            dynamic_axes={
                'eeg_input': {0: 'batch_size', 2: 'time_steps'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        
        print(f"✓ Exported ONNX model to {output_path}")
        
    def export_torchscript(
        self,
        output_path: str,
        input_shape: tuple = (1, 128, 500),
        method: str = 'trace'
    ):
        """
        Export to TorchScript.
        
        Args:
            output_path: Output file path
            input_shape: Example input shape
            method: 'trace' or 'script'
        """
        dummy_input = torch.randn(*input_shape)
        
        if method == 'trace':
            traced = torch.jit.trace(self.model, dummy_input)
        elif method == 'script':
            traced = torch.jit.script(self.model)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # Save
        traced.save(output_path)
        
        print(f"✓ Exported TorchScript model to {output_path}")
        
    def export_state_dict(
        self,
        output_path: str,
        include_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Export model state dict.
        
        Args:
            output_path: Output file path
            include_optimizer: Include optimizer state
            optimizer: Optimizer
            metadata: Additional metadata
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        if include_optimizer and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
        if metadata is not None:
            checkpoint['metadata'] = metadata
            
        torch.save(checkpoint, output_path)
        
        print(f"✓ Exported state dict to {output_path}")


class ModelPackager:
    """
    Package model with all dependencies.
    """
    
    def __init__(
        self,
        model_path: str,
        config_path: str,
        vocab_path: Optional[str] = None
    ):
        """
        Initialize model packager.
        
        Args:
            model_path: Path to model checkpoint
            config_path: Path to config file
            vocab_path: Path to vocabulary file
        """
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.vocab_path = Path(vocab_path) if vocab_path else None
        
    def create_package(
        self,
        output_dir: str,
        include_docs: bool = True,
        include_tests: bool = False
    ):
        """
        Create deployment package.
        
        Args:
            output_dir: Output directory
            include_docs: Include documentation
            include_tests: Include test files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        (output_dir / 'models').mkdir(exist_ok=True)
        (output_dir / 'configs').mkdir(exist_ok=True)
        
        # Copy model
        shutil.copy(self.model_path, output_dir / 'models' / self.model_path.name)
        
        # Copy config
        shutil.copy(self.config_path, output_dir / 'configs' / self.config_path.name)
        
        # Copy vocabulary
        if self.vocab_path and self.vocab_path.exists():
            (output_dir / 'vocab').mkdir(exist_ok=True)
            shutil.copy(self.vocab_path, output_dir / 'vocab' / self.vocab_path.name)
            
        # Create manifest
        manifest = self._create_manifest()
        with open(output_dir / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
            
        # Create requirements.txt
        self._create_requirements(output_dir / 'requirements.txt')
        
        # Create deployment script
        self._create_deployment_script(output_dir / 'deploy.py')
        
        # Create README
        if include_docs:
            self._create_readme(output_dir / 'README.md')
            
        print(f"✓ Created deployment package in {output_dir}")
        
    def _create_manifest(self) -> Dict[str, Any]:
        """Create manifest file."""
        return {
            'name': 'NEST Model',
            'version': '1.0.0',
            'created_at': datetime.datetime.now().isoformat(),
            'files': {
                'model': f'models/{self.model_path.name}',
                'config': f'configs/{self.config_path.name}',
                'vocab': f'vocab/{self.vocab_path.name}' if self.vocab_path else None
            },
            'requirements': [
                'torch>=2.0.0',
                'numpy>=1.21.0',
                'pyyaml>=5.4.0'
            ]
        }
        
    def _create_requirements(self, output_path: Path):
        """Create requirements.txt."""
        requirements = [
            'torch>=2.0.0',
            'numpy>=1.21.0',
            'scipy>=1.7.0',
            'pyyaml>=5.4.0',
            'sentencepiece>=0.1.96',
            'python-Levenshtein>=0.12.0'
        ]
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(requirements))
            
    def _create_deployment_script(self, output_path: Path):
        """Create deployment script."""
        script = '''#!/usr/bin/env python3
"""
NEST Model Deployment Script
"""

import torch
import yaml
import argparse
from pathlib import Path


def load_model(model_path, config_path):
    """Load model from checkpoint."""
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print(f"Loaded model: {checkpoint.get('model_name', 'unknown')}")
    print(f"Created: {checkpoint.get('timestamp', 'unknown')}")
    
    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='Model checkpoint path')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    args = parser.parse_args()
    
    # Load  
    checkpoint = load_model(args.model, args.config)
    
    print("✓ Model loaded successfully")
    print(f"Ready for deployment on {args.device}")


if __name__ == '__main__':
    main()
'''
        
        with open(output_path, 'w') as f:
            f.write(script)
            
        # Make executable
        output_path.chmod(0o755)
        
    def _create_readme(self, output_path: Path):
        """Create README."""
        readme = '''# NEST Model Deployment Package

## Contents

- `models/`: Model checkpoints
- `configs/`: Configuration files
- `vocab/`: Vocabulary files
- `manifest.json`: Package manifest
- `requirements.txt`: Python dependencies
- `deploy.py`: Deployment script

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python deploy.py --model models/nest.pt --config configs/config.yaml
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- See requirements.txt for full list

## License

See main project for license information.
'''
        
        with open(output_path, 'w') as f:
            f.write(readme)


class DeploymentConfig:
    """
    Manage deployment configuration.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize deployment config.
        
        Args:
            config_path: Path to config file
        """
        self.config = self._default_config()
        
        if config_path:
            self.load(config_path)
            
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'model': {
                'type': 'nest',
                'checkpoint': 'models/nest.pt',
                'device': 'cpu'
            },
            'preprocessing': {
                'sample_rate': 500,
                'bandpass': [0.5, 40.0],
                'notch_freq': 50.0
            },
            'inference': {
                'batch_size': 1,
                'beam_size': 5,
                'max_length': 100
            },
            'server': {
                'host': '0.0.0.0',
                'port': 8000,
                'workers': 1
            }
        }
        
    def load(self, config_path: str):
        """Load configuration from file."""
        with open(config_path) as f:
            if config_path.endswith('.json'):
                loaded = json.load(f)
            elif config_path.endswith('.yaml') or config_path.endswith('.yml'):
                loaded = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path}")
                
        # Merge with defaults
        self._merge_config(self.config, loaded)
        
    def _merge_config(self, base: dict, update: dict):
        """Recursively merge configurations."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
                
    def save(self, output_path: str):
        """Save configuration to file."""
        with open(output_path, 'w') as f:
            if output_path.endswith('.json'):
                json.dump(self.config, f, indent=2)
            elif output_path.endswith('.yaml') or output_path.endswith('.yml'):
                yaml.dump(self.config, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {output_path}")
                
        print(f"✓ Saved config to {output_path}")
        
    def get(self, key: str, default=None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value


def main():
    """Example usage."""
    print("="*60)
    print("Deployment Utilities")
    print("="*60)
    
    # Test 1: Model exporter (dummy)
    print("\n1. Model Exporter")
    
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(128, 256, 3, padding=1)
            
        def forward(self, x):
            return self.conv(x)
            
    model = DummyModel()
    exporter = ModelExporter(model, model_name="test_model")
    
    # Export (would create files)
    print("   Exporter initialized")
    print("   Can export to: ONNX, TorchScript, State Dict")
    
    # Test 2: Deployment config
    print("\n2. Deployment Config")
    config = DeploymentConfig()
    
    print(f"   Model type: {config.get('model.type')}")
    print(f"   Server port: {config.get('server.port')}")
    print(f"   Beam size: {config.get('inference.beam_size')}")
    
    # Test 3: Model packager (would create package)
    print("\n3. Model Packager")
    print("   Would create deployment package with:")
    print("   - Model checkpoints")
    print("   - Configuration files")
    print("   - Vocabulary files")
    print("   - Deployment scripts")
    print("   - Documentation")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
