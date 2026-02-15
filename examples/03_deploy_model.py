#!/usr/bin/env python3
"""
Model Deployment Example

This script demonstrates how to package a NEST model for deployment.
"""

import argparse
from pathlib import Path

from src.evaluation import ModelPackager, DeploymentConfig


def main():
    parser = argparse.ArgumentParser(description='Package NEST model for deployment')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Model config')
    parser.add_argument('--vocab', type=str, help='Vocabulary file')
    parser.add_argument('--output_dir', type=str, default='deployment', help='Output directory')
    args = parser.parse_args()
    
    print("="*80)
    print("NEST Model Deployment Packaging")
    print("="*80)
    
    # Create packager
    print("\n[1/3] Creating deployment package...")
    print("-"*80)
    
    packager = ModelPackager(
        model_path=args.model,
        config_path=args.config,
        vocab_path=args.vocab
    )
    
    # Create package
    packager.create_package(
        output_dir=args.output_dir,
        include_docs=True,
        include_tests=False
    )
    
    print(f"✓ Package created in {args.output_dir}")
    
    # Create deployment config
    print("\n[2/3] Creating deployment configuration...")
    print("-"*80)
    
    config = DeploymentConfig()
    
    # Customize config
    config.config['model']['checkpoint'] = f'models/{Path(args.model).name}'
    config.config['model']['device'] = 'cuda'
    
    config_path = Path(args.output_dir) / 'deployment_config.yaml'
    config.save(str(config_path))
    
    print(f"✓ Config saved to {config_path}")
    
    # Instructions
    print("\n[3/3] Deployment package ready!")
    print("-"*80)
    
    print(f"\nPackage contents:")
    print(f"  {args.output_dir}/")
    print(f"  ├── models/          # Model checkpoints")
    print(f"  ├── configs/         # Configuration files")
    print(f"  ├── vocab/           # Vocabulary files")
    print(f"  ├── manifest.json    # Package manifest")
    print(f"  ├── requirements.txt # Python dependencies")
    print(f"  ├── deploy.py        # Deployment script")
    print(f"  └── README.md        # Documentation")
    
    print("\n" + "="*80)
    print("Deployment Ready!")
    print("="*80)
    
    print("\nTo deploy:")
    print(f"  1. Copy the entire '{args.output_dir}' directory to target server")
    print(f"  2. Install dependencies:")
    print(f"     cd {args.output_dir} && pip install -r requirements.txt")
    print(f"  3. Run deployment:")
    print(f"     python deploy.py --model models/*.pt --config configs/*.yaml")
    
    print("\nFor production:")
    print("  - Set up Docker container")
    print("  - Configure load balancing")
    print("  - Set up monitoring and logging")
    print("  - Enable HTTPS/authentication")


if __name__ == '__main__':
    main()
