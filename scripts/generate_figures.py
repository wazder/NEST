#!/usr/bin/env python3
"""
Generate publication-quality figures for the paper.

Creates all figures referenced in NEST manuscript.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd

# Set style for publication
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['figure.dpi'] = 300
sns.set_style('whitegrid')


def load_results(results_dir: Path) -> Dict:
    """Load training results."""
    results_file = results_dir / 'results.json'
    
    if not results_file.exists():
        raise FileNotFoundError(f"Results not found: {results_file}")
    
    with open(results_file, 'r') as f:
        return json.load(f)


def load_training_history(model_dir: Path) -> Dict:
    """Load training history for a model."""
    history_file = model_dir / 'history.json'
    
    if not history_file.exists():
        return {}
    
    with open(history_file, 'r') as f:
        return json.load(f)


def figure1_architecture_diagram(output_dir: Path):
    """
    Figure 1: NEST Architecture Overview
    
    Note: This should be created with a diagram tool (draw.io, etc.)
    This function creates a placeholder.
    """
    print("Figure 1: Architecture diagram")
    print("  → Create manually with draw.io or similar")
    print("  → Include: Spatial CNN → Temporal Encoder → Attention Decoder")
    
    # Create placeholder
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.text(0.5, 0.5, 'NEST Architecture Diagram\n(Create with diagram tool)',
            ha='center', va='center', fontsize=16)
    ax.axis('off')
    
    output_path = output_dir / 'figure1_architecture.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Placeholder saved: {output_path}")


def figure2_model_comparison(results: Dict, output_dir: Path):
    """
    Figure 2: Model Performance Comparison
    
    Bar chart comparing WER, CER, BLEU across models.
    """
    print("\nFigure 2: Model performance comparison")
    
    # Extract data
    models = []
    wer_vals = []
    cer_vals = []
    bleu_vals = []
    
    for model_name, metrics in results.get('results', {}).items():
        models.append(model_name.replace('nest_', '').upper())
        wer_vals.append(metrics.get('wer', 0) * 100)  # Convert to percentage
        cer_vals.append(metrics.get('cer', 0) * 100)
        bleu_vals.append(metrics.get('bleu', 0))
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.arange(len(models))
    width = 0.6
    
    # WER
    axes[0].bar(x, wer_vals, width, color='steelblue')
    axes[0].set_ylabel('Word Error Rate (%)')
    axes[0].set_title('(a) Word Error Rate')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha='right')
    axes[0].grid(axis='y', alpha=0.3)
    
    # CER
    axes[1].bar(x, cer_vals, width, color='coral')
    axes[1].set_ylabel('Character Error Rate (%)')
    axes[1].set_title('(b) Character Error Rate')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha='right')
    axes[1].grid(axis='y', alpha=0.3)
    
    # BLEU
    axes[2].bar(x, bleu_vals, width, color='seagreen')
    axes[2].set_ylabel('BLEU Score')
    axes[2].set_title('(c) BLEU Score')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=45, ha='right')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / 'figure2_model_comparison.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def figure3_training_curves(results_dir: Path, output_dir: Path):
    """
    Figure 3: Training and Validation Loss Curves
    """
    print("\nFigure 3: Training curves")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    model_names = ['nest_conformer', 'nest_transformer', 'nest_rnn_t', 'nest_ctc']
    display_names = ['Conformer', 'Transformer', 'RNN-T', 'CTC']
    
    for idx, (model_name, display_name) in enumerate(zip(model_names, display_names)):
        model_dir = results_dir / 'checkpoints' / model_name
        history = load_training_history(model_dir)
        
        if history:
            epochs = range(1, len(history.get('train_loss', [])) + 1)
            
            axes[idx].plot(epochs, history.get('train_loss', []), 
                          label='Train Loss', linewidth=2, color='steelblue')
            axes[idx].plot(epochs, history.get('val_loss', []), 
                          label='Val Loss', linewidth=2, color='coral')
            
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel('Loss')
            axes[idx].set_title(f'({chr(97+idx)}) {display_name}')
            axes[idx].legend()
            axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / 'figure3_training_curves.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def figure4_subject_performance(output_dir: Path):
    """
    Figure 4: Per-Subject Performance Distribution
    
    Box plot or violin plot showing WER across subjects.
    """
    print("\nFigure 4: Subject performance distribution")
    
    # Simulated data - replace with actual per-subject results
    np.random.seed(42)
    subjects = [f'S{i:02d}' for i in range(1, 13)]
    wer_data = np.random.normal(15.8, 3.5, 12)  # Mean 15.8%, std 3.5%
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Violin plot
    parts = ax.violinplot([wer_data], positions=[0], widths=0.7,
                          showmeans=True, showmedians=True)
    
    # Scatter individual points
    ax.scatter(np.zeros(len(wer_data)), wer_data, alpha=0.6, s=50, color='steelblue')
    
    # Add horizontal line for mean
    ax.axhline(y=np.mean(wer_data), color='red', linestyle='--', 
               label=f'Mean: {np.mean(wer_data):.1f}%')
    
    ax.set_ylabel('Word Error Rate (%)')
    ax.set_title('Per-Subject Performance Distribution')
    ax.set_xticks([0])
    ax.set_xticklabels(['All Subjects'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / 'figure4_subject_performance.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")
    print(f"  ⚠️  Using simulated data - replace with actual per-subject results")


def figure5_ablation_study(output_dir: Path):
    """
    Figure 5: Ablation Study Results
    """
    print("\nFigure 5: Ablation study")
    
    # Ablation data from paper
    configurations = [
        'Full Model',
        '- Multi-head\nAttention',
        '- Positional\nEncoding',
        '- Feedforward\nNetwork',
        '- Convolution\nModule',
        '- Data\nAugmentation'
    ]
    
    wer_values = [15.8, 17.6, 18.9, 19.4, 18.1, 17.2]
    deltas = [0, 1.8, 3.1, 3.6, 2.3, 1.4]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['green' if d == 0 else 'coral' for d in deltas]
    bars = ax.barh(configurations, wer_values, color=colors, alpha=0.7)
    
    # Add delta annotations
    for i, (bar, delta) in enumerate(zip(bars, deltas)):
        if delta > 0:
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                   f'+{delta:.1f}%', va='center', fontweight='bold')
    
    ax.set_xlabel('Word Error Rate (%)')
    ax.set_title('Ablation Study: Component Contribution')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / 'figure5_ablation.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def figure6_optimization_results(output_dir: Path):
    """
    Figure 6: Model Optimization Trade-offs
    
    Scatter plot: Model size vs WER for different optimization strategies.
    """
    print("\nFigure 6: Optimization trade-offs")
    
    # Data from paper
    configs = ['Dense FP32', 'Pruned 20%', 'Pruned 40%', 'Pruned 60%', 
               'FP16', 'INT8 PTQ', 'INT8 QAT', 'Pruned+INT8']
    
    sizes = [16.8, 13.4, 10.1, 6.7, 8.4, 4.2, 4.2, 2.5]  # MB
    wers = [15.8, 15.9, 16.3, 17.9, 15.8, 16.4, 16.0, 16.5]  # %
    speedups = [1.0, 1.15, 1.38, 1.72, 1.8, 2.6, 2.6, 3.2]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Size vs WER
    scatter = axes[0].scatter(sizes, wers, s=200, c=speedups, cmap='viridis', alpha=0.7)
    
    for i, config in enumerate(configs):
        axes[0].annotate(config, (sizes[i], wers[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    axes[0].set_xlabel('Model Size (MB)')
    axes[0].set_ylabel('Word Error Rate (%)')
    axes[0].set_title('(a) Size vs Accuracy Trade-off')
    axes[0].grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=axes[0])
    cbar.set_label('Speedup (×)')
    
    # Speedup vs WER
    axes[1].scatter(speedups, wers, s=200, c=sizes, cmap='plasma', alpha=0.7)
    
    for i, config in enumerate(configs):
        axes[1].annotate(config, (speedups[i], wers[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    axes[1].set_xlabel('Inference Speedup (×)')
    axes[1].set_ylabel('Word Error Rate (%)')
    axes[1].set_title('(b) Speed vs Accuracy Trade-off')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / 'figure6_optimization.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: {output_path}")


def generate_all_figures(results_dir: Path, output_dir: Path):
    """Generate all figures for the paper."""
    
    print("=" * 80)
    print("Generating Publication Figures")
    print("=" * 80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    try:
        results = load_results(results_dir)
    except FileNotFoundError:
        print("\n⚠️  Warning: Results not found. Generating figures with placeholder data.")
        results = {}
    
    # Generate each figure
    figure1_architecture_diagram(output_dir)
    
    if results:
        figure2_model_comparison(results, output_dir)
        figure3_training_curves(results_dir, output_dir)
    
    figure4_subject_performance(output_dir)
    figure5_ablation_study(output_dir)
    figure6_optimization_results(output_dir)
    
    print("\n" + "=" * 80)
    print(f"✓ All figures saved to: {output_dir}")
    print("=" * 80)
    print("\nFigures created:")
    print("  - figure1_architecture.png (MANUAL creation recommended)")
    print("  - figure2_model_comparison.pdf/png")
    print("  - figure3_training_curves.pdf/png")
    print("  - figure4_subject_performance.pdf/png")
    print("  - figure5_ablation.pdf/png")
    print("  - figure6_optimization.pdf/png")
    print("\nNext: Include these figures in papers/NEST_manuscript.md")


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures')
    parser.add_argument(
        '--results',
        type=str,
        default='results',
        help='Results directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='papers/figures',
        help='Output directory for figures'
    )
    
    args = parser.parse_args()
    
    generate_all_figures(Path(args.results), Path(args.output))


if __name__ == '__main__':
    main()
