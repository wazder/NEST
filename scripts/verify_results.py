#!/usr/bin/env python3
"""
Verify training results against paper claims.

Compares actual experimental results with expected values from the paper.
"""

import json
import argparse
from pathlib import Path
from typing import Dict
import pandas as pd


# Expected results from paper
EXPECTED_RESULTS = {
    'nest_conformer': {
        'wer': 15.8,
        'cer': 8.3,
        'bleu': 0.76,
        'perplexity': 4.7,
        'inference_time_ms': 18.0,
    },
    'nest_transformer': {
        'wer': 18.1,
        'cer': 9.6,
        'bleu': 0.71,
        'perplexity': 5.4,
        'inference_time_ms': 16.0,
    },
    'nest_rnn_t': {
        'wer': 19.7,
        'cer': 10.4,
        'bleu': 0.67,
        'perplexity': 6.1,
        'inference_time_ms': 22.0,
    },
    'nest_ctc': {
        'wer': 24.3,
        'cer': 12.7,
        'bleu': 0.58,
        'perplexity': 8.2,
        'inference_time_ms': 15.0,
    }
}

# Tolerance thresholds (% deviation allowed)
TOLERANCES = {
    'wer': 0.15,  # ±15%
    'cer': 0.15,
    'bleu': 0.10,
    'perplexity': 0.20,
    'inference_time_ms': 0.30,
}


def load_results(results_path: Path) -> Dict:
    """Load results from JSON file."""
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        return json.load(f)


def calculate_deviation(actual: float, expected: float) -> float:
    """Calculate percentage deviation."""
    if expected == 0:
        return float('inf')
    return abs(actual - expected) / expected


def verify_single_model(
    model_name: str,
    actual_results: Dict,
    expected_results: Dict
) -> Dict:
    """Verify results for a single model."""
    
    print(f"\n{'=' * 80}")
    print(f"Model: {model_name.upper()}")
    print(f"{'=' * 80}")
    
    verification = {
        'model': model_name,
        'metrics': {},
        'passed': True
    }
    
    for metric, expected_val in expected_results.items():
        actual_val = actual_results.get(metric)
        
        if actual_val is None:
            print(f"  ✗ {metric}: MISSING")
            verification['metrics'][metric] = {
                'status': 'missing',
                'expected': expected_val,
                'actual': None
            }
            verification['passed'] = False
            continue
        
        deviation = calculate_deviation(actual_val, expected_val)
        tolerance = TOLERANCES.get(metric, 0.15)
        passed = deviation <= tolerance
        
        status = "✓" if passed else "⚠️"
        print(f"  {status} {metric}:")
        print(f"      Expected: {expected_val:.3f}")
        print(f"      Actual:   {actual_val:.3f}")
        print(f"      Deviation: {deviation*100:.1f}%")
        
        verification['metrics'][metric] = {
            'status': 'pass' if passed else 'fail',
            'expected': expected_val,
            'actual': actual_val,
            'deviation': deviation
        }
        
        if not passed:
            verification['passed'] = False
    
    return verification


def generate_report(verifications: list) -> str:
    """Generate markdown report."""
    
    report = "# NEST Results Verification Report\n\n"
    report += "## Summary\n\n"
    
    # Summary table
    passed_count = sum(v['passed'] for v in verifications)
    total_count = len(verifications)
    
    report += f"- **Models Verified**: {total_count}\n"
    report += f"- **Passed**: {passed_count}\n"
    report += f"- **Failed**: {total_count - passed_count}\n\n"
    
    # Detailed results
    report += "## Detailed Results\n\n"
    
    for verification in verifications:
        model = verification['model']
        status = "✅ PASS" if verification['passed'] else "❌ FAIL"
        
        report += f"### {model.upper()} - {status}\n\n"
        report += "| Metric | Expected | Actual | Deviation | Status |\n"
        report += "|--------|----------|--------|-----------|--------|\n"
        
        for metric, data in verification['metrics'].items():
            if data['status'] == 'missing':
                report += f"| {metric} | {data['expected']:.3f} | MISSING | - | ❌ |\n"
            else:
                status_icon = "✅" if data['status'] == 'pass' else "⚠️"
                report += f"| {metric} | {data['expected']:.3f} | {data['actual']:.3f} | "
                report += f"{data['deviation']*100:.1f}% | {status_icon} |\n"
        
        report += "\n"
    
    # Recommendations
    report += "## Recommendations\n\n"
    
    failed = [v for v in verifications if not v['passed']]
    if failed:
        report += "The following models did not meet expected performance:\n\n"
        for v in failed:
            report += f"- **{v['model']}**: "
            failed_metrics = [m for m, d in v['metrics'].items() if d['status'] == 'fail']
            report += f"Poor performance on {', '.join(failed_metrics)}\n"
        
        report += "\n**Suggested Actions**:\n"
        report += "1. Check hyperparameters match paper specifications\n"
        report += "2. Verify preprocessing pipeline is correct\n"
        report += "3. Ensure training converged (check loss curves)\n"
        report += "4. Try training for more epochs\n"
        report += "5. Check for data leakage or errors\n"
    else:
        report += "✅ All models meet expected performance! Ready for paper submission.\n"
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Verify NEST results')
    parser.add_argument(
        '--results',
        type=str,
        default='results/results.json',
        help='Path to results JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/verification_report.md',
        help='Output path for verification report'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Use stricter tolerance (5 percent instead of 15 percent)'
    )
    
    args = parser.parse_args()
    
    # Adjust tolerances if strict mode
    if args.strict:
        for key in TOLERANCES:
            TOLERANCES[key] = 0.05
    
    print("=" * 80)
    print("NEST Results Verification")
    print("=" * 80)
    print(f"Results file: {args.results}")
    print(f"Tolerance mode: {'STRICT (±5%)' if args.strict else 'NORMAL (±15%)'}")
    print("=" * 80)
    
    # Load results
    try:
        results = load_results(Path(args.results))
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nNo results found. Have you run training yet?")
        print("Run: python scripts/train_zuco_full.py")
        return 1
    
    # Verify each model
    verifications = []
    
    for model_name, expected in EXPECTED_RESULTS.items():
        actual = results.get('results', {}).get(model_name, {})
        
        if not actual:
            print(f"\n⚠️  No results found for {model_name}")
            continue
        
        verification = verify_single_model(model_name, actual, expected)
        verifications.append(verification)
    
    # Generate report
    report = generate_report(verifications)
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\n{'=' * 80}")
    print(f"✓ Verification report saved to: {output_path}")
    print(f"{'=' * 80}")
    
    # Print summary
    passed = sum(v['passed'] for v in verifications)
    total = len(verifications)
    
    print(f"\nSummary: {passed}/{total} models passed verification")
    
    if passed == total:
        print("\n🎉 All models meet expected performance!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} model(s) need attention")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
