#!/usr/bin/env python3
"""
Generate synthetic EEG data for testing NEST pipeline.

This creates small synthetic dataset matching ZuCo format
for quick testing without downloading 18GB of real data.
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
import argparse


def generate_synthetic_eeg_sample(
    n_channels=105,
    n_timepoints=500,
    sentence="The quick brown fox jumps over the lazy dog",
    subject_id=1
):
    """
    Generate synthetic EEG data matching ZuCo format.
    
    Args:
        n_channels: Number of EEG channels
        n_timepoints: Number of time points
        sentence: Text being "read"
        subject_id: Subject identifier
    
    Returns:
        Dictionary with ZuCo-like structure
    """
    words = sentence.split()
    n_words = len(words)
    
    # Generate synthetic EEG data with some structure
    # Real EEG has frequency bands (theta: 4-8Hz, alpha: 8-13Hz, beta: 13-30Hz)
    eeg_data = np.zeros((n_channels, n_timepoints))
    
    # Add some realistic-looking patterns
    t = np.linspace(0, 1, n_timepoints)
    
    for ch in range(n_channels):
        # Mix of frequency components
        theta = 0.3 * np.sin(2 * np.pi * 6 * t)  # 6 Hz theta
        alpha = 0.5 * np.sin(2 * np.pi * 10 * t)  # 10 Hz alpha
        beta = 0.2 * np.sin(2 * np.pi * 20 * t)  # 20 Hz beta
        noise = 0.1 * np.random.randn(n_timepoints)
        
        eeg_data[ch] = theta + alpha + beta + noise
    
    # Word-level data
    word_data = []
    
    for i, word in enumerate(words):
        word_info = {
            'word': word,
            'word_idx': i,
            'fix_dur': np.random.randint(200, 400),  # Fixation duration (ms)
            'FFD': np.random.randint(180, 350),  # First fixation duration
            'GD': np.random.randint(200, 450),  # Gaze duration
            'TRT': np.random.randint(220, 500),  # Total reading time
            'nFix': np.random.randint(1, 3),  # Number of fixations
        }
        word_data.append(word_info)
    
    # Construct output matching ZuCo format
    output = {
        'subject_id': subject_id,
        'sentence': sentence,
        'sentence_id': 1,
        'content': sentence,
        'word_level_data': word_data,
        'eeg_data': eeg_data,
        'sampling_rate': 500,  # 500 Hz
        'channel_names': [f'Ch{i+1}' for i in range(n_channels)],
    }
    
    return output


def generate_synthetic_dataset(
    output_dir: Path,
    n_subjects=12,
    n_sentences_per_subject=50,
    task_name='task1_SR'
):
    """Generate full synthetic dataset."""
    
    print(f"Generating synthetic ZuCo dataset: {task_name}")
    print(f"  Subjects: {n_subjects}")
    print(f"  Sentences per subject: {n_sentences_per_subject}")
    
    # Sample sentences from children's books (simple for testing)
    sample_sentences = [
        "The sun was shining in the sky",
        "A cat sat on the mat and purred",
        "Birds flew high above the trees",
        "The dog ran quickly through the park",
        "Children played happily in the garden",
        "Rain fell softly on the roof",
        "The moon glowed brightly at night",
        "Fish swam swiftly in the pond",
        "Flowers bloomed beautifully in spring",
        "The wind blew gently through the leaves",
        "Stars twinkled in the dark sky",
        "A butterfly landed on a flower",
        "The train moved slowly down the track",
        "Snow covered the ground in winter",
        "A rabbit hopped quickly across the field",
        "The river flowed peacefully downstream",
        "Thunder rumbled loudly in the distance",
        "A bee buzzed around the roses",
        "The boat sailed smoothly on the lake",
        "Leaves fell quietly from the oak tree",
    ]
    
    task_dir = output_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    
    for subject_id in range(1, n_subjects + 1):
        print(f"  Generating data for Subject {subject_id:02d}...")
        
        subject_data = {
            'subject_id': f'YA{subject_id:02d}',
            'age': 20 + np.random.randint(0, 10),
            'gender': 'M' if subject_id % 2 == 0 else 'F',
            'handedness': 'R',
            'native_language': 'English',
            'sentence_data': []
        }
        
        for sent_idx in range(n_sentences_per_subject):
            # Cycle through sample sentences
            sentence = sample_sentences[sent_idx % len(sample_sentences)]
            
            # Generate EEG data for this sentence
            sample = generate_synthetic_eeg_sample(
                sentence=sentence,
                subject_id=subject_id
            )
            sample['sentence_id'] = sent_idx + 1
            
            subject_data['sentence_data'].append(sample)
        
        # Save subject data
        subject_file = task_dir / f'results{subject_id}.mat'
        sio.savemat(subject_file, {'subject_data': subject_data})
        
        all_data.append(subject_data)
        print(f"    ✓ Saved {subject_file.name}")
    
    # Save metadata
    metadata = {
        'task_name': task_name,
        'n_subjects': n_subjects,
        'n_sentences': n_sentences_per_subject,
        'sampling_rate': 500,
        'n_channels': 105,
        'description': 'Synthetic EEG data for testing NEST pipeline',
        'note': 'This is NOT real ZuCo data - replace with actual dataset for real experiments'
    }
    
    metadata_file = task_dir / 'metadata.mat'
    sio.savemat(metadata_file, metadata)
    
    print(f"\n✓ Synthetic dataset created: {task_dir}")
    print(f"  Total files: {len(all_data)} subjects")
    print(f"  Total sentences: {n_subjects * n_sentences_per_subject}")
    print(f"\n⚠️  NOTE: This is SYNTHETIC data for testing only!")
    print(f"  For real experiments, download actual ZuCo dataset from:")
    print(f"  https://osf.io/q3zws/")
    
    return task_dir


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic ZuCo data')
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/zuco',
        help='Output directory'
    )
    parser.add_argument(
        '--subjects',
        type=int,
        default=12,
        help='Number of subjects'
    )
    parser.add_argument(
        '--sentences',
        type=int,
        default=50,
        help='Sentences per subject'
    )
    parser.add_argument(
        '--task',
        type=str,
        default='task1_SR',
        choices=['task1_SR', 'task2_NR', 'task3_TSR'],
        help='Task name'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    generate_synthetic_dataset(
        output_dir,
        n_subjects=args.subjects,
        n_sentences_per_subject=args.sentences,
        task_name=args.task
    )
    
    print("\nNext step: Run preprocessing")
    print(f"  python scripts/train_zuco_full.py --preprocess-only")


if __name__ == '__main__':
    main()
