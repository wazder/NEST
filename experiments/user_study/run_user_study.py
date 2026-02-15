#!/usr/bin/env python3
"""
NEST User Study Implementation

This module provides the implementation for running the NEST user study,
including data collection, real-time model inference, and result analysis.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, asdict

# Import NEST components
from src.models.factory import ModelFactory
from src.preprocessing.pipeline import PreprocessingPipeline
from src.evaluation.realtime_inference import RealTimeInference
from src.training.metrics import compute_wer, compute_cer, compute_bleu


@dataclass
class ParticipantInfo:
    """Store participant metadata."""
    participant_id: str
    age: int
    gender: str
    handedness: str
    session_number: int
    date: str
    start_time: str


@dataclass
class TrialResult:
    """Store results from a single trial."""
    trial_id: int
    target_text: str
    predicted_text: str
    wer: float
    cer: float
    bleu: float
    inference_time_ms: float
    confidence: float
    timestamp: str


class UserStudySession:
    """Manage a single user study session."""
    
    def __init__(
        self,
        participant_id: str,
        session_number: int,
        output_dir: str,
        model_path: Optional[str] = None,
        config_path: str = 'configs/model.yaml'
    ):
        """
        Initialize user study session.
        
        Args:
            participant_id: Unique participant identifier
            session_number: Session number (1-4)
            output_dir: Directory to save results
            model_path: Path to pre-trained model (optional)
            config_path: Path to model configuration
        """
        self.participant_id = participant_id
        self.session_number = session_number
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session directory
        self.session_dir = self.output_dir / participant_id / f"session_{session_number}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocessing = PreprocessingPipeline()
        
        # Load model if provided
        self.model = None
        if model_path:
            self.load_model(model_path)
            
        # Session data
        self.participant_info = None
        self.trial_results = []
        self.eeg_data = []
        
    def load_model(self, model_path: str) -> None:
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model from checkpoint config
        self.model = ModelFactory.create(
            checkpoint['config']['model_type'],
            **checkpoint['config']['model_args']
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Loaded model from {model_path}")
        
    def collect_participant_info(self) -> ParticipantInfo:
        """Collect participant demographic information."""
        print("\n" + "=" * 80)
        print("PARTICIPANT INFORMATION")
        print("=" * 80)
        
        if self.session_number == 1:
            # Collect full demographics on first session
            age = int(input("Age: "))
            gender = input("Gender (M/F/Other): ")
            handedness = input("Handedness (L/R): ")
        else:
            # Load from previous session
            prev_session = self.output_dir / self.participant_id / "session_1" / "participant_info.json"
            with open(prev_session, 'r') as f:
                prev_info = json.load(f)
            age = prev_info['age']
            gender = prev_info['gender']
            handedness = prev_info['handedness']
            
        self.participant_info = ParticipantInfo(
            participant_id=self.participant_id,
            age=age,
            gender=gender,
            handedness=handedness,
            session_number=self.session_number,
            date=datetime.now().strftime('%Y-%m-%d'),
            start_time=datetime.now().strftime('%H:%M:%S')
        )
        
        # Save participant info
        with open(self.session_dir / 'participant_info.json', 'w') as f:
            json.dump(asdict(self.participant_info), f, indent=2)
            
        return self.participant_info
        
    def run_reading_task(self, stimuli: List[str]) -> np.ndarray:
        """
        Run reading task to collect training data.
        
        Args:
            stimuli: List of sentences to present
            
        Returns:
            Collected EEG data
        """
        print("\n" + "=" * 80)
        print(f"READING TASK - {len(stimuli)} sentences")
        print("=" * 80)
        print("Instructions:")
        print("- Read each sentence silently")
        print("- Maintain focus on the center of the screen")
        print("- Press ENTER when you've finished reading each sentence")
        print("- Take breaks as needed")
        print("=" * 80)
        input("Press ENTER to begin...")
        
        collected_data = []
        
        for i, sentence in enumerate(stimuli, 1):
            print(f"\n[Trial {i}/{len(stimuli)}]")
            print("─" * 80)
            print(f"\n{sentence}\n")
            print("─" * 80)
            
            # Simulate EEG collection
            # In real implementation, this would interface with EEG hardware
            print("Recording EEG... (5 seconds)")
            time.sleep(1)  # Simulate recording time
            
            # Collect EEG data (simulated)
            eeg_segment = self._simulate_eeg_recording(duration=5.0)
            
            collected_data.append({
                'sentence': sentence,
                'eeg': eeg_segment,
                'trial_id': i,
                'timestamp': datetime.now().isoformat()
            })
            
            input("Press ENTER for next trial...")
            
            # Breaks every 30 trials
            if i % 30 == 0 and i < len(stimuli):
                print("\n⏸ Break time! Rest for 2 minutes.")
                print("Press ENTER when ready to continue...")
                input()
                
        # Save collected data
        save_path = self.session_dir / 'reading_task_data.npz'
        np.savez_compressed(
            save_path,
            data=collected_data,
            participant_id=self.participant_id,
            session=self.session_number
        )
        
        print(f"\n✓ Saved reading task data to {save_path}")
        
        return collected_data
        
    def run_text_generation_task(
        self,
        stimuli: List[str],
        model_type: str = "subject_specific"
    ) -> List[TrialResult]:
        """
        Run text generation task.
        
        Args:
            stimuli: Target sentences
            model_type: "subject_specific" or "subject_independent"
            
        Returns:
            List of trial results
        """
        print("\n" + "=" * 80)
        print(f"TEXT GENERATION TASK - {model_type.upper()}")
        print("=" * 80)
        print("Instructions:")
        print("1. Read the target sentence (shown for 3 seconds)")
        print("2. Mentally rehearse the sentence")
        print("3. When prompted, 'speak' the sentence silently in your mind")
        print("4. The system will decode your EEG signals")
        print("=" * 80)
        input("Press ENTER to begin...")
        
        results = []
        
        for i, target in enumerate(stimuli, 1):
            print(f"\n[Trial {i}/{len(stimuli)}]")
            print("─" * 80)
            print("Target (3 seconds):")
            print(f"\n{target}\n")
            time.sleep(3)
            
            print("─" * 80)
            input("Ready to generate? Press ENTER...")
            
            print("🧠 Thinking... (recording EEG)")
            
            # Record EEG
            start_time = time.time()
            eeg_data = self._simulate_eeg_recording(duration=len(target.split()) * 0.5)
            
            # Preprocess and decode
            processed_eeg = self.preprocessing.process(eeg_data)
            
            if self.model is not None:
                # Run inference
                with torch.no_grad():
                    eeg_tensor = torch.FloatTensor(processed_eeg).unsqueeze(0).to(self.device)
                    output = self.model(eeg_tensor)
                    predicted_text = self._decode_output(output)
            else:
                # Fallback to simple simulation
                predicted_text = self._simulate_prediction(target)
                
            inference_time = (time.time() - start_time) * 1000
            
            # Compute metrics
            wer = compute_wer([predicted_text], [target])
            cer = compute_cer([predicted_text], [target])
            bleu = compute_bleu([predicted_text], [target])
            
            # Display result
            print("\n📝 Decoded text:")
            print(f"'{predicted_text}'")
            print(f"\n📊 Metrics: WER={wer:.3f}, CER={cer:.3f}, BLEU={bleu:.3f}")
            print(f"⏱  Inference time: {inference_time:.1f} ms")
            
            # Store result
            result = TrialResult(
                trial_id=i,
                target_text=target,
                predicted_text=predicted_text,
                wer=wer,
                cer=cer,
                bleu=bleu,
                inference_time_ms=inference_time,
                confidence=0.85,  # Placeholder
                timestamp=datetime.now().isoformat()
            )
            
            results.append(result)
            
            input("\nPress ENTER for next trial...")
            
        # Save results
        results_df = pd.DataFrame([asdict(r) for r in results])
        results_df.to_csv(
            self.session_dir / f'text_generation_{model_type}.csv',
            index=False
        )
        
        # Print summary
        avg_wer = np.mean([r.wer for r in results])
        avg_cer = np.mean([r.cer for r in results])
        avg_bleu = np.mean([r.bleu for r in results])
        avg_time = np.mean([r.inference_time_ms for r in results])
        
        print("\n" + "=" * 80)
        print("SESSION SUMMARY")
        print("=" * 80)
        print(f"Average WER:  {avg_wer:.3f}")
        print(f"Average CER:  {avg_cer:.3f}")
        print(f"Average BLEU: {avg_bleu:.3f}")
        print(f"Average Time: {avg_time:.1f} ms")
        print("=" * 80)
        
        return results
        
    def administer_questionnaire(self, questionnaire_type: str) -> Dict:
        """
        Administer questionnaire.
        
        Args:
            questionnaire_type: Type of questionnaire (SUS, NASA-TLX, etc.)
            
        Returns:
            Questionnaire responses
        """
        print("\n" + "=" * 80)
        print(f"{questionnaire_type.upper()} QUESTIONNAIRE")
        print("=" * 80)
        
        responses = {}
        
        if questionnaire_type == "SUS":
            questions = self._get_sus_questions()
        elif questionnaire_type == "NASA-TLX":
            questions = self._get_nasa_tlx_questions()
        elif questionnaire_type == "SATISFACTION":
            questions = self._get_satisfaction_questions()
        else:
            questions = {}
            
        for q_id, question in questions.items():
            print(f"\n{question['text']}")
            print(f"Scale: {question['scale']}")
            response = input("Your rating: ")
            responses[q_id] = {
                'question': question['text'],
                'response': response,
                'timestamp': datetime.now().isoformat()
            }
            
        # Save responses
        with open(self.session_dir / f'{questionnaire_type.lower()}_responses.json', 'w') as f:
            json.dump(responses, f, indent=2)
            
        return responses
        
    def _get_sus_questions(self) -> Dict:
        """Get System Usability Scale questions."""
        return {
            'q1': {'text': "I think that I would like to use this system frequently.", 'scale': '1-5 (Strongly Disagree - Strongly Agree)'},
            'q2': {'text': "I found the system unnecessarily complex.", 'scale': '1-5'},
            'q3': {'text': "I thought the system was easy to use.", 'scale': '1-5'},
            'q4': {'text': "I think that I would need the support of a technical person to use this system.", 'scale': '1-5'},
            'q5': {'text': "I found the various functions in this system were well integrated.", 'scale': '1-5'},
            'q6': {'text': "I thought there was too much inconsistency in this system.", 'scale': '1-5'},
            'q7': {'text': "I would imagine that most people would learn to use this system very quickly.", 'scale': '1-5'},
            'q8': {'text': "I found the system very cumbersome to use.", 'scale': '1-5'},
            'q9': {'text': "I felt very confident using the system.", 'scale': '1-5'},
            'q10': {'text': "I needed to learn a lot of things before I could get going with this system.", 'scale': '1-5'},
        }
        
    def _get_nasa_tlx_questions(self) -> Dict:
        """Get NASA Task Load Index questions."""
        return {
            'mental': {'text': "Mental Demand: How mentally demanding was the task?", 'scale': '1-20 (Low - High)'},
            'physical': {'text': "Physical Demand: How physically demanding was the task?", 'scale': '1-20'},
            'temporal': {'text': "Temporal Demand: How hurried or rushed was the pace?", 'scale': '1-20'},
            'performance': {'text': "Performance: How successful were you in accomplishing the task?", 'scale': '1-20 (Poor - Good)'},
            'effort': {'text': "Effort: How hard did you have to work?", 'scale': '1-20 (Low - High)'},
            'frustration': {'text': "Frustration: How insecure, discouraged, irritated, stressed?", 'scale': '1-20 (Low - High)'},
        }
        
    def _get_satisfaction_questions(self) -> Dict:
        """Get satisfaction questions."""
        return {
            'accuracy': {'text': "The system accurately captured my intended text.", 'scale': '1-7 (Strongly Disagree - Strongly Agree)'},
            'speed': {'text': "The system's speed was acceptable.", 'scale': '1-7'},
            'control': {'text': "I felt in control of the system.", 'scale': '1-7'},
            'frustration': {'text': "The system was frustrating to use.", 'scale': '1-7 (Reverse scored)'},
            'recommend': {'text': "I would recommend this system to others.", 'scale': '1-7'},
        }
        
    def _simulate_eeg_recording(self, duration: float = 5.0, n_channels: int = 32) -> np.ndarray:
        """
        Simulate EEG recording.
        
        In real implementation, this would interface with actual EEG hardware.
        
        Args:
            duration: Recording duration in seconds
            n_channels: Number of EEG channels
            
        Returns:
            Simulated EEG data (channels × timepoints)
        """
        sample_rate = 500  # Hz
        n_samples = int(duration * sample_rate)
        
        # Generate realistic EEG-like data
        # Mix of different frequency components
        t = np.linspace(0, duration, n_samples)
        eeg = np.zeros((n_channels, n_samples))
        
        for ch in range(n_channels):
            # Alpha (8-13 Hz)
            eeg[ch] += 10 * np.sin(2 * np.pi * 10 * t + np.random.rand() * 2 * np.pi)
            # Beta (13-30 Hz)
            eeg[ch] += 5 * np.sin(2 * np.pi * 20 * t + np.random.rand() * 2 * np.pi)
            # Noise
            eeg[ch] += np.random.randn(n_samples) * 2
            
        return eeg
        
    def _decode_output(self, output: torch.Tensor) -> str:
        """Decode model output to text."""
        # Placeholder - implement actual decoding logic
        # This would use beam search or greedy decoding
        return "decoded text placeholder"
        
    def _simulate_prediction(self, target: str) -> str:
        """Simulate prediction with controlled errors."""
        words = target.split()
        predicted_words = []
        
        for word in words:
            # 80% chance of correct prediction
            if np.random.rand() < 0.8:
                predicted_words.append(word)
            else:
                # Introduce error
                if len(word) > 3:
                    # Character substitution
                    chars = list(word)
                    idx = np.random.randint(len(chars))
                    chars[idx] = chr(ord('a') + np.random.randint(26))
                    predicted_words.append(''.join(chars))
                else:
                    predicted_words.append(word)
                    
        return ' '.join(predicted_words)
        
    def generate_session_report(self) -> str:
        """Generate markdown report for session."""
        report = f"""# Session {self.session_number} Report

**Participant ID**: {self.participant_id}  
**Date**: {datetime.now().strftime('%Y-%m-%d')}  
**Session**: {self.session_number}/4

## Session Summary

[Session-specific summary would go here based on session type]

## Data Collected

- EEG recordings: {len(self.eeg_data)} trials
- Behavioral responses: {len(self.trial_results)} trials
- Questionnaires: [List questionnaires completed]

## Next Steps

Session {self.session_number + 1 if self.session_number < 4 else 'Complete'}

---
Generated by NEST User Study System
"""
        
        report_path = self.session_dir / 'session_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
            
        return report


class UserStudyAnalyzer:
    """Analyze user study results across all participants."""
    
    def __init__(self, study_dir: str):
        """
        Initialize analyzer.
        
        Args:
            study_dir: Directory containing all participant data
        """
        self.study_dir = Path(study_dir)
        self.participants = list(self.study_dir.glob('P*'))
        
    def analyze_performance_metrics(self) -> pd.DataFrame:
        """Analyze performance metrics across participants."""
        all_results = []
        
        for participant_dir in self.participants:
            participant_id = participant_dir.name
            
            for session_dir in participant_dir.glob('session_*'):
                session_num = int(session_dir.name.split('_')[1])
                
                # Load text generation results
                for result_file in session_dir.glob('text_generation_*.csv'):
                    model_type = result_file.stem.split('_')[-1]
                    df = pd.read_csv(result_file)
                    df['participant_id'] = participant_id
                    df['session'] = session_num
                    df['model_type'] = model_type
                    all_results.append(df)
                    
        if all_results:
            combined = pd.concat(all_results, ignore_index=True)
            return combined
        else:
            return pd.DataFrame()
            
    def compute_aggregate_statistics(self) -> Dict:
        """Compute aggregate statistics across all participants."""
        df = self.analyze_performance_metrics()
        
        if df.empty:
            return {}
            
        stats = {
            'overall': {
                'mean_wer': df['wer'].mean(),
                'std_wer': df['wer'].std(),
                'mean_cer': df['cer'].mean(),
                'mean_bleu': df['bleu'].mean(),
                'mean_inference_time_ms': df['inference_time_ms'].mean()
            },
            'by_model_type': df.groupby('model_type').agg({
                'wer': ['mean', 'std'],
                'cer': ['mean', 'std'],
                'bleu': ['mean', 'std']
            }).to_dict(),
            'by_session': df.groupby('session').agg({
                'wer': ['mean', 'std'],
                'cer': ['mean', 'std']
            }).to_dict()
        }
        
        return stats
        
    def generate_final_report(self, output_path: str) -> None:
        """Generate final study report."""
        stats = self.compute_aggregate_statistics()
        
        report = f"""# NEST User Study - Final Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total Participants**: {len(self.participants)}

## Performance Summary

### Overall Results

- **Mean WER**: {stats['overall']['mean_wer']:.3f} ± {stats['overall']['std_wer']:.3f}
- **Mean CER**: {stats['overall']['mean_cer']:.3f}
- **Mean BLEU**: {stats['overall']['mean_bleu']:.3f}
- **Mean Inference Time**: {stats['overall']['mean_inference_time_ms']:.1f} ms

### By Model Type

[Performance comparison between subject-specific and subject-independent models]

### Learning Curve

[Performance improvement across sessions]

## Statistical Tests

[ANOVA/t-test results]

## Qualitative Findings

[Themes from interviews and questionnaires]

## Conclusions

[Study conclusions and implications]

---
For detailed analysis, see individual participant reports and analysis notebooks.
"""
        
        with open(output_path, 'w') as f:
            f.write(report)
            
        print(f"✓ Generated final report: {output_path}")


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run NEST user study session')
    parser.add_argument('--participant', type=str, required=True, help='Participant ID')
    parser.add_argument('--session', type=int, required=True, help='Session number (1-4)')
    parser.add_argument('--output', type=str, default='user_study_data', help='Output directory')
    parser.add_argument('--model', type=str, help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    # Create session
    session = UserStudySession(
        participant_id=args.participant,
        session_number=args.session,
        output_dir=args.output,
        model_path=args.model
    )
    
    # Collect participant info
    session.collect_participant_info()
    
    # Run session-specific tasks
    if args.session == 1:
        # Session 1: Baseline data collection
        stimuli = [
            "The quick brown fox jumps over the lazy dog.",
            "She sells seashells by the seashore.",
            # ... more stimuli
        ]
        session.run_reading_task(stimuli)
        session.administer_questionnaire("SATISFACTION")
        
    elif args.session == 2:
        # Session 2: Calibration
        calibration_stimuli = [...]  # Load calibration sentences
        session.run_reading_task(calibration_stimuli)
        
    elif args.session == 3:
        # Session 3: Performance evaluation
        test_stimuli = [...]  # Load test sentences
        session.run_text_generation_task(test_stimuli, model_type="subject_specific")
        session.run_text_generation_task(test_stimuli, model_type="subject_independent")
        session.administer_questionnaire("SUS")
        
    elif args.session == 4:
        # Session 4: Final evaluation
        free_form_prompts = [...]
        # Run free-form generation
        session.administer_questionnaire("NASA-TLX")
        session.administer_questionnaire("SATISFACTION")
        
    # Generate report
    session. session_report()
    
    print("\n✓ Session complete!")


if __name__ == '__main__':
    main()
