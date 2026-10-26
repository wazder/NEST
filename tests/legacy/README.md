# Legacy tests

These tests are archived. pytest does not collect them: `pyproject.toml`
lists `tests/legacy` in `norecursedirs`.

## Why

The tests were written for a planned v1 API (raw EEG into `src/models/nest.py`)
that `src/` never got in this form. They call functions and classes that do
not exist (`MultiplicativeAttention`, `DANNAdapter`, `BenchmarkSuite`,
`ThresholdRejecter`, ...), or real classes with a different design (other constructor arguments,
other return values, a different workflow). Before the move, the suite stopped
at three import errors and twelve failures.

Tests that only used other names for code that exists stay live. They now call
the real names, for example `calculate_wer(ref, hyp)` became
`word_error_rate([ref], [hyp])` and `NESTWithCTC` became `NEST_CTC`. A test
moved here only when making it pass would change what it asserts or need new
code in `src/`.

The tests are kept verbatim, each under its original class. Only the module
docstring is new. To bring one back, rewrite it against the current API and
move it into `tests/unit` or `tests/integration`.

## Contents

| File | Origin | Tests | Missing or different API |
|---|---|---|---|
| `test_models.py` | `tests/unit/test_models.py` | `test_additive_attention`, `test_multiplicative_attention` | `MultiplicativeAttention` does not exist. `AdditiveAttention` needs `query_dim`, `key_dim` and a 3D query, and returns a 3D context. |
| `test_preprocessing.py` | `tests/unit/test_preprocessing.py` | `test_variance_rejection` | `ThresholdRejecter` does not exist. `AutomaticArtifactRejection.reject_by_variance` uses a z-score and returns a boolean mask, not a list. |
| | | `test_pca_reduction` | `ElectrodeSelector` has no `method="pca"`. `ChannelOptimizer.optimize_by_pca` takes samples x channels and returns samples x components. |
| | | `test_basic_split`, `test_subject_aware_split`, `test_temporal_split` | `train_val_test_split`, `subject_aware_split` and `temporal_split` do not exist. `EEGDataSplitter` takes `(data, labels)` arrays, not lists of samples, and does not return subject IDs or timestamps. |
| | | `test_pipeline_execution`, `test_pipeline_reproducibility` | `PreprocessingPipeline` takes a YAML path, has no `process()` and no `random_state`. |
| `test_training.py` | `tests/unit/test_training.py` | `test_metrics_tracker`, `test_metrics_tracker_reset` | `MetricsTracker.update` takes a dict. There is no `get_average`, `get_latest` or per-metric value list. |
| | | 4 tests in `TestCheckpointManager` | `CheckpointManager` takes the model and saves its state itself. `save_checkpoint` takes `(epoch, metrics)`, not a state dict and `metric_value`. There is no `get_best_checkpoint`. |
| | | 4 tests in `TestTrainer` | `Trainer` takes an optimizer, criterion and device. There is no config, `checkpoint_dir`, `training_step` or `validation_step`. |
| | | 3 tests in `TestBeamSearch` | `BeamSearchDecoder` takes a model and token IDs, not a tokenizer, and returns `Hypothesis` objects, not strings. |
| | | 3 tests in `TestBenchmark` | `BenchmarkSuite` does not exist. `benchmark_suite()` runs a model on a data loader. |
| | | `test_inference_profiling` | `profile_model` does not exist. `ComparisonBenchmark.compare_models` reports `latency_ms` and `memory_mb`, but for a dict of models and an input shape. |
| `test_workflows.py` | `tests/integration/test_workflows.py` | `test_full_pipeline_preprocessing_to_model` | `PreprocessingPipeline.process()` does not exist. |
| | | `test_training_and_evaluation_workflow` | Same `Trainer` and `CheckpointManager` gaps as above. |
| | | `test_different_encoder_decoder_combinations` | `ModelFactory.create_model` takes one config dict, not `model_type` and keyword arguments. |
| | | `test_domain_adaptation_dann`, `test_coral_adaptation` | `DANNAdapter` and `CORALAdapter` do not exist. `DomainAdversarialNetwork` and `CORAL` work on features, not on a wrapped model. |
| | | `test_adversarial_training` | `generate_adversarial_examples` is a method of `AdversarialTrainer`, which needs a criterion. |
| | | `test_realtime_inference_latency` | `RealtimeInference` does not exist. `StreamingInference` takes a `StreamConfig` and has no `infer()`. |

33 tests in total.
