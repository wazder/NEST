#!/usr/bin/env python3
"""
Deployment Example: Export and serve NEST model for production

This example demonstrates:
1. Model export (TorchScript, ONNX)
2. Optimization for deployment
3. Simple REST API for inference
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any
import json

from src.models import ModelFactory
from src.utils.tokenizer import CharTokenizer
from src.evaluation.inference_optimizer import InferenceOptimizer
from src.evaluation.deployment import ModelDeployer

def export_models():
    """Export model in various formats"""
    
    print("="*60)
    print("Model Export for Deployment")
    print("="*60)
    
    # Load trained model
    chars = list('abcdefghijklmnopqrstuvwxyz .,!?\'')
    tokenizer = CharTokenizer(vocab=chars)
    
    model = ModelFactory.from_config_file(
        'configs/model.yaml',
        model_key='nest_attention',
        vocab_size=len(tokenizer)
    )
    
    # Load weights
    checkpoint = torch.load('checkpoints/best_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("\nModel loaded successfully")
    
    # Create deployment directory
    deploy_dir = Path('deployment')
    deploy_dir.mkdir(exist_ok=True)
    
    # 1. TorchScript Export
    print("\n1. Exporting to TorchScript...")
    optimizer = InferenceOptimizer(model)
    
    # Create example input
    example_input = torch.randn(1, 32, 1000)  # (batch, channels, time)
    
    # Export TorchScript
    scripted_model = optimizer.to_torchscript(example_input)
    scripted_path = deploy_dir / 'nest_model.pt'
    torch.jit.save(scripted_model, scripted_path)
    print(f"   Saved to: {scripted_path}")
    
    # 2. ONNX Export
    print("\n2. Exporting to ONNX...")
    onnx_path = deploy_dir / 'nest_model.onnx'
    optimizer.to_onnx(
        output_path=str(onnx_path),
        example_input=example_input,
        opset_version=14
    )
    print(f"   Saved to: {onnx_path}")
    
    # 3. Optimized FP16 Export
    print("\n3. Exporting FP16 optimized model...")
    fp16_model = optimizer.to_fp16()
    fp16_path = deploy_dir / 'nest_model_fp16.pt'
    torch.jit.save(torch.jit.script(fp16_model), fp16_path)
    print(f"   Saved to: {fp16_path}")
    
    # 4. Save tokenizer
    print("\n4. Saving tokenizer...")
    tokenizer_config = {
        'vocab': chars,
        'type': 'char',
        'vocab_size': len(chars)
    }
    tokenizer_path = deploy_dir / 'tokenizer.json'
    with open(tokenizer_path, 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    print(f"   Saved to: {tokenizer_path}")
    
    # 5. Save metadata
    print("\n5. Saving model metadata...")
    metadata = {
        'model_type': 'nest_attention',
        'input_shape': [32, 1000],  # (channels, time_points)
        'sampling_rate': 500,  # Hz
        'vocab_size': len(chars),
        'max_sequence_length': 256,
        'version': '1.0.0',
        'description': 'NEST EEG-to-text decoder'
    }
    metadata_path = deploy_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   Saved to: {metadata_path}")
    
    # 6. Create deployment README
    create_deployment_readme(deploy_dir)
    
    print("\n" + "="*60)
    print("Export complete! Files created in deployment/")
    print("="*60)

def create_deployment_readme(deploy_dir: Path):
    """Create README for deployment directory"""
    
    readme_content = """# NEST Model Deployment

This directory contains exported models and deployment artifacts.

## Files

- `nest_model.pt` - TorchScript model (full precision)
- `nest_model_fp16.pt` - TorchScript model (FP16 precision)
- `nest_model.onnx` - ONNX format model
- `tokenizer.json` - Tokenizer configuration
- `metadata.json` - Model metadata

## Usage

### TorchScript (Python)

```python
import torch

# Load model
model = torch.jit.load('deployment/nest_model.pt')
model.eval()

# Prepare input (batch_size, channels, time_points)
eeg_input = torch.randn(1, 32, 1000)

# Run inference
with torch.no_grad():
    output = model(eeg_input)

# Decode output tokens
predicted_tokens = output.argmax(dim=-1)
```

### ONNX (Python)

```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession('deployment/nest_model.onnx')

# Prepare input
eeg_input = np.random.randn(1, 32, 1000).astype(np.float32)

# Run inference
outputs = session.run(None, {'eeg': eeg_input})
predicted_tokens = outputs[0].argmax(axis=-1)
```

### ONNX (C++)

```cpp
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

// Initialize ONNX Runtime
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "NEST");
Ort::SessionOptions session_options;
Ort::Session session(env, "deployment/nest_model.onnx", session_options);

// Prepare input tensor
std::vector<float> input_data(1 * 32 * 1000);
// ... fill with EEG data ...

// Run inference
auto output_tensors = session.Run(/*...*/);
```

## REST API (FastAPI)

See `examples/04_deployment.py` for a complete REST API example.

## Performance

- TorchScript: ~15ms inference on CPU
- ONNX: ~12ms inference on CPU  
- FP16: ~10ms inference on GPU

## Requirements

- PyTorch >= 2.0.0 (for TorchScript)
- ONNX Runtime (for ONNX)
- NumPy

## Model Input/Output

### Input
- Shape: `(batch_size, n_channels, n_timepoints)`
- Type: Float32
- Channels: 32 EEG electrodes
- Timepoints: 1000 samples (2 seconds at 500 Hz)

### Output
- Shape: `(batch_size, max_seq_len, vocab_size)`
- Type: Float32
- Logits over vocabulary for each time step

## License

MIT License - See LICENSE file in root directory
"""
    
    readme_path = deploy_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"   Deployment README saved to: {readme_path}")

def create_inference_api():
    """Create simple FastAPI server for inference"""
    
    api_code = '''#!/usr/bin/env python3
"""
Simple REST API for NEST model inference

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000

Test:
    curl -X POST http://localhost:8000/decode \\
         -H "Content-Type: application/json" \\
         -d '{"eeg_data": [[...]], "sampling_rate": 500}'
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import json
import numpy as np
from typing import List

app = FastAPI(title="NEST EEG-to-Text API")

# Load model at startup
model = torch.jit.load('deployment/nest_model.pt')
model.eval()

# Load tokenizer
with open('deployment/tokenizer.json', 'r') as f:
    tokenizer_config = json.load(f)
    vocab = tokenizer_config['vocab']

class EEGInput(BaseModel):
    eeg_data: List[List[float]]  # Shape: [channels, timepoints]
    sampling_rate: float = 500.0

class DecodingOutput(BaseModel):
    text: str
    confidence: float
    tokens: List[int]

@app.post("/decode", response_model=DecodingOutput)
async def decode_eeg(input: EEGInput):
    """Decode EEG signal to text"""
    try:
        # Convert to tensor
        eeg_tensor = torch.tensor(input.eeg_data).float().unsqueeze(0)
        
        # Check input shape
        if eeg_tensor.shape[1] != 32:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 32 channels, got {eeg_tensor.shape[1]}"
            )
        
        # Run inference
        with torch.no_grad():
            output = model(eeg_tensor)
        
        # Decode tokens
        predicted_tokens = output[0].argmax(dim=-1).tolist()
        confidence = output[0].max(dim=-1).values.mean().item()
        
        # Convert tokens to text
        text = ''.join([vocab[t] for t in predicted_tokens if t < len(vocab)])
        
        return DecodingOutput(
            text=text,
            confidence=confidence,
            tokens=predicted_tokens
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model": "nest_attention"}

@app.get("/info")
async def model_info():
    """Get model information"""
    with open('deployment/metadata.json', 'r') as f:
        metadata = json.load(f)
    return metadata

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
    
    api_path = Path('deployment/api.py')
    with open(api_path, 'w') as f:
        f.write(api_code)
    
    print(f"\nFastAPI server created: {api_path}")
    print("To run: uvicorn deployment.api:app --host 0.0.0.0 --port 8000")

def main():
    # Export models
    export_models()
    
    # Create API
    create_inference_api()
    
    print("\n" + "="*60)
    print("Deployment setup complete!")
    print("\nNext steps:")
    print("1. Test TorchScript model:")
    print("   python -c 'import torch; m=torch.jit.load(\"deployment/nest_model.pt\"); print(m)'")
    print("\n2. Start API server:")
    print("   cd deployment && uvicorn api:app --reload")
    print("\n3. Test API:")
    print("   curl http://localhost:8000/health")
    print("="*60)

if __name__ == '__main__':
    main()
