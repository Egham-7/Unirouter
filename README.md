# Universal Model Routing Experiment

Implementation of **"Universal Model Routing for Efficient LLM Inference"** by Jitkrittum et al. (2025)

## Key Concept

Route queries to different LLMs based on cost-quality tradeoff using **cluster-based error profiles** that work with new unseen models without retraining.

## Quick Start

1. **Setup**: `01_unirouter_experiment.ipynb` - Dependencies and model configuration
2. **Characterization**: `02_model_characterization.ipynb` - Compute error profiles and routing
3. **Evaluation**: `03_evaluation.ipynb` - Deferral curves and adding new models

## Core Innovation

**Ψ(m) Vectors**: Each model represented as error rates per question cluster
- Enables routing to new models without expensive retraining
- Cost-quality tradeoff via λ parameter: `score = error_rate + λ × cost`

## Setup

```bash
pip install openai scikit-learn sentence-transformers datasets groq
```

Add your API keys:
```python
API_KEYS = {
    'openai': 'your-key-here',
    'groq': 'your-key-here'
}
```

## Results

- **Universal**: Works with any new LLM by computing its error profile
- **Efficient**: No retraining required for new models

---

**Paper**: https://arxiv.org/pdf/2502.08773  
**Authors**: Jitkrittum et al. (2025)