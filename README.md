# UniGraph: Learning a Unified Cross-Domain Foundation Model for Text-Attributed Graphs

This repository contains the implementation of UniGraph, a unified cross-domain foundation model for text-attributed graphs. UniGraph combines the power of language models and graph neural networks to learn rich representations from text-attributed graphs across different domains.

## Architecture

UniGraph consists of three main components:

1. **Cascaded LM-GNN Backbone**
   - Language Model (LM) encoder for processing text attributes
   - Graph Neural Network (GNN) encoder for capturing graph structure
   - Fusion layer to combine LM and GNN outputs

2. **Graph Siamese Masked Autoencoders**
   - Masked Language Modeling (MLM) for text understanding
   - Latent space regularization for better representation learning
   - Target networks with exponential moving average updates

3. **Instruction Tuning**
   - Zero-shot transfer capabilities
   - Task-specific instruction processing
   - LoRA-based parameter-efficient fine-tuning

## Requirements

- Python 3.8+
- PyTorch 1.9+
- DGL 0.7+
- Transformers 4.15+
- NumPy
- SciPy
- scikit-learn

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Format

The model expects text-attributed graphs in the following format:

1. **Graph Structure**
   - `edge_list.txt`: Edge list in format `src dst`
   - `node_texts.txt`: Text attributes for nodes
   - `edge_texts.txt` (optional): Text attributes for edges

2. **Labels** (for supervised tasks)
   - `{split}_labels.txt`: Labels for nodes/edges/graphs
   - `instructions.txt`: Task-specific instructions

## Usage

### Pre-training

```bash
python main.py \
    --data_dir /path/to/data \
    --lm_type microsoft/deberta-base \
    --hidden_size 768 \
    --num_layers 3 \
    --num_heads 8 \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --lam 0.1 \
    --mask_rate 0.15
```

### Instruction Tuning

```bash
python main.py \
    --mode instruction_tuning \
    --data_dir /path/to/data \
    --checkpoint /path/to/pretrained/model \
    --llm_type microsoft/deberta-base \
    --lora_r 8 \
    --lora_alpha 16 \
    --batch_size 16 \
    --epochs 10 \
    --lr 1e-4
```

### Inference

```python
from model import UniGraph
import torch

# Load model
model = UniGraph(args)
model.load_state_dict(torch.load('checkpoint.pt'))

# Get embeddings
embeddings = model.get_embeddings(batch)
```

## Key Features

1. **Unified Architecture**
   - Combines language models and graph neural networks
   - Handles both text and graph structure
   - Supports multiple downstream tasks

2. **Efficient Training**
   - PPR-based subgraph sampling
   - Latent space regularization
   - Parameter-efficient fine-tuning with LoRA

3. **Zero-shot Transfer**
   - Instruction-based task adaptation
   - Cross-domain generalization
   - Flexible task formulation

## Citation

If you use this code in your research, please cite:

```bibtex
@article{he2025unigraph,
  title={UniGraph: Learning a Unified Cross-Domain Foundation Model for Text-Attributed Graphs},
  author={He, Yufei and Sui, Yuan and He, Xiaoxin and Hooi, Bryan},
  journal={ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.


