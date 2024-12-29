# Attention Is All I Need

This repository implements a detailed transformer model based on the "Attention Is All You Need" architecture. It serves as an educational guide, providing insights into the intricacies of transformer data flow, attention mechanisms, and dimensional transformations, with verbose logging to enhance understanding.

---

## Learning Highlights

### 1. **Key-Query-Value (KQV) Dynamics**
- When predicting the next token `C` in a sequence like `AB`, the Key (K) corresponds to `B` because `C` is still unknown at prediction time. This distinction is vital for understanding the transformer attention mechanism.

### 2. **Layer Normalization**
- The transformer uses **Layer Normalization**, referencing the 2015 paper ["Layer Normalization"](https://arxiv.org/abs/1607.06450).
- The decision to use Layer Normalization instead of Batch Normalization is based on better compatibility with sequence processing.

### 3. **Decoder Block and Encoder-Decoder Attention**
- Encoder-Decoder Attention is the second step in the decoder block, bridging information between the encoder's outputs and the decoder's self-attention data.
- This mechanism is repeated across all `Nx` decoder layers to enhance context integration.

### 4. **Detailed Logging for Transformer Data Flow**
- The implementation logs every critical operation, including input/output shapes for:
  - Embeddings
  - Positional Encoding
  - Attention Layers
  - Feedforward Networks
  
- These logs are invaluable for debugging and understanding the data flow, ensuring every dimension is clear and consistent.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/attention_is_all_i_need.git
   cd attention_is_all_i_need
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio --pre --index-url https://download.pytorch.org/whl/nightly/cpu
   ```

---

## Usage

To run the transformer and observe the detailed logs:

```bash
python main.py
```

Example Output:
```
[Transformer] src.shape: torch.Size([16, 7]), trg.shape: torch.Size([16, 7])
[make_src_mask] mask.shape: torch.Size([16, 1, 1, 7])
...
[Main] Final output shape: torch.Size([16, 7, 128])
```

---

## Key Features

### 1. **Positional Encoding**
- Adds positional information to input embeddings.
- Logs input/output shapes for easy verification.

### 2. **Multi-Head Attention**
- Performs scaled dot-product attention across multiple heads.
- Logs each step, including linear projections and attention weight computation.

### 3. **Feedforward Network**
- Fully connected layers with ReLU activation.
- Logs input/output shapes for clarity.

### 4. **Layer Normalization**
- Normalizes inputs at each sub-layer.
- Enhances stability and convergence.

### 5. **Comprehensive Logging**
- Tracks every step of the transformer pipeline, including intermediate shapes and operations.

---

## Model Architecture

- **Embedding Size**: 40
- **Number of Layers**: 2
- **Number of Heads**: 4
- **Feedforward Hidden Dimension**: 64
- **Dropout**: 0.1
- **Source/Target Vocabulary Size**: 128
- **Sequence Length**: 7

---

## Example Logs

Sample logs for a forward pass:

```bash
[TransformerEncoder] Input x.shape: torch.Size([16, 7])
[TransformerEncoder] After word embedding x.shape: torch.Size([16, 7, 40])
[PositionalEncoding] Input x.shape: torch.Size([16, 7, 40])
[TransformerBlock] After attention and residual connection: torch.Size([16, 7, 40])
...
[TransformerDecoder] Processing layer 1
[MultiHeadAttention] Attention SoftMax weights shape: torch.Size([16, 4, 7, 7])
[FeedForward] Output shape: torch.Size([16, 7, 40])
```

---

## References

1. Vaswani, A., et al. ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).
2. Ba, J. L., et al. ["Layer Normalization"](https://arxiv.org/abs/1607.06450).

---

## Future Work
- Extend the transformer to support different input/output modalities (e.g., images, audio).
- Explore optimizations for GPU/TPU environments.
- Implement visualizations for attention weights.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
Special thanks to the PyTorch community and the authors of the original transformer paper.


