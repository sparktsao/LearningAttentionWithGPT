# Attention Is All I Need

The "Attention Is All I Need" repository offers a deep dive into the transformer architecture, providing detailed logs to enhance understanding of its processes and logic. This implementation is designed to educate and demystify the intricacies of transformers, offering insights into data flow, attention mechanisms, and dimensional transformations.

---

## Learning Highlights

### 1. **Key-Query-Value (KQV) Dynamics**
- When predicting the next token `C` in a sequence like `AB`, the Key (K) corresponds to `B`, as `C` remains unknown during prediction. This distinction is crucial to grasping the transformer attention mechanism and correcting common misconceptions, such as assuming K represents `C`.

### 2. **Layer Normalization**
- The transformer employs **Layer Normalization**, referencing the 2015 paper ["Layer Normalization"](https://arxiv.org/abs/1607.06450).
- Layer Normalization is chosen over Batch Normalization due to its superior compatibility with sequence data processing.

### 3. **Decoder Block and Encoder-Decoder Attention**
- Encoder-Decoder Attention occurs as the second step in the decoder block, enabling integration of encoder outputs with decoder self-attention data.
- This attention mechanism is repeated across all `Nx` decoder layers, contrary to the misconception that encoder outputs are used only once.

### 4. **Detailed Logging for Transformer Data Flow**
- Comprehensive logs capture every critical operation, detailing input/output shapes for:
  - Embeddings
  - Positional Encoding
  - Attention Layers
  - Feedforward Networks

- These logs are invaluable for debugging and enhancing comprehension, ensuring clarity in dimensional transformations.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sparktsao/attention_is_all_i_need.git
   cd attention_is_all_i_need
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio --pre --index-url https://download.pytorch.org/whl/nightly/cpu
   ```

---

## Usage

Run the transformer and observe detailed logs:

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
- Adds positional context to input embeddings.
- Logs input/output shapes for verification.

### 2. **Multi-Head Attention**
- Executes scaled dot-product attention across multiple heads.
- Logs each step, including linear projections and attention weight calculations.

### 3. **Feedforward Network**
- Employs fully connected layers with ReLU activation.
- Logs input/output shapes for clarity.

### 4. **Layer Normalization**
- Normalizes inputs at each sub-layer.
- Improves stability and convergence.

### 5. **Comprehensive Logging**
- Tracks each step of the transformer pipeline, detailing intermediate shapes and processes.

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

Sample logs from a forward pass:

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
- Expand support for various input/output modalities (e.g., images, audio).
- Optimize implementation for GPU/TPU environments.
- Add visualizations for attention weights.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments
Special thanks to the PyTorch community and the authors of the original transformer paper for their foundational contributions.

