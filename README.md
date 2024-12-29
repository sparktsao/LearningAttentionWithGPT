# Learning Transformer by Coding with ChatGPT

Authored by **Spark Tsao**



Reading a paper is one thing; truly understanding the details is another. To bridge this gap, I implemented the transformer architecture and clarified key concepts. This journey has helped me deepen my understanding and address common misconceptions.

## Learning Highlights

### 1. **Understanding Attention Mechanisms**

- Explores the differences between the three types of attention: **Self-Attention**, **Encoder-Decoder Attention**, and **Decoder Causal Attention**.
- Clarifies how each type of attention operates and their roles in the transformer architecture.
- [Read more](CH1_Attention.md)

### 2. **Q is F, Not G**

- Highlights why the Query () is not the token we are predicting but the most recent token available.
- This distinction is crucial for understanding how attention mechanisms compute alignment scores.
- [Read more](CH2_QisFNotG.md)

### 3. **Does F Dominate G?**

- Investigates whether the last token () in a sequence dominates the generation of the next token ().
- Analyzes how attention balances the influence of preceding tokens.
- [Read more](CH3_DoesFDominateG.md)

### 4. **Layer Normalization Over Batch Normalization**

- Discusses why **Layer Normalization** is chosen in transformers.
- Explains its compatibility with sequence data, focus on embeddings, and ability to handle multi-head attention results seamlessly.
- [Read more](CH4_LayerNormDiscussion.md)

### 5. **Add vs. Concat**

- Compares the Add operation with Concatenation in transformers.
- Explains why Add is preferred for maintaining dimensional consistency and computational efficiency.
- [Read more](CH5_AddandConcat.md)


### 6. *** Log Story ***

- explain the logs.txt
- [Read more](CH6_log_story.md)

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

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

Special thanks to the PyTorch community and the authors of the original transformer paper for their foundational contributions.

