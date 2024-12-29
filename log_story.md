### Exploring Transformer Logs: A Deep Dive into Model Execution

When working with Transformers, understanding the intermediate outputs is essential for debugging and improving your model's performance. Here is a detailed walkthrough of the logged outputs from a Python script (`main.py`) implementing a Transformer.

---

#### 1. **Input Data Shape**

The script begins by initializing input tensors for the source (`src`) and target (`trg`):
```
[Transformer] src.shape: torch.Size([16, 7]), trg.shape: torch.Size([16, 7])
```
- **Batch Size**: 16
- **Sequence Length**: 7

#### 2. **Mask Creation**

To manage the attention mechanism, masks are created for the source and target sequences:
```
[make_src_mask] mask.shape: torch.Size([16, 1, 1, 7])
[make_trg_mask] mask.shape: torch.Size([16, 1, 7, 7])
```
- The source mask blocks padding tokens.
- The target mask ensures that predictions depend only on previous tokens.

#### 3. **Transformer Encoder**

The encoder processes the input in multiple stages:
```
[TransformerEncoder] Input x.shape: torch.Size([16, 7])
[TransformerEncoder] After word embedding x.shape: torch.Size([16, 7, 40])
```
- **Word Embedding**: Maps each token to a 40-dimensional vector.

##### Positional Encoding:
```
[PositionalEncoding] Input x.shape: torch.Size([16, 7, 40])
[PositionalEncoding] Output shape: torch.Size([16, 7, 40])
```
- Adds information about token positions to the embeddings.

##### Multi-Head Attention (MHA):

Each encoder layer performs self-attention and feedforward operations:
```
[MultiHeadAttention:1] Input shapes (V, K, Q): torch.Size([16, 7, 40])
[MultiHeadAttention:3] Energy = Q * K shape: torch.Size([16, 4, 7, 7])
[MultiHeadAttention:5] Attention SoftMax weights shape: torch.Size([16, 4, 7, 7])
```
- Input tensors are split into 4 attention heads, each with 10 dimensions.
- Softmax ensures the attention scores sum to 1.

##### Feedforward:
```
[FeedForward] Input shape: torch.Size([16, 7, 40])
[FeedForward] Output shape: torch.Size([16, 7, 40])
```
- Applies two linear transformations with an activation function in between.

#### 4. **Transformer Decoder**

The decoder combines information from the encoder and the target sequence:
```
[TransformerDecoder] Input x.shape: torch.Size([16, 7]), encoder_out.shape: torch.Size([16, 7, 40])
```

##### Masked Multi-Head Attention:
```
[MultiHeadAttention:4] Applying mask with shape: torch.Size([16, 1, 7, 7])
```
- The mask ensures autoregressive behavior, preventing the decoder from "seeing" future tokens.

##### Final Output:
```
[TransformerDecoder] Output x.shape: torch.Size([16, 7, 128])
[Main] Final output shape: torch.Size([16, 7, 128])
```
- The output tensor is transformed into a 128-dimensional representation for each token.

---

### Key Takeaways:

1. **Intermediate Shapes**: Observing tensor shapes at each stage helps verify model configuration.
2. **Debugging MHA**: Inspecting attention scores and mask applications ensures correct functionality.
3. **Residual Connections**: The combination of attention and feedforward blocks with residual connections ensures gradient flow.

These logs provide a clear view into the inner workings of the Transformer, making it easier to debug and optimize your model for specific tasks.

