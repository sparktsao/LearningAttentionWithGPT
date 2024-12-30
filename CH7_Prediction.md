# Understanding GPT Token Prediction and Autoregressive Decoding

When working with Transformer-based models like GPT, understanding how these models generate predictions can help clarify their behavior and use cases. This blog post explores two key questions:

1. Why does the final output of the Transformer model have the shape `[BATCH_SIZE, SEQ_LEN, VOCAB_SIZE]` during training?
2. How does this relate to autoregressive decoding, such as generating the target sequence `CDF` from the source `ABC`?

---

## Why Is the Final Output Shape `[BATCH_SIZE, SEQ_LEN, VOCAB_SIZE]`?

In Transformer models like GPT, the final output during training has a shape of:

```python
[BATCH_SIZE, SEQ_LEN, VOCAB_SIZE]
```

### Key Reasons for Predicting the Entire Sequence

1. **Parallelization**:
   - During training, the model predicts all tokens in the target sequence simultaneously.
   - This is more efficient than predicting one token at a time because modern GPUs and TPUs can parallelize the computation over the entire sequence.

2. **Training Objective**:
   - The model is trained to minimize the loss over all positions in the target sequence.
   - For example, when the input sequence is `ABC` and the target sequence is `CDF`, the model learns to predict `C` for position 1, `D` for position 2, and `F` for position 3 all at once.

3. **Probabilistic Outputs**:
   - At each position in the sequence, the model outputs a probability distribution over the entire vocabulary (`VOCAB_SIZE`).
   - These logits are used to compute the loss (e.g., cross-entropy) during training.

4. **Efficiency**:
   - Parallel sequence prediction avoids the inefficiency of generating tokens sequentially during training, leveraging the Transformer’s architecture.

---

## How Does This Work During Inference?

While the model predicts all tokens in the sequence during training, inference works differently. During inference, GPT employs **autoregressive decoding**, where tokens are predicted one at a time, and each prediction depends on the tokens generated so far.

### Example: Predicting `CDF` from `ABC`

Let’s consider generating the target sequence `CDF` from the source `ABC` using a Transformer model. Here’s how autoregressive decoding works step-by-step:

1. **Step 1: Predict the First Token (`C`)**:
   - Input the source `ABC` and an initial target sequence (e.g., `[<sos>]`, the start-of-sequence token).
   - The model outputs probabilities for the next token. The most likely token (`C`) is selected and appended to the target sequence.

2. **Step 2: Predict the Second Token (`D`)**:
   - Input the source `ABC` and the target sequence `[<sos>, C]` so far.
   - The model predicts the next token (`D`) and appends it to the target sequence.

3. **Step 3: Predict the Third Token (`F`)**:
   - Input the source `ABC` and the target sequence `[<sos>, C, D]` so far.
   - The model predicts the next token (`F`) and appends it to the target sequence.

At the end of these steps, the model has generated the sequence `CDF`.

### Why Multiple Forward Passes?

During training, the model predicts the entire sequence at once. However, during inference, the model predicts one token at a time because:
- Each token prediction depends on previously generated tokens.
- The target sequence is not available during inference, so it must be generated incrementally.

---

## Example Code for Autoregressive Decoding

Here’s an example of how autoregressive decoding works in practice:

```python
src = torch.tensor([[1, 2, 3]])  # Example tokenized source "ABC"
trg_vocab_size = 10             # Assume a vocabulary size of 10
trg = [0]  # Start with <sos> token (e.g., index 0)

for _ in range(3):  # Generate 3 tokens (length of "CDF")
    trg_tensor = torch.tensor([trg])  # Current target sequence
    output = model(src, trg_tensor)  # Forward pass through the model
    next_token = output[:, -1, :].argmax(dim=-1).item()  # Get next token
    trg.append(next_token)  # Append predicted token to target sequence

print("Generated sequence:", trg)  # Should output something like [0, 3, 4, 5]
```

### Output Shapes During Training and Inference

- **Training**: `[BATCH_SIZE, SEQ_LEN, VOCAB_SIZE]`
  - Parallel prediction of all tokens in the target sequence.
- **Inference**: `[BATCH_SIZE, 1, VOCAB_SIZE]` (per step)
  - One token predicted at a time, growing the target sequence incrementally.

---

## Key Takeaways

1. During training, the Transformer predicts the entire sequence in parallel, resulting in the shape `[BATCH_SIZE, SEQ_LEN, VOCAB_SIZE]`.
2. During inference, the model generates tokens one at a time using autoregressive decoding.
3. Predicting the target sequence `CDF` from `ABC` requires 3 forward passes, one for each token in the sequence.

Understanding this difference is crucial for effectively using Transformer models in both training and inference scenarios!

