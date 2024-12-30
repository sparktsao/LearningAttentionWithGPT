# Understanding Self-Attention and Encoder-Decoder Attention in Transformers

Attention mechanisms are at the core of transformer architectures, enabling models to capture relationships within and across sequences. Two critical attention types are **Self-Attention** and **Encoder-Decoder Attention**, each serving a distinct purpose in the data flow of transformers. This blog explores their differences and clarifies the unique role of uni-directionality in Encoder-Decoder Attention, with an in-depth explanation of attention during sequence generation.

---

## What Is Self-Attention?

### Definition:

Self-Attention allows each token in a sequence to attend to every other token within the same sequence. This mechanism captures contextual relationships and dependencies, making it essential for tasks like language modeling and sequence generation.

### Key Characteristics:

- **Bi-directional Context**:

  - Tokens can access information from both past and future tokens within the sequence.
  - Example: In the sequence `ABCDE`, the token `C` can attend to `A`, `B`, `D`, and `E`.

- **Parallel Processing**:

  - Self-Attention is computed in parallel for all tokens, leveraging the efficiency of matrix operations.

### Applications:

- Found in both the encoder and decoder components of a transformer.
- Used in tasks requiring full-sequence understanding, such as classification and translation.

---

## What Is Encoder-Decoder Attention?

### Definition:

Encoder-Decoder Attention enables the decoder to attend to the encoder’s output representations, bridging the gap between the input and output sequences.

### Key Characteristics:

- **Uni-directional Context**:

  - The decoder attends only to the encoder’s outputs, which represent the input sequence.
  - Example: When translating `"I am fine"` to `"Je vais bien"`, the decoder token `"Je"` attends to the encoder's representation of the entire input sequence.

- **Cross-Sequence Attention**:

  - Unlike self-attention, which operates within a single sequence, encoder-decoder attention links two distinct sequences: the input (encoder output) and the partially generated output (decoder input).

- **Static Encoder Context**:

  - The encoder outputs remain fixed, providing a stable reference for the decoder throughout the generation process.

---

## Key Differences Between Self-Attention and Encoder-Decoder Attention

| Feature                    | Self-Attention                               | Encoder-Decoder Attention                |
| -------------------------- | -------------------------------------------- | ---------------------------------------- |
| **Sequence Scope**         | Single sequence (input or output)            | Cross-sequence (input to output)         |
| **Context Directionality** | Bi-directional in encoder; causal in decoder | Uni-directional (encoder to decoder)     |
| **Data Flow**              | Token-to-token within the same sequence      | Token-to-token across sequences          |
| **Applications**           | Context modeling within sequences            | Context transfer from encoder to decoder |

---

## Is Encoder-Decoder Attention Uni-Directional?

Yes, **Encoder-Decoder Attention is uni-directional**, but this uni-directionality refers specifically to the data flow between the encoder and decoder. Let’s break it down:

### 1. **Directionality in Encoder-Decoder Attention**

- The decoder tokens attend only to the encoder’s fixed outputs, which represent the input sequence.
- This ensures that the decoder can focus entirely on the context provided by the input sequence, without introducing inter-dependencies between the decoder tokens themselves.

### 2. **Comparison with Self-Attention**

- **Self-Attention in Encoder**: Bi-directional, allowing each token to attend to all others.
- **Self-Attention in Decoder**: Causal (uni-directional) to prevent future tokens from influencing the current prediction.
- **Encoder-Decoder Attention**: Uni-directional from encoder outputs to decoder tokens.

### 3. **Why Uni-Directionality Matters**

- **Stability**: Provides a static reference for the decoder, preventing dynamic changes that could destabilize training.
- **Clarity in Context Transfer**: Simplifies the interaction between input and output sequences, ensuring a clear directional flow of information.

---

## Generating a Sequence: How Decoder Attention Operates

When generating a sequence like `DEF`, let’s examine the scenario where the decoder has already generated `DE` and is now predicting the next token, `F`:

### Attention Details:

1. **Query (Q):**
   - Derived from the most recent token, `E`.

2. **Key (K) and Value (V):**
   - Derived from all previous tokens in the sequence up to and including `E`. In this case, `(D, E)`.

### Causal Attention and Uni-Directionality:

- The attention mechanism is **causal**, meaning the decoder token `E` can only attend to itself and the preceding tokens `(D)`. It cannot attend to the future token `F`, ensuring predictions are made based on past context alone.
- This restriction enforces a **uni-directional** attention flow within the decoder, critical for autoregressive sequence generation.

---

## Visualizing the Attention Mechanisms

### Self-Attention:

```
Sequence: A B C D E
Token C attends to: A, B, D, E
```

### Encoder-Decoder Attention:

```
Input Sequence:  X Y Z
Output Sequence: A B C
Token B attends to: X, Y, Z (via encoder outputs)
```

### Decoder Causal Attention:

```
Output Sequence: A B C
Token B attends to: A
Token B does not attend to: C
```

### Generating the Next Token:

For example:
```
Sequence: DE
Query (Q): E
Key (K): D, E
Value (V): D, E
```
The model computes attention weights to predict the next token `F` based on this limited, uni-directional context.

---

## Conclusion

Self-Attention and Encoder-Decoder Attention serve complementary roles in transformers. Self-Attention captures intra-sequence relationships, providing contextual understanding within the encoder or decoder. In contrast, Encoder-Decoder Attention focuses on inter-sequence relationships, transferring information uni-directionally from encoder to decoder. This uni-directionality ensures stable and effective context integration, making transformers robust and versatile for tasks like machine translation and text generation.

