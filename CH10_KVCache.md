# Can KV Cache Save 50% of Computation? An Analytics Report

Transformer models are the backbone of modern natural language processing (NLP) applications. However, their computational cost during inference, especially for long sequences, can be prohibitively high. One commonly used optimization is the **Key-Value (KV) Cache**, which reduces redundant computations. But can it really save up to 50% of computations? Let’s analyze.

---

## What is KV Cache?

In transformer-based architectures, the attention mechanism computes:

\[ \text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V \]

Here:
- **Query (Q)**: Represents the current token.
- **Key (K)** and **Value (V)**: Represent all previous tokens in the sequence.

During autoregressive inference, the attention mechanism computes the interaction between the current query and all previous tokens. Without caching, the keys and values for all past tokens are recomputed for each step, leading to redundant operations.

The KV cache stores the computed \( K \cdot W_k \) and \( V \cdot W_v \) for past tokens, avoiding their recomputation. This saves significant computational overhead during inference.

---

## Breakdown of Computation

Let’s analyze the computational cost of the \( KQV \) operations in terms of the sequence length (\( n \)) and embedding size (\( d \)).

### Without KV Cache

For a sequence of length \( n \):
- **Key (\( K \cdot W_k \))**: \( n \times d \) multiplied by \( d \times d \) costs \( n \cdot d^2 \).
- **Value (\( V \cdot W_v \))**: \( n \times d \) multiplied by \( d \times d \) costs \( n \cdot d^2 \).
- **Query (\( Q \cdot W_q \))**: For each query, \( 1 \times d \) multiplied by \( d \times d \) costs \( d^2 \).
- **Attention Computation (\( QK^T \))**: \( 1 \times d \) (query) multiplied by \( n \times d \) (keys) costs \( n \cdot d \).
- **Weighted Sum (\( \text{Softmax}(QK^T)V \))**: \( 1 \times n \) (attention scores) multiplied by \( n \times d \) (values) costs \( n \cdot d \).

Total cost per step:
\[
2 \cdot n \cdot d^2 + d^2 + 2 \cdot n \cdot d
\]

For all \( n \) steps:
\[
\text{Cost}_{\text{no cache}} = n \cdot (2 \cdot n \cdot d^2 + d^2 + 2 \cdot n \cdot d)
\]

---

### With KV Cache

When using KV cache:
- **Cached Components**: \( K \cdot W_k \) and \( V \cdot W_v \) for past tokens.
- **Non-Cached Components**:
  - \( Q \cdot W_q \): Each new query still requires computation. Cost: \( d^2 \).
  - Attention scores (\( QK^T \)): The dot product between the current query and cached keys. Cost: \( n \cdot d \).
  - Weighted sum (\( \text{Softmax}(QK^T)V \)): The attention-weighted sum over cached values. Cost: \( n \cdot d \).

For \( n \) steps:
- Initial computation of keys and values:
  \[
  2 \cdot n \cdot d^2
  \]
- Per-step computation for queries and attention:
  \[
  n \cdot (d^2 + 2 \cdot n \cdot d)
  \]

Total cost with KV cache:
\[
\text{Cost}_{\text{cache}} = 2 \cdot n \cdot d^2 + n \cdot (d^2 + 2 \cdot n \cdot d)
\]

---

### Computation Savings

The percentage of computations saved by using KV cache can be calculated as:
\[
\text{Savings} = 1 - \frac{\text{Cost}_{\text{cache}}}{\text{Cost}_{\text{no cache}}}
\]

Substituting the formulas:
\[
\text{Savings} = 1 - \frac{2 \cdot n \cdot d^2 + n \cdot (d^2 + 2 \cdot n \cdot d)}{n \cdot (2 \cdot n \cdot d^2 + d^2 + 2 \cdot n \cdot d)}
\]

Simplify:
\[
\text{Savings} = \frac{2 \cdot n \cdot d^2}{2 \cdot n \cdot d^2 + d^2 + 2 \cdot n \cdot d}
\]

For large sequence lengths (\( n \gg d \)):
\[
\text{Savings} \approx 1 - \frac{1}{2 \cdot n}
\]

As \( n \) becomes large, the savings approach **50%**.

---

## Key Takeaways

1. **KV Cache Significantly Reduces Computation**:
   - By caching \( K \cdot W_k \) and \( V \cdot W_v \), we avoid recomputation for past tokens, saving a substantial fraction of computations.

2. **Savings Scale with Sequence Length**:
   - For long sequences, the savings approach 50%. For shorter sequences, the savings are proportionally less but still meaningful.

3. **Not All Computation is Cached**:
   - Operations involving \( Q \cdot W_q \), attention score computation (\( QK^T \)), and weighted sum still need to be performed for each step.

---

## Conclusion

The KV cache is a powerful optimization that reduces redundant computations during transformer inference. While the exact savings depend on the sequence length and embedding size, for long sequences, KV caching can save approximately **50% of the computation** required for the \( KQV \) operations, making it an essential technique for efficient large-scale deployment of transformers.

