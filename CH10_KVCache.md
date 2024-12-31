# Can KV Cache Save 50% of Computation? An Analytics Report

Transformer models are the backbone of modern natural language processing (NLP) applications. However, their computational cost during inference, especially for long sequences, can be prohibitively high. One commonly used optimization is the **Key-Value (KV) Cache**, which reduces redundant computations. But can it really save up to 50% of computations? Let’s analyze.

---

## What is KV Cache?

In transformer-based architectures, the attention mechanism computes:

`Attention(Q, K, V) = Softmax((Q * K^T) / sqrt(d)) * V`

Here:
- **Query (Q)**: Represents the current token.
- **Key (K)** and **Value (V)**: Represent all previous tokens in the sequence.

During autoregressive inference, the attention mechanism computes the interaction between the current query and all previous tokens. Without caching, the keys and values for all past tokens are recomputed for each step, leading to redundant operations.

The KV cache stores the computed `K * Wk` and `V * Wv` for past tokens, avoiding their recomputation. This saves significant computational overhead during inference.

---

## Breakdown of Computation

Let’s analyze the computational cost of the `KQV` operations in terms of the sequence length (`n`) and embedding size (`d`).

### Without KV Cache

For a sequence of length `n`:
- **Key (`K * Wk`)**: `n x d` multiplied by `d x d` costs `n * d^2`.
- **Value (`V * Wv`)**: `n x d` multiplied by `d x d` costs `n * d^2`.
- **Query (`Q * Wq`)**: For each query, `1 x d` multiplied by `d x d` costs `d^2`.
- **Attention Computation (`Q * K^T`)**: `1 x d` (query) multiplied by `n x d` (keys) costs `n * d`.
- **Weighted Sum (`Softmax(Q * K^T) * V`)**: `1 x n` (attention scores) multiplied by `n x d` (values) costs `n * d`.

Total cost per step:
```
2 * n * d^2 + d^2 + 2 * n * d
```

For all `n` steps:
```
Cost_no_cache = n * (2 * n * d^2 + d^2 + 2 * n * d)
```

---

### With KV Cache

When using KV cache:
- **Cached Components**: `K * Wk` and `V * Wv` for past tokens.
- **Non-Cached Components**:
  - `Q * Wq`: Each new query still requires computation. Cost: `d^2`.
  - Attention scores (`Q * K^T`): The dot product between the current query and cached keys. Cost: `n * d`.
  - Weighted sum (`Softmax(Q * K^T) * V`): The attention-weighted sum over cached values. Cost: `n * d`.

For `n` steps:
- Initial computation of keys and values:
```
2 * n * d^2
```
- Per-step computation for queries and attention:
```
n * (d^2 + 2 * n * d)
```

Total cost with KV cache:
```
Cost_cache = 2 * n * d^2 + n * (d^2 + 2 * n * d)
```

---

### Computation Savings

The percentage of computations saved by using KV cache can be calculated as:
```
Savings = 1 - (Cost_cache / Cost_no_cache)
```

Substituting the formulas:
```
Savings = 1 - (2 * n * d^2 + n * (d^2 + 2 * n * d)) / (n * (2 * n * d^2 + d^2 + 2 * n * d))
```

Simplify:
```
Savings = (2 * n * d^2) / (2 * n * d^2 + d^2 + 2 * n * d)
```

For large sequence lengths (`n >> d`):
```
Savings ≈ 1 - (1 / (2 * n))
```

As `n` becomes large, the savings approach **50%**.

---

## Key Takeaways

1. **KV Cache Significantly Reduces Computation**:
   - By caching `K * Wk` and `V * Wv`, we avoid recomputation for past tokens, saving a substantial fraction of computations.

2. **Savings Scale with Sequence Length**:
   - For long sequences, the savings approach 50%. For shorter sequences, the savings are proportionally less but still meaningful.

3. **Not All Computation is Cached**:
   - Operations involving `Q * Wq`, attention score computation (`Q * K^T`), and weighted sum still need to be performed for each step.

---

## Conclusion

The KV cache is a powerful optimization that reduces redundant computations during transformer inference. While the exact savings depend on the sequence length and embedding size, for long sequences, KV caching can save approximately **50% of the computation** required for the `KQV` operations, making it an essential technique for efficient large-scale deployment of transformers.

