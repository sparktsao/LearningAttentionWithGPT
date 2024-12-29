import torch
import torch.nn as nn
import math

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        print(f"[PositionalEncoding] Input x.shape: {x.shape}, Adding positional encodings for sequence length and embedding size.")
        result = x + self.pe[:, :x.size(1)]
        print(f"[PositionalEncoding] Output shape: {result.shape}")
        return result

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        assert embed_size % heads == 0, "Embedding size must be divisible by heads"
        self.heads = heads
        self.head_dim = embed_size // heads
        
        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, queries, mask=None):
        N, query_len, embed_size = queries.size()
        print(f"[MultiHeadAttention:1] Input shapes (V, K, Q) - values: {values.shape}, keys: {keys.shape}, queries: {queries.shape}")

        values = self.values(values).view(N, -1, self.heads, self.head_dim)
        keys = self.keys(keys).view(N, -1, self.heads, self.head_dim)
        queries = self.queries(queries).view(N, query_len, self.heads, self.head_dim)
        print(f"[MultiHeadAttention:2] After linear projections V/K/Q multihead- values: {values.shape}, keys: {keys.shape}, queries: {queries.shape}")

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        print(f"[MultiHeadAttention:3] Energy = Q * K (raw attention scores) shape: {energy.shape}")

        if mask is not None:
            print(f"[MultiHeadAttention:4] Applying mask with shape: {mask.shape}")
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / math.sqrt(self.head_dim), dim=-1)
        print(f"[MultiHeadAttention:5] Attention SoftMax weights shape: {attention.shape}")

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).contiguous()
        out = out.view(N, query_len, embed_size)
        print(f"[MultiHeadAttention:6] Attention * V = Output shape after weighted sum: {out.shape}")
        return self.fc_out(out)

# Feedforward Layer
class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_dim):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_size)
        )

    def forward(self, x):
        print(f"[FeedForward] Input shape: {x.shape}, Performing feedforward transformations.")
        result = self.net(x)
        print(f"[FeedForward] Output shape: {result.shape}")
        return result

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, ff_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        print(f"[TransformerBlock] Starting block processing. Input dimensions - value: {value.shape}, key: {key.shape}, query: {query.shape}")
        attention = self.attention(value, key, query, mask)
        x = self.norm1(query + self.dropout(attention))
        print(f"[TransformerBlock] After attention and residual connection: {x.shape}")
        forward = self.feed_forward(x)
        out = self.norm2(x + self.dropout(forward))
        print(f"[TransformerBlock] After feedforward and residual connection: {out.shape}")
        return out

# Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, ff_hidden_dim, dropout, vocab_size, max_len):
        super(TransformerEncoder, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_encoding = PositionalEncoding(embed_size, max_len)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, ff_hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        print(f"[TransformerEncoder] Input x.shape: {x.shape}")
        x = self.word_embedding(x)
        print(f"[TransformerEncoder] After word embedding x.shape: {x.shape}")
        x = self.position_encoding(x)
        for i, layer in enumerate(self.layers):
            print(f"[TransformerEncoder] Processing layer {i+1}")
            x = layer(x, x, x, mask)
        print(f"[TransformerEncoder] Output x.shape: {x.shape}")
        return x

# Decoder Block
class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, ff_hidden_dim, dropout):
        super(TransformerDecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.norm3 = nn.LayerNorm(embed_size)
        self.feed_forward = FeedForward(embed_size, ff_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.encoder_attention = MultiHeadAttention(embed_size, heads)

    def forward(self, x, encoder_out, src_mask, trg_mask):
        print(f"[TransformerDecoderBlock] x.shape: {x.shape}, encoder_out.shape: {encoder_out.shape}")
        attention = self.attention(x, x, x, trg_mask)
        x = self.norm1(x + self.dropout(attention))
        encoder_attention = self.encoder_attention(x, encoder_out, encoder_out, src_mask)
        x = self.norm2(x + self.dropout(encoder_attention))
        forward = self.feed_forward(x)
        out = self.norm3(x + self.dropout(forward))
        print(f"[TransformerDecoderBlock] out.shape: {out.shape}")
        return out

# Decoder
class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, num_layers, heads, ff_hidden_dim, dropout, vocab_size, max_len):
        super(TransformerDecoder, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_encoding = PositionalEncoding(embed_size, max_len)
        self.layers = nn.ModuleList(
            [TransformerDecoderBlock(embed_size, heads, ff_hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, src_mask, trg_mask):
        print(f"[TransformerDecoder] Input x.shape: {x.shape}, encoder_out.shape: {encoder_out.shape}")
        x = self.word_embedding(x)
        x = self.position_encoding(x)
        for i, layer in enumerate(self.layers):
            print()
            print(f"[TransformerDecoder] Processing layer {i+1}")
            x = layer(x, encoder_out, src_mask, trg_mask)
            
        x = self.fc_out(x)
        print(f"[TransformerDecoder] Output x.shape: {x.shape}")
        return x

# Transformer
class Transformer(nn.Module):
    def __init__(self, embed_size, num_layers, heads, ff_hidden_dim, dropout, src_vocab_size, trg_vocab_size, max_len):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(embed_size, num_layers, heads, ff_hidden_dim, dropout, src_vocab_size, max_len)
        self.decoder = TransformerDecoder(embed_size, num_layers, heads, ff_hidden_dim, dropout, trg_vocab_size, max_len)

    def make_src_mask(self, src):
        mask = (src != 0).unsqueeze(1).unsqueeze(2)
        print(f"[make_src_mask] mask.shape: {mask.shape}")
        return mask

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).expand(N, 1, trg_len, trg_len)
        print(f"[make_trg_mask] mask.shape: {mask.shape}")
        return mask

    def forward(self, src, trg):
        print(f"[Transformer] src.shape: {src.shape}, trg.shape: {trg.shape}")
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        encoder_out = self.encoder(src, src_mask)
        out = self.decoder(trg, encoder_out, src_mask, trg_mask)
        return out

# Example usage
embed_size = 40
num_layers = 2
heads = 4
ff_hidden_dim = 64
dropout = 0.1
src_vocab_size = 128
trg_vocab_size = 128
max_len = 100
batch_size = 16
seq_n = 7

model = Transformer(embed_size, num_layers, heads, ff_hidden_dim, dropout, src_vocab_size, trg_vocab_size, max_len)
src = torch.randint(0, src_vocab_size, (batch_size, seq_n))  
trg = torch.randint(0, trg_vocab_size, (batch_size, seq_n))  

output = model(src, trg)
print(f"[Main] Final output shape: {output.shape}")
