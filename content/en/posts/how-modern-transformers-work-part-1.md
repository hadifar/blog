---
title: "How Modern Transformers Work — Part 1"
date: 2026-08-25
categories: [Building Blocks]
draft: false
slug: how-modern-transformers-work-part-1
---

At a high level, modern transformers take an input (text, image, voice) and generate an
output (text, image, voice). Simply put:

```python
tokenizer = Tokenizer()
transformer = Transformer()

q = "what's your name?"

input_idx = tokenizer.encode(q)   # [13347, 885, 634, 1308, 30]
output_idx = transformer(input_idx)  # [15390, 59595, 11, 8113, 656, 166036, 13, 220]
o = tokenizer.decode(output_idx)  # I'm transformer, built by Amir.
```

## Tokenizer

We skip explaining the tokenizer piece here, because the goal of this post is the
transformer itself. The tokenizer converts raw text into a sequence of ids (using
`encode()`):

`"What's your name?" -> [13347, 885, 634, 1308, 30]`

The pretrained transformer model generate a `output_idx` and tokenizer again converts a sequence of ids back into raw text (using `decode()`):

`[13347, 885, 634, 1308, 30] -> "What's your name?"`

## A basic transformer

Let's start with a basic (GPT-2-style) transformer class. I skipped dropout, tensor
reshaping/transposing, and some other details for brevity.

Generally, a transformer decomposes into multiple pieces such as embedding layers, attention layers,
feed-forward layers. Modern transformer architectures usually differ from each other by:

a) how they implement embeddings (e.g. to support larger context — 1M tokens),
b) how they implement the attention mechanism (Multi-Head Attention, Grouped-Query
Attention, Linear Attention, etc.),
c) how residual connections, layer norms, and feed-forward layers are arranged across
layers, and
d) other innovations such as KV-caching, the optimizer, or the activation
function.

Let's look at the original transformer with Multi-Head Attention (MHA):

```python
class Transformer(nn.Module):
    def __init__(self, vocab_size=32_000, pos_size=1024, hidden_dim=512, n_layers=4, n_attention_heads=12):
        
	self.tok_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(pos_size, hidden_dim)

        self.attention_blocks = [MultiHeadAttentionBlock(n_attention_heads, hidden_dim) for i in range(n_layers)]

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def __call__(self, input_idx):
        x = self.tok_embedding(input_idx) + self.pos_embedding(input_idx)

        for attn_blk in self.attention_blocks:
            x = attn_blk(x)

        x = self.layer_norm(x)
        x = self.output_layer(x)
        return x
```

## Multi-Head Attention

Probably the most important piece is the attention block, which we can write as:

```python
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, hidden_dim):

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.linear = nn.Linear(hidden_dim, hidden_dim)

        self.mlp = nn.Linear(hidden_dim, hidden_dim)

    def attention(self, input_idx):
        q = self.q_proj(input_idx)
        k = self.k_proj(input_idx)
        v = self.v_proj(input_idx)

        # softmax((q @ k^T) / scale) @ v
        out = F.scaled_dot_product_attention(q, k, v)

        return self.linear(out)

    def __call__(self, input_idx):
        x = input_idx
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x
```

![Multi-Head Attention block diagram](/images/how-modern-transformers-work-part-1/multi-head-attention.png)
_Multi-Head Attention_

## Reducing the memory bottleneck

The above architecture has a severe bottleneck: *memory* bandwidth overhead at decoding time.
This is where the interesting design space opens up — the goal becomes reducing this memory
bottleneck, and several solutions have been proposed to do it.

### Grouped-Query Attention

One solution is [Grouped-Query Attention](https://arxiv.org/pdf/2305.13245):

![Grouped-Query Attention diagram](/images/how-modern-transformers-work-part-1/group-query-attention.png)
_Grouped-Query Attention_

Grouped-query attention divides queries (q) into G groups, each of which shares a single
key (k) and value (v). Basically:

```python
class GroupQueryAttentionBlock(nn.Module):
    def __init__(self, n_attention_heads = 16, hidden_dim = 1024, n_kv_group = 8):
        self.head_dim = hidden_dim // n_attention_heads #64
	self.n_head = n_attention_heads                 #16
	self.n_kv_group = n_kv_group                    #8

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)                 # 1024x1024
        self.k_proj = nn.Linear(hidden_dim, n_kv_group * self.head_dim) # 1024x512
        self.v_proj = nn.Linear(hidden_dim, n_kv_group * self.head_dim) # 1024x512

        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def attention(self, x):
        q = self.q_proj(x) # 1024
        k = self.k_proj(x) # 512 
        v = self.v_proj(x) # 512

        groups = self.n_head // self.n_kv_group
        k = k.repeat_interleave(groups, dim=1) # 512 -> 1024
        v = v.repeat_interleave(groups, dim=1) # 512 -> 1024

        out = F.scaled_dot_product_attention(q, k, v)

        return self.linear(out)
```

As you can see, instead of three linear layers (`self.q_proj, self.k_proj, self.v_proj`) of size `hidden_dim * hidden_dim`, we have
two smaller linear layers (`self.k_proj`, `self.v_proj`) whose size is determined by `n_kv_groups`.

This reduces the size of the k/v projection (1024×512 instead of 1024×1024) and, more
importantly in practice, the size of the kv cache during decoding (8 kv heads stored instead
of 16).

### What is the KV cache?

As the name suggests, it's about caching k and v values and reusing them later in the
process. You might ask why. As you know, we generate one token at a time during decoding:

![Transformer decoding, one token at a time](/images/how-modern-transformers-work-part-1/transformer-decoding.jpg)
_Transformer decoding, one token at a time_

During decoding, we have to wait and see what token we generated previously in order to
generate the next one. What happens is a loop:

```python
input_idx = [13347, 885, 634, 1308, 30]  # "What's your name?"
while (new_id := -1) != 0:
    logits = transformer(input_idx)
    new_id = sample(logits[-1])
    input_idx.append(new_id)

tokenizer.decode(input_idx[6:])  # I'm transformer, built by Amir.
```

We generate one token after another and feed the concatenation back to the model until we
reach a specific token — an end-of-sequence marker, here assumed to be `0`.

As you can see in this while-loop, for every new token we want to generate, we pass the
entire sequence (`input_idx`) and recompute all the attention and linear layers. But we can
cache part of that computation to avoid repeating it. That's where KV-caching comes in.

KV-caching has two phases:

**Prefill**: we feed the initial `input_idx` (our prompt, "What's your name?") to build the
initial KV cache.

**Decoding**: we feed one token at a time and reuse the already-computed KV values in the
cache.

Our while-loop then becomes:

```python
# prefill phase
input_idx = [13347, 885, 634, 1308, 30]  # "What's your name?"
logits, kv_caches = transformer(input_idx, None)

while (new_id := -1) != 0:
    new_id = sample(logits[-1])
    # decoding phase
    logits = transformer([new_id], kv_caches)
    input_idx.append(new_id)
```

We pass a new argument to `transformer.__call__` (`kv_caches`). Looking back at the
transformer class:

```python
class Transformer(nn.Module):

    def __call__(self, input_idx, kv_caches):

        ...

        if kv_caches is None:
            kv_caches = [None] * len(self.attention_blocks)

        new_kv_caches = []
        for attn_blk, cache in zip(self.attention_blocks, kv_caches):
            x, new_cache = attn_blk(x, cache)
            new_kv_caches.append(new_cache)

        x = self.layer_norm(x)
        x = self.output_layer(x)
        return x, new_kv_caches
```

The important part is the attention mechanism though:

```python
def attention(self, x, kv_cache=None):
    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)

    if kv_cache is not None:
        past_k, past_v = kv_cache
        if past_k is not None:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)
        new_kv_cache = (k, v)
    else:
        new_kv_cache = None

    out = F.scaled_dot_product_attention(q, k, v)

    return self.linear(out), new_kv_cache
```

Note that above, instead of passing the entire sequence (`[13347, 885, 634, 1308, 30]`) for
a new token, we only compute projections on the last single token. Then, to calculate
attention, we reuse the cache. The attention function, instead of computing `q_proj`,
`k_proj`, `v_proj` on the full sequence of length 5, computes them on a single token.
Skipping that computation means fewer matrix multiplications and a faster response time.

*Ask yourself*: why don't we cache the query (q)?

However, this useful caching mechanism comes at a cost: *memory* bandwidth, as briefly
mentioned above. As sequences grow, storing the cache becomes increasingly expensive. For
example, a large input prompt of 10,000 tokens requires storing ~491.5 MB in memory
(2 [for K and V] × 12 [number of layers] × 10,000 tokens × 1024 [hidden size] × 2 bytes
[fp16] = 491,520,000 bytes ≈ 491.5 MB) — and that's just the KV cache, before accounting for
the rest of the transformer's computations. In model deployment, this KV cache is a
major bottleneck that limits the maximum *batch size* and *sequence length*.

At batch=128 (a realistic serving batch), we're already at ~63 GB — more than an A100/H100's
80 GB, leaving almost nothing for the model weights themselves. Note that our 12-layer,
1024-hidden config is small — a real 70B model has ~80 layers and ~8K hidden dim.

### Multi-Head Latent Attention

As we saw, Grouped-Query Attention reduces the computation requirements. But
[DeepSeek-V2](https://arxiv.org/pdf/2405.04434) claimed that the downstream task performance
of Grouped-Query Attention is inferior to original Multi-Head Attention, and introduced
Multi-Head Latent Attention instead.

Instead of sharing KV heads across groups, it compresses (down projection) k and v into a smaller vector for
each token, caches that (a much smaller vector than the full k/v), and reconstructs (up projection) the
full-size k/v from the compressed vector on the fly. This keeps a much smaller cache.
Specifically:

```python
class MultiHeadLatentAttentionBlock(nn.Module):
    def __init__(self, n_attention_heads, hidden_dim, kv_latent_dim):

        self.head_dim = hidden_dim // n_attention_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)

        self.kv_down_proj = nn.Linear(hidden_dim, kv_latent_dim)

        self.k_up_proj = nn.Linear(kv_latent_dim, hidden_dim)
        self.v_up_proj = nn.Linear(kv_latent_dim, hidden_dim)

        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def attention(self, x, kv_cache=None):
        q = self.q_proj(x)

        c_kv = self.kv_down_proj(x)

        if kv_cache is not None:
            past_c_kv = kv_cache
            if past_c_kv is not None:
                c_kv = torch.cat([past_c_kv, c_kv], dim=1)
            new_kv_cache = c_kv
        else:
            new_kv_cache = None

        k = self.k_up_proj(c_kv)
        v = self.v_up_proj(c_kv)

        out = F.scaled_dot_product_attention(q, k, v)

        return self.linear(out), new_kv_cache
```

As you can see above, they compressed `x` down to a smaller size (via `self.kv_down_proj`) and only
cache that. When needed, they reconstruct it via `self.k_up_proj` and `self.v_up_proj` — we
reduce the KV cache (memory) at the cost of higher compute. Why is that a good trade? Because
at decoding (inference time) we're memory-bound rather than compute-bound. It means, the GPU spends a
lot of time moving data, in & out of GPU's memory VRAM, rather than doing actual computation (matrix
multiplication). We have lots of memory in VRAM but we have only a few kilobytes of register space.

VRAM (GB) -> L2 Cache -> L1 Cache -> Registers (KB) -> Tensor-core (actual computation happens here!)

Now let's assume you want to move a 1024x1024 matrix into the tensor-core — that means moving
1024*1024 ~= 1 million elements, roughly 1M x 2 bytes = 2MB, which is already far larger than
the register space!

In the next post in this series, I'll cover some other tricks used in modern transformers to
work around this memory bottleneck.

## References

- [Attention is all you need](https://arxiv.org/pdf/1706.03762)
- [Qwen2 technical report](https://arxiv.org/pdf/2407.10671)
- [Grouped-query attention](https://arxiv.org/pdf/2305.13245)
- [DeepSeek-V2](https://arxiv.org/pdf/2405.04434)
</content>
