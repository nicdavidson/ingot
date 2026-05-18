# DeepSeek V4-Flash Architecture — Ingot Implementation Notes

## Model Overview
- **Architecture:** DeepseekV4ForCausalLM
- **Total params:** 284B, **Active:** 13B per token
- **Layers:** 43 + 1 MTP
- **Hidden size:** 4096
- **Heads:** 64 (attention), 1 KV head (MQA)
- **Head dim:** 512 (448 non-RoPE + 64 RoPE)
- **Vocab:** 129,280
- **Context:** 1M tokens

## Layer Types (from compress_ratios)
```
Layer 0,1:   Sliding window only (ratio=0) — no compressor, no indexer
Layer 2:     CSA (ratio=4) — has compressor + indexer
Layer 3:     HCA (ratio=128) — has compressor, NO indexer
Layers 4-41: Alternating CSA(4)/HCA(128)
Layer 42:    Sliding window only (ratio=0)
Layer 43:    MTP layer
```

## Attention Path
### Q Projection (low-rank)
```
x -> wq_a [hidden_size -> q_lora_rank=1024] -> RMSNorm -> wq_b [q_lora_rank -> num_heads * head_dim = 64*512 = 32768]
-> reshape [batch, seq, 64 heads, 512 dim]
-> inline Q norm (rsqrt(mean(q^2) + eps))
-> RoPE on last 64 dims
```

### KV Projection (MQA, single head)
```
x -> wkv [hidden_size -> head_dim=512]
-> kv_norm (RMSNorm)
-> RoPE on last 64 dims (first 448 = non-positional)
```

### O Projection (grouped low-rank)
```
attn_output [batch, seq, 64 heads, 512 dim]
-> de-RoPE last 64 dims
-> reshape to [batch, seq, o_groups=8, 8*512]
-> per-group wo_a [8*512=4096 -> o_lora_rank=1024]
-> wo_b [o_groups * o_lora_rank = 8*1024=8192 -> hidden_size=4096]
```

### Compressed Sparse Attention (CSA, ratio=4)
- Compressor: gated pooling of 4 consecutive KV tokens
  - wkv_compress [hidden -> head_dim], wgate [hidden -> head_dim]
  - Add absolute position embeddings (APE)
  - Softmax-weighted sum within group
  - Overlapping windows (4 tokens from previous group)
- Indexer: selects top-512 compressed positions
  - Own compressor (with Hadamard rotation)
  - Learned scoring: q @ compressed_kv -> topk(512)
- Attention = window(128) + selected_compressed(512)

### Heavily Compressed Attention (HCA, ratio=128)
- Same compressor but ratio=128
- No indexer — attend to ALL compressed positions (few enough)
- Attention = window(128) + all_compressed

### Sliding Window Only (layers 0,1,42)
- Standard attention on last 128 tokens
- No compression, no indexer

## Hyper-Connections (HC)
Hidden state carried as 4 copies: [batch, seq, hc_mult=4, hidden_size]

**Every sublayer (attention and FFN):**
```
Pre:  x_flat [4*hidden] -> RMSNorm -> linear(hc_fn) -> split to pre[4], post[4], comb[4,4]
      pre: sigmoid weights to reduce 4 copies -> 1
      comb: Sinkhorn-normalized mixing matrix (20 iterations)
      y = sum(pre * copies)  -> feed to sublayer

Post: output = post * sublayer_out + comb @ residual_copies
      -> [batch, seq, 4, hidden]
```

**Final head:**
```
pre = sigmoid(mixes * scale + base)
y = sum(pre * copies) -> RMSNorm -> lm_head
```

## MoE
- 256 routed experts, top-6 active, 1 shared expert
- Expert FFN: SwiGLU [hidden=4096 -> moe_intermediate=2048]
  - gate_proj, up_proj: SiLU activation, clamped at +/-10.0
  - down_proj: [moe_intermediate -> hidden]
- Gating: sqrtsoftplus (sqrt(softplus(x))) + bias correction + top-6
- Hash routing: first 3 layers use tid2eid lookup table (no gating compute)
- Shared expert: same SwiGLU, always active, no sigmoid gate (unlike Qwen)

## Weight Names (from safetensors)
### Per Layer — Attention
- model.layers.X.attn.wq_a — [hidden, q_lora_rank]
- model.layers.X.attn.wq_b — [q_lora_rank, num_heads * head_dim]
- model.layers.X.attn.wkv — [hidden, head_dim]
- model.layers.X.attn.wo_a — [o_groups, num_heads/o_groups * head_dim, o_lora_rank]
- model.layers.X.attn.wo_b — [o_groups * o_lora_rank, hidden]
- model.layers.X.attn.q_norm.weight — [q_lora_rank]
- model.layers.X.attn.kv_norm.weight — [head_dim]

### Per Layer — Compressor (CSA/HCA only)
- model.layers.X.attn.compressor.wkv — [hidden, head_dim]
- model.layers.X.attn.compressor.wgate — [hidden, head_dim]
- model.layers.X.attn.compressor.ape — [ratio, head_dim]
- model.layers.X.attn.compressor.kv_norm.weight — [head_dim]

### Per Layer — Indexer (CSA only)
- model.layers.X.attn.indexer.wq_b — [q_lora_rank, index_n_heads * index_head_dim]
- model.layers.X.attn.indexer.weights_proj — [..., ...]
- model.layers.X.attn.indexer.compressor.wkv
- model.layers.X.attn.indexer.compressor.wgate
- model.layers.X.attn.indexer.compressor.ape
- model.layers.X.attn.indexer.compressor.kv_norm.weight

### Per Layer — Hyper-Connections
- model.layers.X.attn_hc.hc_fn — [4*hidden, hc_output_dim]
- model.layers.X.attn_hc.hc_scale — [3]
- model.layers.X.attn_hc.hc_base — [hc_output_dim]
- model.layers.X.ffn_hc.hc_fn / hc_scale / hc_base

### Per Layer — MoE
- model.layers.X.ffn.gate.weight — [n_routed_experts, hidden]
- model.layers.X.ffn.switch_mlp.gate_proj — [256, moe_inter, hidden]
- model.layers.X.ffn.switch_mlp.up_proj — [256, moe_inter, hidden]
- model.layers.X.ffn.switch_mlp.down_proj — [256, hidden, moe_inter]
- model.layers.X.ffn.shared_experts.gate_proj — [moe_inter, hidden]
- model.layers.X.ffn.shared_experts.up_proj — [moe_inter, hidden]
- model.layers.X.ffn.shared_experts.down_proj — [hidden, moe_inter]

### Per Layer — Norms
- model.layers.X.input_layernorm.weight
- model.layers.X.post_attention_layernorm.weight

### Global
- model.embed_tokens.weight — [vocab, hidden]
- model.norm.weight — [hidden]
- lm_head.weight — [vocab, hidden]
- model.final_hc.hc_fn / hc_scale / hc_base

## Quantization (Q2-mixed, Thump604)
- Default: 4-bit affine, group_size=128
- Attention weights: 6-bit
- Expert gate_proj/up_proj: 2-bit
- Expert down_proj: 6-bit
- Shared expert: 6-bit
- Embeddings: 6-bit
- Norms: fp32 (unquantized)

## Key Differences from Qwen 3.5
1. NO DeltaNet — all layers use MQA + compression variants
2. Hyper-Connections replace residual add (4x hidden width)
3. Low-rank Q and O projections (instead of direct Q/K/V/O)
4. Compressed KV cache (window + compressed buffer)
5. Per-layer compression ratios (0/4/128 pattern)
6. Hash routing for first 3 MoE layers
7. sqrtsoftplus gating instead of softmax
8. Mixed-precision quantization (2/4/6-bit per weight group)
9. Different tokenizer (129K vs 248K vocab)
