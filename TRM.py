#COMP5801 Final Project
#Jacob Lane
#Qayam Damji (101287631)
#April 6 2026
#Adapted from Jolicoeur-Martineau (2025).
#TRM: single shared transformer block applied n_recurrence times with anchor re-injection

import copy
import dataclasses
import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention

from config import ModelConfig


@dataclasses.dataclass(kw_only=True)
class TRMConfig(ModelConfig):
    #TRM-specific config; everything else comes from ModelConfig
    num_layers: int = 2  # transformer layers per recurrence step
    n_recurrence: int = 8  # how many times the shared block is applied
    gradient_checkpointing: bool = False  # saves VRAM at high recurrence; see CLAUDE.md


#custom implementation of truncated normal initialization, as nn.init.trunc_normal_ doesn't produce the correct std.
def trunc_normal_init_(tensor, std=1.0, lower=-2.0, upper=2.0):
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a = math.erf(lower / sqrt2)
            b = math.erf(upper / sqrt2)
            z = (b - a) / 2
            c = (2 * math.pi) ** -0.5
            pdf_u = c * math.exp(-0.5 * lower ** 2)
            pdf_l = c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(
                1 - (upper * pdf_u - lower * pdf_l) / z
                - ((pdf_u - pdf_l) / z) ** 2
            )
            tensor.uniform_(a, b)
            tensor.erfinv_()
            tensor.mul_(sqrt2 * comp_std)
            tensor.clip_(lower * comp_std, upper * comp_std)
    return tensor


def _apply_trunc_normal_init(module):
    #if the current layer is linear, apply the truncated normal init
    if isinstance(module, nn.Linear):
        trunc_normal_init_(module.weight, std=1.0 / (module.in_features ** 0.5))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


#EMA, per Jolicoeur-Martineau (2025)
class EMAHelper:
    #Exponential moving average of model weights for stable evaluation.

    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel): module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel): module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1.0 - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel): module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self): return self.shadow
    def load_state_dict(self, state_dict): self.shadow = state_dict


#ROPE (From Jolicoeur-Martineau (2025))
CosSin = Tuple[torch.Tensor, torch.Tensor]

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    #q, k: [B, S, H, D] / cos, sin: [S, D] -> rotated q, k.
    orig_dtype = q.dtype
    q, k = q.to(cos.dtype), k.to(cos.dtype)
    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class RotaryEmbedding(nn.Module):
    #Pre-computed RoPE cos/sin cache.
    def __init__(self, dim, max_position_embeddings, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached

#Custom Attention mechanism, as nn.MultiheadAttention can't inject RoPE mid-pipeline
#I feel like we could drop positional encodings all together as per Gelberg et al. 2025, but thats probably too adventurous
class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = nn.Linear(hidden_size, (num_heads + 2 * num_key_value_heads) * head_dim, bias=False)
        self.o_proj = nn.Linear(self.output_size, hidden_size, bias=False)

    def forward(self, cos_sin, hidden_states):
        B, S, _ = hidden_states.shape

        qkv = self.qkv_proj(hidden_states).view(B, S, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key   = qkv[:, :, self.num_heads : self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads :]

        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # [B, S, H, D] -> [B, H, S, D] for SDPA
        query, key, value = (t.permute(0, 2, 1, 3) for t in (query, key, value))
        out = scaled_dot_product_attention(query=query, key=key, value=value, is_causal=self.causal)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, S, self.output_size)
        return self.o_proj(out)


#SwiGLU FFN
class SwiGLU(nn.Module):
    def __init__(self, hidden_size, expansion=4.0):
        super().__init__()
        #round up to nearest 256x
        inter = math.ceil(expansion * hidden_size * 2 / 3 / 256) * 256
        self.gate_up_proj = nn.Linear(hidden_size, inter * 2, bias=False)
        self.down_proj = nn.Linear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)

#Single shared block: RMSNorm -> Attention -> RMSNorm -> SwiGLU.
class TRMBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_dim = config.embedding_dim // config.num_heads
        self.attn_norm = nn.RMSNorm(config.embedding_dim)
        self.ffn_norm  = nn.RMSNorm(config.embedding_dim)
        self.attention = Attention(
            hidden_size=config.embedding_dim, head_dim=head_dim,
            num_heads=config.num_heads, num_key_value_heads=config.num_heads,
            causal=config.use_causal_mask,
        )
        self.ffn = SwiGLU(config.embedding_dim, expansion=config.widening_factor)

    def forward(self, cos_sin, h):
        h = h + self.attention(cos_sin, self.attn_norm(h))
        h = h + self.ffn(self.ffn_norm(h))
        return h


#The full TRM Architecture, made drop-in compatible w deepmind's'transformerDecoder'
#usagge:
#logits = model(tokens)      # [B, T] -> [B, T, output_size]
#value  = logits[:, -1, :]   # board evaluation from last position
class TRM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_dim = config.embedding_dim // config.num_heads

        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        trunc_normal_init_(self.token_embedding.weight, std=config.emb_init_scale)

        self.rope = RotaryEmbedding(head_dim, config.max_sequence_length)
        self.shared_layers = nn.ModuleList([TRMBlock(config) for _ in range(config.num_layers)])
        self.post_norm = nn.RMSNorm(config.embedding_dim) if config.apply_post_ln else None
        self.output_proj = nn.Linear(config.embedding_dim, config.output_size, bias=False)

        self.apply(_apply_trunc_normal_init)

    def _embed(self, tokens):
        return self.token_embedding(tokens) * math.sqrt(self.config.embedding_dim)

    def _get_cos_sin(self, seq_len):
        cos, sin = self.rope()
        return cos[:seq_len], sin[:seq_len]

    def _apply_block(self, cos_sin, h):
        for layer in self.shared_layers:
            h = layer(cos_sin, h)
        return h

    def _head(self, h):
        if self.post_norm is not None:
            h = self.post_norm(h)
        return F.log_softmax(self.output_proj(h), dim=-1)

    def _apply_block_maybe_ckpt(self, cos_sin, h):
        if self.config.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._apply_block, cos_sin, h, use_reentrant=False,
            )
        return self._apply_block(cos_sin, h)

    def forward(self, targets):
        #[B, T] -> [B, T, output_size] log-softmax logits.
        x0 = self._embed(targets)
        cos_sin = self._get_cos_sin(x0.shape[1])

        h = x0
        for _ in range(self.config.n_recurrence):
            h = self._apply_block_maybe_ckpt(cos_sin, h)
            h = h + x0  # anchor re-injection

        return self._head(h)

    def forward_deep_supervision(self, targets):
        #Returns one output per recurrence step for deep-supervision training.
        #this is from the Jolicoeur-Martineau (2025) paper, but we should probably find the original source to cite
        x0 = self._embed(targets)
        cos_sin = self._get_cos_sin(x0.shape[1])

        h = x0
        outputs = []
        for _ in range(self.config.n_recurrence):
            h = self._apply_block_maybe_ckpt(cos_sin, h)
            h = h + x0
            outputs.append(self._head(h))

        return outputs