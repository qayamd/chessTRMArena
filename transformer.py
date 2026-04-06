#COMP5801 Final Project
#Jacob Lane
#Qayam Damji (101287631)
#April 6 2026
#Adapted from Ruoss et al. (2024) -- DeepMind searchless chess (Apache 2.0).
#standard LLaMA-style transformer decoder for chess position evaluation

import dataclasses
import enum
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class PositionalEncodings(enum.Enum):
    SINUSOID = enum.auto()
    LEARNED = enum.auto()


@dataclasses.dataclass(kw_only=True)
class TransformerConfig(ModelConfig):
    #transformer-specific config; everything else comes from ModelConfig
    num_layers: int = 4
    pos_encodings: PositionalEncodings = PositionalEncodings.SINUSOID  # TRM uses RoPE instead
    apply_qk_layernorm: bool = False  # optional QK norm before attention


def sinusoid_position_encoding(
    sequence_length: int,
    hidden_size: int,
    max_timescale: float = 1e4,
) -> torch.Tensor:
    #sinusoidal positional encodings from Vaswani et al. (2017)
    freqs = np.arange(0, hidden_size, 2)
    inv_freq = max_timescale ** (-freqs / hidden_size)
    pos_seq = np.arange(start=0, stop=sequence_length)
    sinusoid_inp = np.einsum('i,j->ij', pos_seq, inv_freq)
    embeddings = np.concatenate(
        [np.sin(sinusoid_inp), np.cos(sinusoid_inp)], axis=-1
    )
    return torch.from_numpy(embeddings[:, :hidden_size]).float()


class MultiHeadDotProductAttention(nn.Module):
    #multi-head dot-product attention (Vaswani et al., 2017)

    def __init__(
        self,
        num_heads: int,
        num_hiddens_per_head: int,
        embedding_dim: int,
        apply_qk_layernorm: bool = False,
    ) -> None:
        super().__init__()
        self._num_heads = num_heads
        self._num_hiddens_per_head = num_hiddens_per_head
        num_hiddens = num_heads * num_hiddens_per_head

        self.q_proj = nn.Linear(embedding_dim, num_hiddens, bias=False)
        self.k_proj = nn.Linear(embedding_dim, num_hiddens, bias=False)
        self.v_proj = nn.Linear(embedding_dim, num_hiddens, bias=False)
        self.out_proj = nn.Linear(num_hiddens, embedding_dim, bias=False)

        self._apply_qk_layernorm = apply_qk_layernorm
        if apply_qk_layernorm:
            self.q_ln = nn.LayerNorm(num_hiddens)
            self.k_ln = nn.LayerNorm(num_hiddens)

    def forward(
        self,
        inputs_q: torch.Tensor,
        inputs_kv: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = inputs_q.shape

        q = self.q_proj(inputs_q)
        k = self.k_proj(inputs_kv)

        if self._apply_qk_layernorm:
            q = self.q_ln(q)
            k = self.k_ln(k)

        v = self.v_proj(inputs_kv)

        new_shape = (batch_size, -1, self._num_heads, self._num_hiddens_per_head)
        q = q.reshape(new_shape)
        k = k.reshape(new_shape)
        v = v.reshape(new_shape)

        # bthd,bThd->bhtT
        attention = torch.einsum('bthd,bThd->bhtT', q, k)
        attention = attention * (1.0 / math.sqrt(self._num_hiddens_per_head))

        if mask is not None:
            attention = attention.masked_fill(~mask.bool(), float('-inf'))

        normalized_attention = F.softmax(attention, dim=-1)

        output = torch.einsum('bhtT,bThd->bthd', normalized_attention, v)
        output = output.reshape(batch_size, sequence_length, -1)
        return self.out_proj(output)


class MLPBlock(nn.Module):
    #gated MLP (SwiGLU) for the transformer

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        ffn_dim = config.embedding_dim * config.widening_factor
        self.linear1 = nn.Linear(config.embedding_dim, ffn_dim, bias=False)
        self.linear2 = nn.Linear(config.embedding_dim, ffn_dim, bias=False)
        self.out_proj = nn.Linear(ffn_dim, config.embedding_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_proj(F.silu(self.linear1(x)) * self.linear2(x))


class TransformerDecoderLayer(nn.Module):
    #single transformer layer: pre-norm attention + pre-norm MLP

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attn_ln = nn.LayerNorm(config.embedding_dim)
        self.mlp_ln = nn.LayerNorm(config.embedding_dim)
        self.attention = MultiHeadDotProductAttention(
            num_heads=config.num_heads,
            num_hiddens_per_head=config.embedding_dim // config.num_heads,
            embedding_dim=config.embedding_dim,
            apply_qk_layernorm=config.apply_qk_layernorm,
        )
        self.mlp = MLPBlock(config)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        attention_input = self.attn_ln(h)
        h = h + self.attention(attention_input, attention_input, mask)

        mlp_input = self.mlp_ln(h)
        h = h + self.mlp(mlp_input)
        return h


class TransformerDecoder(nn.Module):
    #LLaMA-style transformer decoder for chess eval (Ruoss et al., 2024)
    #pre-norm, SwiGLU FFN, bidirectional by default (no causal mask needed for evaluation)

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        nn.init.trunc_normal_(self.token_embedding.weight, std=config.emb_init_scale)

        if config.pos_encodings == PositionalEncodings.LEARNED:
            assert config.max_sequence_length is not None
            self.pos_embedding = nn.Embedding(
                config.max_sequence_length, config.embedding_dim
            )
        else:
            self.pos_embedding = None

        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(config) for _ in range(config.num_layers)]
        )

        self.post_ln = nn.LayerNorm(config.embedding_dim) if config.apply_post_ln else None
        self.output_proj = nn.Linear(config.embedding_dim, config.output_size)

    def _shift_right(self, targets: torch.Tensor) -> torch.Tensor:
        #prepend BOS token and drop the last token
        bos = torch.zeros(targets.shape[0], 1, dtype=targets.dtype, device=targets.device)
        return torch.cat([bos, targets[:, :-1]], dim=1)

    def forward(self, targets: torch.Tensor) -> torch.Tensor:
        #[B, T] -> [B, T, V] log-softmax logits
        inputs = self._shift_right(targets)

        # token embeddings scaled by sqrt(d)
        embeddings = self.token_embedding(inputs) * math.sqrt(self.config.embedding_dim)

        # positional encodings
        _, seq_len, emb_size = embeddings.shape
        if self.config.pos_encodings == PositionalEncodings.SINUSOID:
            pos_enc = sinusoid_position_encoding(seq_len, emb_size).to(embeddings.device)
        else:
            assert seq_len <= self.config.max_sequence_length
            positions = torch.arange(seq_len, device=embeddings.device)
            pos_enc = self.pos_embedding(positions)

        h = embeddings + pos_enc

        # causal mask: [1, 1, T, T] broadcastable over batch and heads
        if self.config.use_causal_mask:
            mask = torch.tril(torch.ones(1, 1, seq_len, seq_len, device=h.device))
        else:
            mask = None

        for layer in self.layers:
            h = layer(h, mask)

        if self.post_ln is not None:
            h = self.post_ln(h)

        logits = self.output_proj(h)
        return F.log_softmax(logits, dim=-1)
