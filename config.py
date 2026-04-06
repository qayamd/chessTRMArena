#COMP5801 Final Project
#Jacob Lane
#Qayam Damji (101287631)
#April 6 2026
#Adapted from Ruoss et al. (2024) and Jolicoeur-Martineau (2025).
#shared base config dataclass for both model architectures

import dataclasses


@dataclasses.dataclass(kw_only=True)
class ModelConfig:
  #base config inherited by TRMConfig and TransformerConfig
  vocab_size: int  # 32 for the chess FEN tokenizer
  output_size: int | None = None  # 128 value bins; defaults to vocab_size
  embedding_dim: int = 64
  num_heads: int = 8
  widening_factor: int = 4  # FFN expansion ratio
  use_causal_mask: bool = False  # bidirectional for chess eval
  emb_init_scale: float = 0.02
  max_sequence_length: int = 128  # used by RoPE and learned pos encodings
  apply_post_ln: bool = True
  seed: int = 1

  def __post_init__(self):
    if self.output_size is None:
      self.output_size = self.vocab_size
