#COMP5801 Final Project
#Jacob Lane
#Qayam Damji (101287631)
#April 6 2026
#Adapted from Ruoss et al. (2024) -- ported from DeepMind's JAX codebase (Apache 2.0).
#converts FEN strings into fixed-length int32 token tensors (length 77)

import torch

_CHARACTERS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
    'p', 'n', 'r', 'k', 'q',
    'P', 'B', 'N', 'R', 'Q', 'K',
    'w', '.',
]
_CHARACTERS_INDEX = {letter: index for index, letter in enumerate(_CHARACTERS)}
_SPACES_CHARACTERS = frozenset({'1', '2', '3', '4', '5', '6', '7', '8'})
SEQUENCE_LENGTH = 77


def tokenize(fen: str) -> torch.Tensor:
  #FEN -> int32[77] token tensor
  #spaces expanded (digit -> that many dots), en passant padded to 2 chars, everything fixed-length
  board, side, castling, en_passant, halfmoves_last, fullmoves = fen.split(' ')
  board = board.replace('/', '')
  board = side + board

  indices: list[int] = []

  for char in board:
    if char in _SPACES_CHARACTERS:
      indices.extend(int(char) * [_CHARACTERS_INDEX['.']])
    else:
      indices.append(_CHARACTERS_INDEX[char])

  if castling == '-':
    indices.extend(4 * [_CHARACTERS_INDEX['.']])
  else:
    for char in castling:
      indices.append(_CHARACTERS_INDEX[char])
    if len(castling) < 4:
      indices.extend((4 - len(castling)) * [_CHARACTERS_INDEX['.']])

  if en_passant == '-':
    indices.extend(2 * [_CHARACTERS_INDEX['.']])
  else:
    for char in en_passant:
      indices.append(_CHARACTERS_INDEX[char])

  halfmoves_last = halfmoves_last[:3].ljust(3, '.')
  indices.extend([_CHARACTERS_INDEX[x] for x in halfmoves_last])

  fullmoves = fullmoves[:3].ljust(3, '.')
  indices.extend([_CHARACTERS_INDEX[x] for x in fullmoves])

  assert len(indices) == SEQUENCE_LENGTH

  return torch.tensor(indices, dtype=torch.int32)
