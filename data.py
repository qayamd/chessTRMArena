#COMP5801 Final Project
#Jacob Lane
#Qayam Damji (101287631)
#April 6 2026
#Adapted from Ruoss et al. (2024) -- data formats from DeepMind searchless chess.
#wraps bagz.BagReader with the tokenizer to give shuffled batched PyTorch DataLoaders

import io
import os
import random
import struct
import sys
import threading
from typing import Callable, Literal, Union

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from tokenizer import tokenize, SEQUENCE_LENGTH

# lazy import so we don't fail at import time if etils isn't installed
def _open_bag(path: str):
    #return a BagReader for the given path
    try:
        from bagz import BagReader  # type: ignore[import]
    except ImportError as e:
        raise ImportError(
            "bagz not importable.  Make sure bagz.py and its dependencies "
            "(etils, zstandard) are installed."
        ) from e
    return BagReader(path)


# proto wire-format helpers

def _decode_varint(data: bytes, pos: int) -> tuple[int, int]:
    #decode a proto varint starting at pos; returns (value, new_pos)
    result = 0
    shift = 0
    while True:
        if pos >= len(data):
            raise ValueError("Truncated varint in proto record")
        b = data[pos]
        result |= (b & 0x7F) << shift
        pos += 1
        if not (b & 0x80):
            return result, pos
        shift += 7


def parse_proto_record(data: bytes) -> tuple[str, int]:
    #decode a minimal chess_utils.ChessData proto -- pulls out FEN (field 1) and bucket (field 2)
    pos = 0
    fen = ""
    return_bucket = -1
    while pos < len(data):
        tag, pos = _decode_varint(data, pos)
        field_number = tag >> 3
        wire_type = tag & 0x7
        if wire_type == 0:                          # varint
            value, pos = _decode_varint(data, pos)
            if field_number == 2:
                return_bucket = int(value)
        elif wire_type == 2:                        # length-delimited
            length, pos = _decode_varint(data, pos)
            value_bytes = data[pos : pos + length]
            pos += length
            if field_number == 1:
                fen = value_bytes.decode("utf-8")
        else:
            # skip unknown wire types gracefully
            if wire_type == 5:      # 32-bit fixed
                pos += 4
            elif wire_type == 1:    # 64-bit fixed
                pos += 8
            else:
                break  # groups (3/4) unsupported; stop parsing
    return fen, return_bucket


# simple binary format: [ 2 bytes: uint16 return_bucket (LE) ] [ N bytes: UTF-8 FEN ]

_SIMPLE_HEADER = struct.Struct("<H")   # uint16 little-endian


def parse_simple_record(data: bytes) -> tuple[str, int]:
    #decode a record in the simple binary format
    if len(data) < _SIMPLE_HEADER.size:
        raise ValueError(f"Record too short ({len(data)} bytes) for simple format")
    (bucket,) = _SIMPLE_HEADER.unpack_from(data)
    fen = data[_SIMPLE_HEADER.size :].decode("utf-8")
    return fen, bucket


def write_simple_record(fen: str, return_bucket: int) -> bytes:
    #encode a (FEN, return_bucket) pair in the simple binary format
    return _SIMPLE_HEADER.pack(return_bucket) + fen.encode("utf-8")


# BC format parsers (bcTrain.bag / bcTest.bag)
#
# bcTrain: [1 byte FEN length] [ASCII FEN] [8 bytes BE float64 win prob]
#   win prob from white's perspective (0.0 = white loss, 1.0 = white win)
#   quantised into num_return_buckets bins
#
# bcTest:  [1 byte FEN length] [ASCII FEN] [UCI move as ASCII]
#   no evaluation bucket stored; returns -1

_BC_SUFFIX_LEN = 8   # length of the float64 suffix in bcTrain records
_BE_FLOAT64 = struct.Struct(">d")


def parse_bc_train_record(
    data: bytes,
    num_return_buckets: int = 128,
) -> tuple[str, int]:
    #parse a bcTrain record: FEN + 8-byte win probability -> (fen, bucket)
    if len(data) <= _BC_SUFFIX_LEN + 1:
        raise ValueError(f"bcTrain record too short ({len(data)} bytes)")
    fen_len = data[0]
    fen = data[1 : 1 + fen_len].decode("ascii")
    win_prob = _BE_FLOAT64.unpack_from(data, len(data) - _BC_SUFFIX_LEN)[0]
    bucket = round(win_prob * (num_return_buckets - 1))
    bucket = max(0, min(num_return_buckets - 1, bucket))
    return fen, bucket


def parse_bc_test_record(data: bytes) -> tuple[str, int]:
    #parse a bcTest record: FEN + UCI move -> (fen, -1)
    #bucket is -1 because bcTest has no evaluation value
    fen_len = data[0]
    fen = data[1 : 1 + fen_len].decode("ascii")
    return fen, -1


# format auto-detection
_PROTO_FEN_TAG = 0x0A   # proto tag for field 1, wire type 2 (string)

_FmtLiteral = Literal["auto", "proto", "simple", "bc_train", "bc_test"]


def _auto_detect_fmt(data: bytes) -> str:
    #infer the record format from the raw bytes
    if not data:
        return "simple"
    if data[0] == _PROTO_FEN_TAG:
        return "proto"
    # simple format stores bucket as uint16 LE; for bucket < 256 second byte is 0x00
    if len(data) > 1 and data[1] == 0x00:
        return "simple"
    # bc_train records end with 8 bytes of binary; at least one will be outside printable ASCII
    if len(data) > _BC_SUFFIX_LEN and any(
        b < 0x20 or b > 0x7E for b in data[-_BC_SUFFIX_LEN:]
    ):
        return "bc_train"
    return "bc_test"


def parse_record(
    data: bytes,
    fmt: _FmtLiteral = "auto",
    num_return_buckets: int = 128,
) -> tuple[str, int]:
    #parse a bagz record into (fen, return_bucket); bucket is -1 for bc_test
    if fmt == "auto":
        fmt = _auto_detect_fmt(data)
    if fmt == "proto":
        return parse_proto_record(data)
    if fmt == "bc_train":
        return parse_bc_train_record(data, num_return_buckets=num_return_buckets)
    if fmt == "bc_test":
        return parse_bc_test_record(data)
    return parse_simple_record(data)


class BagChessDataset(Dataset):
    #maps bagz records to (tokens [T], return_bucket) pairs

    def __init__(
        self,
        path: str,
        num_records: int | None = None,
        fmt: _FmtLiteral = "auto",
        num_return_buckets: int = 128,
    ) -> None:
        self._path = path
        self._fmt = fmt
        self._num_return_buckets = num_return_buckets
        self._bag = _open_bag(path)
        total = len(self._bag)
        self._len = total if num_records is None else min(num_records, total)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self._bag[idx]
        fen, bucket = parse_record(
            raw, self._fmt, num_return_buckets=self._num_return_buckets,
        )
        tokens = tokenize(fen)                          # int32 [T]
        label = torch.tensor(bucket, dtype=torch.long)
        return tokens, label

    # pickle support for multiprocessing DataLoader workers
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        del state["_bag"]   # mmap / file descriptor -- not picklable
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._bag = _open_bag(self._path)


class EpochShuffleSampler(Sampler):
    #shuffles indices at the start of each epoch, reproducibly
    #call set_epoch(e) before each epoch

    def __init__(self, dataset: Dataset, seed: int = 0) -> None:
        self._n = len(dataset)  # type: ignore[arg-type]
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return self._n

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self._seed + self._epoch)
        return iter(torch.randperm(self._n, generator=g).tolist())


class PrefetchLoader:
    #wraps a DataLoader to fetch the next batch on a background thread
    #works around Windows mmap/spawn issues (num_workers=0 + background thread instead)

    def __init__(self, loader: DataLoader) -> None:
        self._loader = loader

    @property
    def dataset(self):
        return self._loader.dataset

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self):
        it = iter(self._loader)
        # pre-fetch the first batch
        batch = [None]
        error = [None]
        done = threading.Event()

        def _fetch():
            try:
                batch[0] = next(it)
            except StopIteration:
                batch[0] = None
            except Exception as e:
                error[0] = e
            done.set()

        t = threading.Thread(target=_fetch, daemon=True)
        t.start()

        while True:
            done.wait()
            if error[0] is not None:
                raise error[0]
            current = batch[0]
            if current is None:
                break
            # start fetching the next batch while we yield the current one
            batch[0] = None
            error[0] = None
            done.clear()
            t = threading.Thread(target=_fetch, daemon=True)
            t.start()
            yield current


def make_dataloader(
    path: str,
    batch_size: int,
    *,
    shuffle: bool = True,
    num_records: int | None = None,
    fmt: _FmtLiteral = "auto",
    num_return_buckets: int = 128,
    seed: int = 0,
    num_workers: int = 0,
    drop_last: bool = True,
    prefetch: bool = True,
) -> tuple[DataLoader | PrefetchLoader, EpochShuffleSampler | None]:
    #create a DataLoader over a bagz chess dataset
    #returns (loader, sampler); call sampler.set_epoch(e) before each epoch when shuffling
    #on Windows use num_workers=0 (mmap can't be pickled across spawn-based workers)
    dataset = BagChessDataset(
        path, num_records=num_records, fmt=fmt,
        num_return_buckets=num_return_buckets,
    )
    if shuffle:
        sampler = EpochShuffleSampler(dataset, seed=seed)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=drop_last,
        )
        if prefetch:
            return PrefetchLoader(loader), sampler
        return loader, sampler
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=drop_last,
        )
        if prefetch:
            return PrefetchLoader(loader), None
        return loader, None


class SyntheticChessDataset(Dataset):
    #generates random (tokens, bucket) pairs -- no file I/O needed
    #useful for unit tests and shape-checking without a real bagz file

    def __init__(
        self,
        size: int = 1024,
        num_return_buckets: int = 128,
        vocab_size: int = 32,
        seq_len: int = SEQUENCE_LENGTH,
        seed: int = 0,
    ) -> None:
        rng = torch.Generator()
        rng.manual_seed(seed)
        self._tokens = torch.randint(
            0, vocab_size, (size, seq_len), dtype=torch.int32, generator=rng
        )
        self._labels = torch.randint(
            0, num_return_buckets, (size,), dtype=torch.long, generator=rng
        )

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._tokens[idx], self._labels[idx]
