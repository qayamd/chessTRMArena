#COMP5801 Final Project
#Jacob Lane
#Qayam Damji (101287631)
#April 6 2026
#Adapted from DeepMind Technologies (2025) -- bagz file format reader/writer (Apache 2.0).
#binary record store with fast index-based lookup; supports sharding and zstd compression

import bisect
from collections.abc import Sequence
import itertools
import mmap
import os
import re
import shutil
import struct
from typing import Any, SupportsIndex

from etils import epath
from typing_extensions import Self
import zstandard as zstd


class BagFileReader(Sequence[bytes]):
    #reader for a single bagz file

    def __init__(
        self,
        filename: str,
        *,
        separate_limits: bool = False,
        decompress: bool | None = None,
    ) -> None:
        if decompress or (decompress is None and filename.endswith('.bagz')):
            self._process = lambda x: zstd.decompress(x) if x else x
        else:
            self._process = lambda x: x
        self._filename = filename
        fd = os.open(filename, os.O_RDONLY)
        try:
            self._records = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            file_size = self._records.size()
        except ValueError:
            self._records = b''
            file_size = 0
        finally:
            os.close(fd)
        if separate_limits:
            directory, name = os.path.split(filename)
            fd = os.open(os.path.join(directory, 'limits.' + name), os.O_RDONLY)
            try:
                self._limits = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
                index_size = self._limits.size()
            except ValueError:
                self._limits = b''
                index_size = 0
            finally:
                os.close(fd)
            index_start = 0
        else:
            if 0 < file_size < 8:
                raise ValueError('Bagz file too small')
            self._limits = self._records
            if file_size:
                (index_start,) = struct.unpack('<Q', self._records[-8:])
            else:
                index_start = 0
            assert file_size >= index_start
            index_size = file_size - index_start
        assert index_size % 8 == 0
        self._num_records = index_size // 8
        self._limits_start = index_start

    def __len__(self) -> int:
        return self._num_records

    def __getitem__(self, index: SupportsIndex) -> bytes:
        i = index.__index__()
        if not 0 <= i < self._num_records:
            raise IndexError('bagz.BagFileReader index out of range')
        end = i * 8 + self._limits_start
        if i:
            rec_range = struct.unpack('<2q', self._limits[end - 8 : end + 8])
        else:
            rec_range = (0, *struct.unpack('<q', self._limits[end : end + 8]))
        return self._process(self._records[slice(*rec_range)])


class BagShardReader(Sequence[bytes]):
    #reader for sharded bagz files (uses @N shard syntax)

    def __init__(
        self,
        filename: str,
        *,
        separate_limits: bool = False,
        decompress: bool | None = None,
    ) -> None:
        matches = re.findall(r'@(\d+)', filename)
        assert len(matches) == 1
        num_files = int(matches[0])
        assert num_files < 100_000
        self._bags = tuple(
            BagFileReader(
                filename=re.sub(
                    r'@(\d+)', f'-{idx:05d}-of-{num_files:05d}', filename
                ),
                separate_limits=separate_limits,
                decompress=decompress,
            )
            for idx in range(num_files)
        )
        self._accum = tuple(itertools.accumulate(map(len, self._bags)))

    def __len__(self) -> int:
        return self._accum[-1]

    def __getitem__(self, index: int) -> bytes:
        if index < 0:
            index += self._accum[-1]
        if seqn := bisect.bisect_left(self._accum, index + 1):
            index -= self._accum[seqn - 1]
        return self._bags[seqn][index]


class BagReader(Sequence[bytes]):
    #unified reader -- auto-detects single file vs sharded based on @N syntax

    def __init__(
        self,
        filename: str,
        *,
        separate_limits: bool = False,
        decompress: bool | None = None,
    ) -> None:
        if matches := re.findall(r'@(\d+)', filename):
            assert len(matches) == 1
            if int(matches[0]) != 0:
                reader_class = BagShardReader
            else:
                filename = filename.replace(matches[0], '')
                reader_class = BagFileReader
        else:
            reader_class = BagFileReader

        self._reader = reader_class(
            filename=filename,
            separate_limits=separate_limits,
            decompress=decompress,
        )

    def __len__(self) -> int:
        return len(self._reader)

    def __getitem__(self, index: SupportsIndex) -> bytes:
        return self._reader[index]


class BagWriter:
    #writer for bagz files; concatenates index to end of data on close

    def __init__(
        self,
        filename: str,
        *,
        separate_limits: bool = False,
        compress: bool | None = None,
        compression_level: int = 0,
    ) -> None:
        if compress or (compress is None and filename.endswith('.bagz')):
            self._process = zstd.ZstdCompressor(level=compression_level).compress
        else:
            self._process = lambda x: x
        self._separate_limits = separate_limits
        directory, name = os.path.split(filename)
        self._records = open(filename, 'wb')
        self._limits = open(os.path.join(directory, 'limits.' + name), 'wb+')

    def write(self, data: bytes) -> None:
        if data:
            self._records.write(self._process(data))
        self._limits.write(struct.pack('<q', self._records.tell()))

    def flush(self) -> None:
        self._records.flush()
        self._limits.flush()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        #concatenates the limits file to end of data file on close
        if self._separate_limits:
            self._records.close()
            self._limits.close()
        else:
            self._limits.seek(0)
            shutil.copyfileobj(self._limits, self._records)
            self._records.close()
            # close before unlinking so Windows doesn't raise PermissionError
            # (POSIX allows unlinking open files; Windows does not)
            limits_name = self._limits.name
            self._limits.close()
            os.unlink(limits_name)


class BagDataSource:
    #PyGrain-compatible data source for bagz files

    def __init__(self, path: epath.PathLike) -> None:
        self._path = os.fspath(path)
        self._reader = BagReader(self._path)
        self._num_records = len(self._reader)

    def __len__(self) -> int:
        return self._num_records

    def __getitem__(self, record_key: SupportsIndex) -> bytes:
        return self._reader[record_key]

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        del state['_reader']
        return state

    def __setstate__(self, state) -> None:
        self.__dict__.update(state)
        self._reader = BagReader(self._path)

    def __repr__(self) -> str:
        return f'BagDataSource(path={self._path!r}'
