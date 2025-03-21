import typing

from astracommon.messages.astra.compact_block_short_ids_serializer import BlockOffsets
from astracommon.utils.object_hash import Sha256Hash


class BlockHeaderInfo(typing.NamedTuple):
    block_offsets: BlockOffsets
    short_ids: typing.List[int]
    short_ids_len: int
    block_hash: Sha256Hash
    offset: int
    txn_count: int
