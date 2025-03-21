from typing import List

import blxr_rlp as rlp
from astracommon.utils import convert
from astracommon.utils.object_hash import Sha256Hash
from astrautils.logging.log_level import LogLevel

from astragateway.messages.eth.protocol.eth_protocol_message import EthProtocolMessage
from astragateway.messages.eth.protocol.eth_protocol_message_type import EthProtocolMessageType
from astracommon.utils.blockchain_utils.eth import rlp_utils


class GetBlockBodiesEthProtocolMessage(EthProtocolMessage):
    msg_type = EthProtocolMessageType.GET_BLOCK_BODIES

    fields = [("block_hashes", rlp.sedes.CountableList(rlp.sedes.binary))]

    def __repr__(self):
        requested_hashes = self.get_field_value("block_hashes")
        request_repr = list(requested_hashes[:1])
        if len(requested_hashes) > 1:
            request_repr.append(requested_hashes[-1])
        request_repr = map(convert.bytes_to_hex, request_repr)
        return (
            f"GetBlockBodiesEthProtocolMessage<"
            f"bodies_count: {len(requested_hashes)}, "
            f"hashes: [{'...'.join([requested_hash for requested_hash in request_repr])}]>"
        )

    def get_block_hashes(self) -> List[Sha256Hash]:
        block_hashes = list(self.get_field_value("block_hashes"))
        if block_hashes and not isinstance(block_hashes[0], Sha256Hash):
            block_hashes = list(map(Sha256Hash, block_hashes))
        return block_hashes

    def log_level(self):
        return LogLevel.DEBUG
