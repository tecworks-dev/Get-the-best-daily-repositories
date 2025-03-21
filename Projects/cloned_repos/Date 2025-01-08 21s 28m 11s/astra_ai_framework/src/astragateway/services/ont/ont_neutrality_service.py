import struct

from astracommon.connections.connection_type import ConnectionType
from astracommon.messages.astra.broadcast_message import BroadcastMessage
from astracommon.models.broadcast_message_type import BroadcastMessageType
from astracommon.utils.stats import stats_format
from astracommon.utils.stats.block_stat_event_type import BlockStatEventType
from astracommon.utils.stats.block_statistics_service import block_stats
from astragateway.services.neutrality_service import NeutralityService
from astrautils import logging

logger = logging.get_logger(__name__)


class OntNeutralityService(NeutralityService):
    def _propagate_unencrypted_block_to_network(self, astra_block, connection, block_info):
        if block_info is None:
            raise ValueError("Block info is required to propagate unencrypted block")

        is_consensus_msg, = struct.unpack_from("?", astra_block[8:9])
        broadcast_type = BroadcastMessageType.CONSENSUS if is_consensus_msg else BroadcastMessageType.BLOCK

        broadcast_message = BroadcastMessage(block_info.block_hash, self._node.network_num,
                                             broadcast_type=broadcast_type, is_encrypted=False, blob=astra_block)

        conns = self._node.broadcast(
            broadcast_message,
            connection,
            connection_types=(ConnectionType.RELAY_BLOCK,)
        )
        handling_duration = self._node.track_block_from_node_handling_ended(block_info.block_hash)
        block_stats.add_block_event_by_block_hash(block_info.block_hash,
                                                  BlockStatEventType.ENC_BLOCK_SENT_FROM_GATEWAY_TO_NETWORK,
                                                  network_num=self._node.network_num,
                                                  broadcast_type=broadcast_type,
                                                  requested_by_peer=False,
                                                  peers=conns,
                                                  more_info="Peers: {}; Unencrypted; {}; Handled in {}".format(
                                                      stats_format.connections(conns),
                                                      self._format_block_info_stats(block_info),
                                                      stats_format.duration(handling_duration)))
        logger.info("Propagating block {} to the BDN.", block_info.block_hash)
        return broadcast_message
