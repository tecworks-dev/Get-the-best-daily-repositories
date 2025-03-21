from typing import TYPE_CHECKING

from astracommon.feed.feed import FeedKey
from astracommon.utils.blockchain_utils.eth import eth_common_utils
from astracommon.utils.object_hash import Sha256Hash
from astragateway.connections.abstract_relay_connection import AbstractRelayConnection
from astracommon.feed.eth.eth_new_transaction_feed import EthNewTransactionFeed
from astracommon.feed.eth.eth_raw_transaction import EthRawTransaction
from astracommon.feed.feed_source import FeedSource

if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    from astragateway.connections.eth.eth_gateway_node import EthGatewayNode


class EthRelayConnection(AbstractRelayConnection):
    node: "EthGatewayNode"

    def publish_new_transaction(
        self, tx_hash: Sha256Hash, tx_contents: memoryview, local_region: bool
    ) -> None:
        gas_price = eth_common_utils.raw_tx_gas_price(tx_contents, 0)
        if gas_price >= self.node.get_network_min_transaction_fee():
            self.node.feed_manager.publish_to_feed(
                FeedKey(EthNewTransactionFeed.NAME, network_num=self.node.network_num),
                EthRawTransaction(
                    tx_hash, tx_contents, FeedSource.BDN_SOCKET, local_region=local_region
                )
            )
