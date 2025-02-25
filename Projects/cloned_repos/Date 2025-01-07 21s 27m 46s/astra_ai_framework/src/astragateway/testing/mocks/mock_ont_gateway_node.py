# pyre-ignore-all-errors
from typing import Optional

from mock import MagicMock

from astracommon.connections.connection_type import ConnectionType
from astracommon.models.node_type import NodeType
from astracommon.network.abstract_socket_connection_protocol import AbstractSocketConnectionProtocol
from astracommon.services.transaction_service import TransactionService
from astracommon.test_utils import helpers
from astracommon.test_utils.mocks.mock_node_ssl_service import MockNodeSSLService
from astracommon.utils.blockchain_utils.ont.ont_object_hash import OntObjectHash
from astragateway.connections.abstract_gateway_blockchain_connection import AbstractGatewayBlockchainConnection
from astragateway.connections.abstract_relay_connection import AbstractRelayConnection
from astragateway.connections.ont.ont_gateway_node import OntGatewayNode
from astragateway.messages.ont.block_ont_message import BlockOntMessage
from astragateway.services.abstract_block_cleanup_service import AbstractBlockCleanupService
from astragateway.services.abstract_block_queuing_service import AbstractBlockQueuingService
from astragateway.services.ont.abstract_ont_block_cleanup_service import AbstractOntBlockCleanupService
from astragateway.services.ont.ont_block_queuing_service import OntBlockQueuingService
from astragateway.testing.mocks.mock_blockchain_connection import MockMessageConverter
from astrautils.services.node_ssl_service import NodeSSLService


class _MockCleanupService(AbstractOntBlockCleanupService):
    def __init__(self, node):
        super(_MockCleanupService, self).__init__(node, 1)

    def clean_block_transactions(self, block_msg: BlockOntMessage, transaction_service: TransactionService) -> None:
        pass

    def contents_cleanup(self, block_msg: BlockOntMessage, transaction_service: TransactionService) -> None:
        pass

    def is_marked_for_cleanup(self, block_hash: OntObjectHash) -> bool:
        return False


class MockOntGatewayNode(OntGatewayNode):
    NODE_TYPE = NodeType.EXTERNAL_GATEWAY

    def __init__(self, opts, node_ssl_service: Optional[NodeSSLService] = None):
        if opts.use_extensions:
            helpers.set_extensions_parallelism()
        if node_ssl_service is None:
            node_ssl_service = MockNodeSSLService(self.NODE_TYPE, MagicMock())
        super(MockOntGatewayNode, self).__init__(opts, node_ssl_service)
        self.requester = MagicMock()

        self.broadcast_messages = []
        self.broadcast_to_nodes_messages = []
        self._tx_service = TransactionService(self, 0)
        self.block_cleanup_service = self._get_cleanup_service()
        self.message_converter = MockMessageConverter()
        if opts.use_extensions:
            from astracommon.services.extension_transaction_service import ExtensionTransactionService
            self._tx_service = ExtensionTransactionService(self, self.network_num)
        else:
            self._tx_service = TransactionService(self, self.network_num)
        self.opts.has_fully_updated_tx_service = True
        self.has_active_blockchain_peer = MagicMock(return_value=True)

    def broadcast(self, msg, broadcasting_conn=None, prepend_to_queue=False, connection_types=None):
        if connection_types is None:
            connection_types = [ConnectionType.RELAY_ALL]

        if len(connection_types) == 1 and connection_types[0] == ConnectionType.BLOCKCHAIN_NODE:
            self.broadcast_to_nodes_messages.append(msg)
        else:
            self.broadcast_messages.append((msg, connection_types))
        return []

    def broadcast_msg_to_nodes(self, msg):
        self.broadcast_to_nodes_messages.append(msg)

    def get_tx_service(self, _network_num=None):
        return self._tx_service

    def build_blockchain_connection(
        self, socket_connection: AbstractSocketConnectionProtocol
    ) -> AbstractGatewayBlockchainConnection:
        pass

    def build_relay_connection(self, socket_connection: AbstractSocketConnectionProtocol) -> AbstractRelayConnection:
        pass

    def build_remote_blockchain_connection(
        self, socket_connection: AbstractSocketConnectionProtocol
    ) -> AbstractGatewayBlockchainConnection:
        pass

    def build_block_queuing_service(
        self, connection: AbstractGatewayBlockchainConnection
    ) -> AbstractBlockQueuingService:
        return OntBlockQueuingService(self, connection)

    def build_block_cleanup_service(self) -> AbstractBlockCleanupService:
        pass

    def _get_cleanup_service(self) -> AbstractOntBlockCleanupService:
        return _MockCleanupService(self)
