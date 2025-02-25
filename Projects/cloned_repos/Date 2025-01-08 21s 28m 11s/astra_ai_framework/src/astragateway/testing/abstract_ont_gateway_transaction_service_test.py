from typing import Tuple, Union, List, Any

from mock import MagicMock

from astracommon.test_utils import helpers
from astragateway.connections.abstract_gateway_node import AbstractGatewayNode
from astragateway.connections.ont.ont_gateway_node import OntGatewayNode
from astragateway.messages.eth.protocol.transactions_eth_protocol_message import TransactionsEthProtocolMessage
from astragateway.messages.ont.tx_ont_message import TxOntMessage
from astragateway.services.gateway_transaction_service import GatewayTransactionService
from astragateway.testing.abstract_gateway_transaction_service_test import TestAbstractGatewayTransactionService


class TestAbstractOntGatewayTransactionService(TestAbstractGatewayTransactionService):

    def _get_transaction_service(self) -> GatewayTransactionService:
        return GatewayTransactionService(self.node, 0)

    def _get_gateway_node(self) -> AbstractGatewayNode:
        mockSslService = MagicMock()
        return OntGatewayNode(self.opts, mockSslService)

    def _get_node_tx_message(
        self
    ) -> Tuple[
        Union[TxOntMessage, TransactionsEthProtocolMessage], List[Tuple[Any, Any]]
    ]:
        magic = 123456
        version = 1
        tx_contents = helpers.generate_bytearray(200)
        msg = TxOntMessage(magic, version, tx_contents)
        tx_contents = msg.payload()
        return msg, [(msg.tx_hash(), tx_contents)]
