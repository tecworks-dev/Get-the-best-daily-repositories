from argparse import Namespace
from typing import Type
import time
from mock import MagicMock

from astragateway.testing import gateway_helpers
from astracommon.messages.astra.tx_message import TxMessage
from astracommon.test_utils import helpers
from astragateway.connections.abstract_gateway_node import AbstractGatewayNode
from astragateway.connections.ont.ont_gateway_node import OntGatewayNode
from astragateway.messages.ont.tx_ont_message import TxOntMessage
from astragateway.testing.abstract_gateway_integration_test import AbstractGatewayIntegrationTest


class OntBlockPropagationTest(AbstractGatewayIntegrationTest):
    def gateway_class(self) -> Type[AbstractGatewayNode]:
        return OntGatewayNode

    def gateway_opts_1(self) -> Namespace:
        return gateway_helpers.get_gateway_opts(
            9000, sync_tx_service=False, include_default_ont_args=True, blockchain_network_num=33, account_id="1234"
        )

    def gateway_opts_2(self) -> Namespace:
        return gateway_helpers.get_gateway_opts(
            9001, sync_tx_service=False, include_default_ont_args=True, blockchain_network_num=33, account_id="1234"
        )

    def test_transaction_propagation(self):
        initial_message = TxOntMessage(
            12345, 123, helpers.generate_bytearray(250)
        )
        self.gateway_1.account_id = "12345"
        transaction_hash = initial_message.tx_hash()
        transaction_key = self.gateway_1._tx_service.get_transaction_key(transaction_hash)
        self.gateway_1_receive_message_from_blockchain(initial_message)

        time.time = MagicMock(return_value=time.time() + 1)
        self.gateway_1.alarm_queue.fire_alarms()

        self.assertTrue(self.gateway_1._tx_service.has_transaction_contents_by_key(transaction_key))
        self.assertFalse(self.gateway_1._tx_service.has_transaction_short_id_by_key(transaction_key))
        messages_for_relay = self.gateway_1_get_queued_messages_for_relay()
        self.assertEqual(1, len(messages_for_relay))

        tx_message = messages_for_relay[0]
        self.assertIsInstance(tx_message, TxMessage)
        self.assertEqual(tx_message.tx_hash(), transaction_hash)

        tx_message_with_short_id = TxMessage(tx_message.message_hash(), tx_message.network_num(), short_id=10,
                                             tx_val=tx_message.tx_val())
        self.gateway_2_receive_message_from_relay(tx_message_with_short_id)

        self.assertTrue(self.gateway_2._tx_service.has_transaction_contents_by_key(transaction_key))
        self.assertTrue(self.gateway_2._tx_service.has_transaction_short_id_by_key(transaction_key))
        self.assertEqual(10, self.gateway_2._tx_service.get_short_id_by_key(transaction_key))

        messages_for_blockchain = self.gateway_2_get_queued_messages_for_blockchain()
        self.assertEqual(1, len(messages_for_blockchain))

        tx_ont_message = messages_for_blockchain[0]
        self.assertIsInstance(tx_ont_message, TxOntMessage)
        self.assertEqual(tx_ont_message.tx_hash(), transaction_hash)
        self.assertEqual(tx_ont_message.tx(), initial_message.tx())
