import time
from asyncio import Future
from typing import Optional, List
from unittest import skip
from mock import MagicMock, call

from astracommon.messages.abstract_block_message import AbstractBlockMessage
from astracommon.messages.abstract_message import AbstractMessage
from astracommon.utils.object_hash import Sha256Hash
from astragateway.testing import gateway_helpers
from astracommon import constants
from astracommon.connections.connection_state import ConnectionState
from astracommon.connections.connection_type import ConnectionType
from astracommon.constants import LOCALHOST, MAX_CONNECT_RETRIES
from astracommon.messages.astra.ping_message import PingMessage
from astracommon.models.node_type import NodeType
from astracommon.models.outbound_peer_model import OutboundPeerModel
from astracommon.network.abstract_socket_connection_protocol import AbstractSocketConnectionProtocol
from astracommon.network.socket_connection_state import SocketConnectionState, SocketConnectionStates
from astracommon.services import sdn_http_service
from astracommon.test_utils import helpers
from astracommon.test_utils.abstract_test_case import AbstractTestCase
from astracommon.test_utils.helpers import async_test
from astracommon.test_utils.mocks.mock_node_ssl_service import MockNodeSSLService
from astracommon.test_utils.mocks.mock_socket_connection import MockSocketConnection
from astracommon.utils import network_latency
from astracommon.utils.alarm_queue import AlarmQueue
from astracommon.utils.buffers.output_buffer import OutputBuffer
from astragateway import gateway_constants
from astragateway.connections.abstract_gateway_blockchain_connection import AbstractGatewayBlockchainConnection
from astragateway.connections.abstract_gateway_node import AbstractGatewayNode
from astragateway.connections.abstract_relay_connection import AbstractRelayConnection
from astragateway.connections.btc.btc_node_connection import BtcNodeConnection
from astragateway.connections.btc.btc_relay_connection import BtcRelayConnection
from astragateway.connections.btc.btc_remote_connection import BtcRemoteConnection
from astragateway.services.abstract_block_cleanup_service import AbstractBlockCleanupService
from astragateway.services.push_block_queuing_service import PushBlockQueuingService
from astragateway.gateway_opts import GatewayOpts
from astrautils.services.node_ssl_service import NodeSSLService


class MockPushBlockQueuingService(
    PushBlockQueuingService[AbstractBlockMessage, AbstractMessage]
):
    def __init__(self, node, node_conn):
        super().__init__(node, node_conn)
        self.blocks_sent: List[Sha256Hash] = []

    def build_block_header_message(
        self, block_hash: Sha256Hash, _block_message: AbstractBlockMessage
    ) -> AbstractMessage:
        return AbstractMessage()

    def get_previous_block_hash_from_message(
        self, block_message: AbstractBlockMessage
    ) -> Sha256Hash:
        pass

    def on_block_sent(
        self, block_hash: Sha256Hash, _block_message: AbstractBlockMessage
    ):
        pass


class GatewayNode(AbstractGatewayNode):

    def __init__(self, opts: GatewayOpts, node_ssl_service: Optional[NodeSSLService] = None):
        if node_ssl_service is None:
            node_ssl_service = MockNodeSSLService(self.NODE_TYPE, MagicMock())
        super().__init__(opts, node_ssl_service)
        self.requester = MagicMock()

    def build_block_queuing_service(self, connection: AbstractGatewayBlockchainConnection) -> PushBlockQueuingService:
        return MockPushBlockQueuingService(self, connection)

    def build_block_cleanup_service(self) -> AbstractBlockCleanupService:
        pass

    def build_blockchain_connection(
        self, socket_connection: AbstractSocketConnectionProtocol
    ) -> AbstractGatewayBlockchainConnection:
        return BtcNodeConnection(socket_connection, self)

    def build_relay_connection(self, socket_connection: AbstractSocketConnectionProtocol) -> AbstractRelayConnection:
        return BtcRelayConnection(socket_connection, self)

    def build_remote_blockchain_connection(
        self, socket_connection: AbstractSocketConnectionProtocol
    ) -> AbstractGatewayBlockchainConnection:
        return BtcRemoteConnection(socket_connection, self)


def initialize_relay_node():
    relay_connections = [OutboundPeerModel(LOCALHOST, 8001, node_type=NodeType.RELAY_BLOCK)]
    network_latency.get_best_relays_by_ping_latency_one_per_country = MagicMock(return_value=[relay_connections[0]])
    opts = gateway_helpers.get_gateway_opts(8000, split_relays=True, include_default_btc_args=True)
    if opts.use_extensions:
        helpers.set_extensions_parallelism()
    node = GatewayNode(opts)
    node.enqueue_connection = MagicMock()

    node._register_potential_relay_peers(
        node._find_best_relay_peers(
            network_latency.get_best_relays_by_ping_latency_one_per_country()
        )
    )
    node.enqueue_connection.assert_has_calls([
        call(LOCALHOST, 8001, ConnectionType.RELAY_ALL),
    ], any_order=True)
    node.on_connection_added(MockSocketConnection(1, node, ip_address=LOCALHOST, port=8001))

    node.alarm_queue = AlarmQueue()
    node.enqueue_connection.reset_mock()
    return node


class AbstractGatewayNodeTest(AbstractTestCase):

    def test_has_active_blockchain_peer(self):
        opts = gateway_helpers.get_gateway_opts(8000, blockchain_address = (LOCALHOST, 8001),)
        node = GatewayNode(opts)
        self.assertFalse(node.has_active_blockchain_peer())

        node.on_connection_added(MockSocketConnection(1, node, ip_address=LOCALHOST, port=8001))
        self.assertEqual(1, len(list(node.connection_pool.get_by_connection_types([ConnectionType.BLOCKCHAIN_NODE]))))
        blockchain_conn = next(iter(node.connection_pool.get_by_connection_types([ConnectionType.BLOCKCHAIN_NODE])))
        node.on_blockchain_connection_ready(blockchain_conn)

        # connection added, but inactive
        self.assertFalse(node.has_active_blockchain_peer())

        blockchain_conn.on_connection_established()
        self.assertTrue(node.has_active_blockchain_peer())

    def test_last_active_blockchain_peer_queuing_service_not_destroyed(self):
        opts = gateway_helpers.get_gateway_opts(8000, blockchain_address=(LOCALHOST, 8001))
        node = GatewayNode(opts)

        conn = MockSocketConnection(1, node, ip_address=LOCALHOST, port=8001)
        node.on_connection_added(conn)
        self.assertEqual(1, len(list(node.connection_pool.get_by_connection_types([ConnectionType.BLOCKCHAIN_NODE]))))
        blockchain_conn = next(iter(node.connection_pool.get_by_connection_types([ConnectionType.BLOCKCHAIN_NODE])))
        node.on_blockchain_connection_ready(blockchain_conn)
        blockchain_conn.on_connection_established()
        self.assertTrue(node.has_active_blockchain_peer())
        self.assertTrue(blockchain_conn in node.block_queuing_service_manager.blockchain_peer_to_block_queuing_service)

        blockchain_conn2 = node.build_blockchain_connection(
            MockSocketConnection(2, node, ip_address=LOCALHOST, port=8002))
        blockchain_conn2.CONNECTION_TYPE = ConnectionType.BLOCKCHAIN_NODE
        node.connection_pool.add(2, LOCALHOST, 8002, blockchain_conn2)
        self.assertEqual(2, len(list(node.connection_pool.get_by_connection_types([ConnectionType.BLOCKCHAIN_NODE]))))
        blockchain_conn2 = node.connection_pool.get_by_ipport(LOCALHOST, 8002)
        node.on_blockchain_connection_ready(blockchain_conn2)
        blockchain_conn2.on_connection_established()
        self.assertIsNotNone(node.block_queuing_service_manager.get_block_queuing_service(blockchain_conn2))
        self.assertTrue(blockchain_conn2 in node.block_queuing_service_manager.blockchain_peer_to_block_queuing_service)

        node.connection_pool.delete(blockchain_conn2)
        node.on_blockchain_connection_destroyed(blockchain_conn2)
        self.assertFalse(blockchain_conn2 in node.block_queuing_service_manager.blockchain_peer_to_block_queuing_service)
        
        node.connection_pool.delete(blockchain_conn)
        node.on_blockchain_connection_destroyed(blockchain_conn)
        self.assertTrue(blockchain_conn in node.block_queuing_service_manager.blockchain_peer_to_block_queuing_service)

        node.block_queuing_service_manager.add_block_queuing_service = MagicMock()
        node.on_blockchain_connection_ready(blockchain_conn)
        node.block_queuing_service_manager.add_block_queuing_service.assert_not_called()

    def test_gateway_peer_sdn_update(self):
        # handle duplicates, keeps old, self ip, and maintain peers from CLI
        peer_gateways = [
            OutboundPeerModel(LOCALHOST, 8001, node_type=NodeType.EXTERNAL_GATEWAY),
            OutboundPeerModel(LOCALHOST, 8002, node_type=NodeType.EXTERNAL_GATEWAY)
        ]
        opts = gateway_helpers.get_gateway_opts(8000, peer_gateways=peer_gateways)
        if opts.use_extensions:
            helpers.set_extensions_parallelism()
        node = GatewayNode(opts)
        node.peer_gateways.add(OutboundPeerModel(LOCALHOST, 8004, node_type=NodeType.EXTERNAL_GATEWAY))

        time.time = MagicMock(return_value=time.time() + 10)
        node.alarm_queue.fire_alarms()

        gateway_peers = [
            OutboundPeerModel(LOCALHOST, 8000, node_type=NodeType.EXTERNAL_GATEWAY),
            OutboundPeerModel(LOCALHOST, 8001, node_type=NodeType.EXTERNAL_GATEWAY),
            OutboundPeerModel(LOCALHOST, 8003, node_type=NodeType.EXTERNAL_GATEWAY)
        ]

        future_obj = Future()
        future_obj.set_result(gateway_peers)
        node._process_gateway_peers_from_sdn(future_obj)

        self.assertEqual(4, len(node.outbound_peers))
        self.assertEqual(4, len(node.peer_gateways))
        self.assertNotIn(OutboundPeerModel(LOCALHOST, 8000, node_type=NodeType.EXTERNAL_GATEWAY), node.peer_gateways)
        self.assertIn(OutboundPeerModel(LOCALHOST, 8001, node_type=NodeType.EXTERNAL_GATEWAY), node.peer_gateways)
        self.assertIn(OutboundPeerModel(LOCALHOST, 8002, node_type=NodeType.EXTERNAL_GATEWAY), node.peer_gateways)
        self.assertIn(OutboundPeerModel(LOCALHOST, 8003, node_type=NodeType.EXTERNAL_GATEWAY), node.peer_gateways)
        self.assertIn(OutboundPeerModel(LOCALHOST, 8004, node_type=NodeType.EXTERNAL_GATEWAY), node.peer_gateways)

    def test_gateway_peer_never_destroy_cli_peer(self):
        peer_gateways = [
            OutboundPeerModel(LOCALHOST, 8001)
        ]
        opts = gateway_helpers.get_gateway_opts(8000, peer_gateways=peer_gateways)
        if opts.use_extensions:
            helpers.set_extensions_parallelism()
        node = GatewayNode(opts)
        node.enqueue_connection = MagicMock()
        node.alarm_queue.register_alarm = MagicMock()
        node.alarm_queue.register_approx_alarm = MagicMock()

        mock_socket = MockSocketConnection(7, node, ip_address=LOCALHOST, port=8001)
        node.on_connection_added(mock_socket)
        cli_peer_conn = node.connection_pool.get_by_ipport(LOCALHOST, 8001)
        node.num_retries_by_ip[(LOCALHOST, 8001)] = MAX_CONNECT_RETRIES
        cli_peer_conn.state = ConnectionState.CONNECTING
        node._connection_timeout(cli_peer_conn)
        self.assertFalse(mock_socket.alive)

        node.on_connection_closed(mock_socket.fileno())
        # timeout is fib(3) == 3
        node.alarm_queue.register_alarm.assert_has_calls([call(3, node._retry_init_client_socket,
                                                               LOCALHOST, 8001, ConnectionType.EXTERNAL_GATEWAY)])
        node._retry_init_client_socket(LOCALHOST, 8001, ConnectionType.EXTERNAL_GATEWAY)
        self.assertEqual(MAX_CONNECT_RETRIES + 1, node.num_retries_by_ip[(LOCALHOST, 8001)])

    @async_test
    async def test_gateway_peer_get_more_peers_when_too_few_gateways(self):
        peer_gateways = [
            OutboundPeerModel(LOCALHOST, 8001, node_type=NodeType.EXTERNAL_GATEWAY),
        ]
        opts = gateway_helpers.get_gateway_opts(8000, peer_gateways=peer_gateways, min_peer_gateways=2)
        if opts.use_extensions:
            helpers.set_extensions_parallelism()
        node = GatewayNode(opts)
        node.peer_gateways.add(OutboundPeerModel(LOCALHOST, 8002, node_type=NodeType.EXTERNAL_GATEWAY))
        sdn_http_service.fetch_gateway_peers = MagicMock(return_value=[
            OutboundPeerModel(LOCALHOST, 8003, "12345", node_type=NodeType.EXTERNAL_GATEWAY)
        ])

        node.on_connection_added(
            MockSocketConnection(1, node=node, ip_address=LOCALHOST, port=8002))
        not_cli_peer_conn = node.connection_pool.get_by_ipport(LOCALHOST, 8002)
        not_cli_peer_conn.mark_for_close(False)

        time.time = MagicMock(return_value=time.time() + constants.SDN_CONTACT_RETRY_SECONDS + 1)
        sdn_http_service.fetch_gateway_peers.assert_not_called()
        node.alarm_queue.fire_alarms()

        gateway_peers = [
            OutboundPeerModel(LOCALHOST, 8000, node_type=NodeType.EXTERNAL_GATEWAY),
            OutboundPeerModel(LOCALHOST, 8001, node_type=NodeType.EXTERNAL_GATEWAY),
            OutboundPeerModel(LOCALHOST, 8003, "12345", node_type=NodeType.EXTERNAL_GATEWAY)
        ]

        future_obj = Future()
        future_obj.set_result(gateway_peers)
        node._process_gateway_peers_from_sdn(future_obj)

        self.assertEqual(2, len([node for node in node.outbound_peers if node.node_type in NodeType.GATEWAY_TYPE]))
        self.assertIn(OutboundPeerModel(LOCALHOST, 8001, node_type=NodeType.EXTERNAL_GATEWAY), node.outbound_peers)
        self.assertIn(
            OutboundPeerModel(LOCALHOST, 8003, "12345", node_type=NodeType.EXTERNAL_GATEWAY), node.outbound_peers
        )

    def test_relay_connection(self):
        relay_connections = [OutboundPeerModel(LOCALHOST, 8001, node_type=NodeType.RELAY)]
        network_latency.get_best_relays_by_ping_latency_one_per_country = MagicMock(return_value=[relay_connections[0]])
        opts = gateway_helpers.get_gateway_opts(8000, include_default_btc_args=True)
        if opts.use_extensions:
            helpers.set_extensions_parallelism()
        node = GatewayNode(opts)
        node.enqueue_connection = MagicMock()

        node._register_potential_relay_peers(
            node._find_best_relay_peers(
                network_latency.get_best_relays_by_ping_latency_one_per_country()
            )
        )
        self.assertEqual(1, len(node.peer_relays))

        node.enqueue_connection.assert_has_calls([
            call(LOCALHOST, 8001, ConnectionType.RELAY_ALL),
        ], any_order=True)

        node.on_connection_added(MockSocketConnection(1, node, ip_address=LOCALHOST, port=8001))
        self._check_connection_pool(node, 1, 1, 1, 1)

    def test_relay_reconnect_disconnect_block(self):
        node = initialize_relay_node()

        relay_block_conn = node.connection_pool.get_by_ipport(LOCALHOST, 8001)
        relay_block_conn.mark_for_close()

        self.assertEqual(1, len(node.alarm_queue.alarms))
        self.assertEqual(node._retry_init_client_socket, node.alarm_queue.alarms[0].alarm.fn)
        self.assertEqual(True, node.continue_retrying_connection(LOCALHOST, 8001, relay_block_conn.CONNECTION_TYPE))

        time.time = MagicMock(return_value=time.time() + 1)
        node.alarm_queue.fire_alarms()
        node.enqueue_connection.assert_called_once_with(LOCALHOST, 8001, relay_block_conn.CONNECTION_TYPE)

    def test_relay_no_reconnect_disconnect_block(self):
        sdn_http_service.submit_peer_connection_error_event = MagicMock()
        node = initialize_relay_node()
        node.num_retries_by_ip[(LOCALHOST, 8001)] = constants.MAX_CONNECT_RETRIES

        relay_block_conn = node.connection_pool.get_by_ipport(LOCALHOST, 8001)
        self.assertEqual(False, node.continue_retrying_connection(LOCALHOST, 8001, relay_block_conn.CONNECTION_TYPE))

        relay_block_conn.mark_for_close()

        sdn_http_service.submit_peer_connection_error_event.called_once_with(node.opts.node_id, LOCALHOST, 8001)
        self.assertEqual(0, len(node.peer_relays))

    @async_test
    async def test_queuing_messages_no_remote_blockchain_connection(self):
        node = self._initialize_gateway(True, True, True)
        remote_blockchain_conn = next(iter(node.connection_pool.get_by_connection_types([ConnectionType.REMOTE_BLOCKCHAIN_NODE])))
        remote_blockchain_conn.mark_for_close()

        queued_message = PingMessage(12345)
        node.send_msg_to_remote_node(queued_message)
        self.assertEqual(1, len(node.remote_node_msg_queue._queue))
        self.assertEqual(queued_message, node.remote_node_msg_queue._queue[0])

        node.on_connection_added(MockSocketConnection(1, node, ip_address=LOCALHOST, port=8003))
        next_conn = next(iter(node.connection_pool.get_by_connection_types([ConnectionType.REMOTE_BLOCKCHAIN_NODE])))
        next_conn.outputbuf = OutputBuffer()  # clear buffer

        node.on_remote_blockchain_connection_ready(next_conn)
        self.assertEqual(queued_message.rawbytes().tobytes(), next_conn.outputbuf.get_buffer().tobytes())

    def test_queuing_messages_cleared_after_timeout(self):
        node = self._initialize_gateway(True, True, True)
        remote_blockchain_conn = next(iter(node.connection_pool.get_by_connection_types([ConnectionType.REMOTE_BLOCKCHAIN_NODE])))
        remote_blockchain_conn.mark_for_close()

        queued_message = PingMessage(12345)
        node.send_msg_to_remote_node(queued_message)
        self.assertEqual(1, len(node.remote_node_msg_queue._queue))
        self.assertEqual(queued_message, node.remote_node_msg_queue._queue[0])

        # queue has been cleared
        time.time = MagicMock(return_value=time.time() + node.opts.remote_blockchain_message_ttl + 0.1)
        node.alarm_queue.fire_alarms()

        node.on_connection_added(MockSocketConnection(3, node, ip_address=LOCALHOST, port=8003))
        next_conn = next(iter(node.connection_pool.get_by_connection_types([ConnectionType.REMOTE_BLOCKCHAIN_NODE])))
        next_conn.outputbuf = OutputBuffer()  # clear buffer

        node.on_remote_blockchain_connection_ready(next_conn)
        self.assertEqual(0, next_conn.outputbuf.length)

        next_conn.mark_for_close()

        queued_message = PingMessage(12345)
        node.send_msg_to_remote_node(queued_message)
        self.assertEqual(1, len(node.remote_node_msg_queue._queue))
        self.assertEqual(queued_message, node.remote_node_msg_queue._queue[0])

        node.on_connection_added(MockSocketConnection(4, node, ip_address=LOCALHOST, port=8003))
        reestablished_conn = next(iter(node.connection_pool.get_by_connection_types([ConnectionType.REMOTE_BLOCKCHAIN_NODE])))
        reestablished_conn.outputbuf = OutputBuffer()  # clear buffer

        node.on_remote_blockchain_connection_ready(reestablished_conn)
        self.assertEqual(queued_message.rawbytes().tobytes(), reestablished_conn.outputbuf.get_buffer().tobytes())

    def test_early_exit_no_blockchain_connection(self):
        node = self._initialize_gateway(False, True, False)
        time.time = MagicMock(return_value=time.time() + gateway_constants.INITIAL_LIVELINESS_CHECK_S)

        node.alarm_queue.fire_alarms()
        self.assertTrue(node.should_force_exit)

    def test_early_exit_no_relay_connection(self):
        node = self._initialize_gateway(True, False, False)
        time.time = MagicMock(return_value=time.time() + gateway_constants.INITIAL_LIVELINESS_CHECK_S)

        node.alarm_queue.fire_alarms()
        self.assertTrue(node.should_force_exit)

    def test_exit_after_losing_blockchain_connection(self):
        node = self._initialize_gateway(True, True, False)

        blockchain_conn = next(iter(node.connection_pool.get_by_connection_types([ConnectionType.BLOCKCHAIN_NODE])))
        blockchain_conn.mark_for_close()

        self.assertFalse(node.has_active_blockchain_peer())

        time.time = MagicMock(return_value=time.time() + gateway_constants.INITIAL_LIVELINESS_CHECK_S)
        node.alarm_queue.fire_alarms()
        self.assertFalse(node.should_force_exit)

        time.time = MagicMock(return_value=time.time() + node.opts.stay_alive_duration -
                                           gateway_constants.INITIAL_LIVELINESS_CHECK_S)
        node.alarm_queue.fire_alarms()
        self.assertFalse(node.should_force_exit)

    def test_exit_after_losing_relay_connection(self):
        node = self._initialize_gateway(True, True, False)

        relay_conn = next(iter(node.connection_pool.get_by_connection_types([ConnectionType.RELAY_ALL])))
        # skip retries
        relay_conn.mark_for_close(False)

        time.time = MagicMock(return_value=time.time() + gateway_constants.INITIAL_LIVELINESS_CHECK_S)
        node.alarm_queue.fire_alarms()
        self.assertFalse(node.should_force_exit)

        time.time = MagicMock(return_value=time.time() + node.opts.stay_alive_duration -
                                           gateway_constants.INITIAL_LIVELINESS_CHECK_S)
        node.alarm_queue.fire_alarms()
        self.assertFalse(node.should_force_exit)

    def test_relay_connection_update(self):
        relay_connections = [
            OutboundPeerModel(LOCALHOST, 8001, node_type=NodeType.RELAY_BLOCK),
            OutboundPeerModel(LOCALHOST, 9001, node_type=NodeType.RELAY_BLOCK)
        ]
        network_latency.get_best_relays_by_ping_latency_one_per_country = MagicMock(return_value=[relay_connections[0]])
        opts = gateway_helpers.get_gateway_opts(8000, split_relays=True, include_default_btc_args=True)
        if opts.use_extensions:
            helpers.set_extensions_parallelism()
        node = GatewayNode(opts)
        node.enqueue_connection = MagicMock()

        node._register_potential_relay_peers(
            node._find_best_relay_peers(
                network_latency.get_best_relays_by_ping_latency_one_per_country()
            )
        )
        self.assertEqual(1, len(node.peer_relays))

        node.enqueue_connection.assert_has_calls([
            call(LOCALHOST, 8001, ConnectionType.RELAY_ALL),
        ], any_order=True)

        node.on_connection_added(
            MockSocketConnection(1, node=node, ip_address=LOCALHOST, port=8001))
        self._check_connection_pool(node, 1, 1, 1, 1)

        network_latency.get_best_relays_by_ping_latency_one_per_country = MagicMock(return_value=[relay_connections[1]])

        node.requester.send_threaded_request.reset_mock()
        node._register_potential_relay_peers(
            node._find_best_relay_peers(
                network_latency.get_best_relays_by_ping_latency_one_per_country()
            )
        )
        node.requester.send_threaded_request.assert_called()
        self.assertEqual(1, len(node.peer_relays))
        self.assertEqual(9001, next(iter(node.peer_relays)).port)

        node.enqueue_connection.assert_has_calls([
            call(LOCALHOST, 9001, ConnectionType.RELAY_ALL),
        ], any_order=True)
        node.on_connection_added(MockSocketConnection(3, node, ip_address=LOCALHOST, port=9001))
        for conn in node.connection_pool.get_by_connection_types([ConnectionType.RELAY_BLOCK]):
            self.assertEqual(9001, conn.peer_port)

    @skip("Test is irrelevant because China gateway is connecting to 1 relay")
    def test_register_potential_relay_peers(self):
        node = initialize_relay_node()
        node.opts.country = constants.NODE_COUNTRY_CHINA

        relay1 = OutboundPeerModel(ip="10.10.10.01", port=1, node_id="1", node_type=NodeType.RELAY)
        relay2 = OutboundPeerModel(ip="10.10.10.02", port=1, node_id="2", node_type=NodeType.RELAY)
        relay3 = OutboundPeerModel(ip="10.10.10.03", port=1, node_id="3", node_type=NodeType.RELAY)
        relay4 = OutboundPeerModel(ip="10.10.10.04", port=1, node_id="4", node_type=NodeType.RELAY)

        # More then 2 potential relay, get 2 fastest
        network_latency.get_best_relays_by_ping_latency_one_per_country.return_value = [relay1, relay2]
        node._register_potential_relay_peers(
            node._find_best_relay_peers(
                network_latency.get_best_relays_by_ping_latency_one_per_country()
            )
        )
        self.assertEqual(2, len(node.peer_relays))
        self.assertIn(relay1, node.peer_relays)
        self.assertIn(relay2, node.peer_relays)
        self.assertEqual(2, len(node.peer_transaction_relays))
        self.assertNotIn(relay1, node.peer_transaction_relays)
        self.assertNotIn(relay2, node.peer_transaction_relays)
        self.assertTrue(
            any(
                peer.ip == relay1.ip and peer.port == relay1.port + 1
                for peer in node.peer_transaction_relays
            )
        )
        self.assertTrue(
            any(
                peer.ip == relay2.ip and peer.port == relay2.port + 1
                for peer in node.peer_transaction_relays
            )
        )

        # Second block relay changed
        network_latency.get_best_relays_by_ping_latency_one_per_country.return_value = [relay1, relay3]
        node._register_potential_relay_peers(
            node._find_best_relay_peers(
                [relay1, relay2, relay3]
            )
        )
        self.assertEqual(2, len(node.peer_relays))
        self.assertIn(relay1, node.peer_relays)
        self.assertIn(relay3, node.peer_relays)
        self.assertEqual(2, len(node.peer_transaction_relays))
        self.assertNotIn(relay1, node.peer_transaction_relays)
        self.assertNotIn(relay3, node.peer_transaction_relays)
        self.assertTrue(
            any(
                peer.ip == relay1.ip and peer.port == relay1.port + 1
                for peer in node.peer_transaction_relays
            )
        )
        self.assertTrue(
            any(
                peer.ip == relay3.ip and peer.port == relay3.port + 1
                for peer in node.peer_transaction_relays
            )
        )

        # Both relays changed
        network_latency.get_best_relays_by_ping_latency_one_per_country.return_value = [relay4, relay2]
        node._register_potential_relay_peers(
            node._find_best_relay_peers(
                [relay1, relay2, relay3, relay4]
            )
        )

        self.assertEqual(2, len(node.peer_relays))
        self.assertIn(relay4, node.peer_relays)
        self.assertIn(relay2, node.peer_relays)
        self.assertEqual(2, len(node.peer_transaction_relays))
        self.assertNotIn(relay4, node.peer_transaction_relays)
        self.assertNotIn(relay2, node.peer_transaction_relays)
        self.assertTrue(
            any(
                peer.ip == relay4.ip and peer.port == relay4.port + 1
                for peer in node.peer_transaction_relays
            )
        )
        self.assertTrue(
            any(
                peer.ip == relay2.ip and peer.port == relay2.port + 1
                for peer in node.peer_transaction_relays
            )
        )

        # Use existing peers if new peers are empty
        network_latency.get_best_relays_by_ping_latency_one_per_country.return_value = [relay4, relay2]
        node._register_potential_relay_peers(
            node._find_best_relay_peers(
                []
            )
        )
        self.assertEqual(2, len(node.peer_relays))
        self.assertIn(relay4, node.peer_relays)
        self.assertIn(relay2, node.peer_relays)
        self.assertEqual(2, len(node.peer_transaction_relays))
        self.assertNotIn(relay4, node.peer_transaction_relays)
        self.assertTrue(
            any(
                peer.ip == relay4.ip and peer.port == relay4.port + 1
                for peer in node.peer_transaction_relays
            )
        )
        self.assertTrue(
            any(
                peer.ip == relay2.ip and peer.port == relay2.port + 1
                for peer in node.peer_transaction_relays
            )
        )

        # Only one potential relay
        node.peer_relays.clear()
        node.peer_transaction_relays.clear()
        network_latency.get_best_relays_by_ping_latency_one_per_country.return_value = []
        node._register_potential_relay_peers([relay1])
        self.assertEqual(1, len(node.peer_relays))
        self.assertIn(relay1, node.peer_relays)
        self.assertEqual(1, len(node.peer_transaction_relays))
        self.assertNotIn(relay1, node.peer_transaction_relays)
        self.assertEqual(relay1.ip, next(iter(node.peer_transaction_relays)).ip)
        self.assertEqual(relay1.port + 1, next(iter(node.peer_transaction_relays)).port)

    def _check_connection_pool(self, node, all, relay_block, relay_tx, relay_all):
        self.assertEqual(all, len(node.connection_pool))
        self.assertEqual(relay_block, len(list(node.connection_pool.get_by_connection_types([ConnectionType.RELAY_BLOCK]))))
        self.assertEqual(relay_tx, len(list(node.connection_pool.get_by_connection_types([ConnectionType.RELAY_TRANSACTION]))))
        self.assertEqual(relay_all, len(list(node.connection_pool.get_by_connection_types([ConnectionType.RELAY_ALL]))))

    def _initialize_gateway(
        self, initialize_blockchain_conn: bool, initialize_relay_conn: bool, initialize_remote_blockchain_conn: bool
    ) -> GatewayNode:
        opts = gateway_helpers.get_gateway_opts(
            8000,
            blockchain_address=(LOCALHOST, 8001),
            peer_relays=[OutboundPeerModel(LOCALHOST, 8002)],
            remote_blockchain_ip=LOCALHOST,
            remote_blockchain_port=8003,
            connect_to_remote_blockchain=True,
            include_default_btc_args=True
        )
        if opts.use_extensions:
            helpers.set_extensions_parallelism()
        node = GatewayNode(opts)

        if initialize_blockchain_conn:
            node.on_connection_added(MockSocketConnection(1, node, ip_address=LOCALHOST, port=8001))

            self.assertEqual(1, len(list(node.connection_pool.get_by_connection_types([ConnectionType.BLOCKCHAIN_NODE]))))
            self.assertEqual(1, len(node.blockchain_peers))
            blockchain_conn = next(iter(node.connection_pool.get_by_connection_types([ConnectionType.BLOCKCHAIN_NODE])))

            node.on_blockchain_connection_ready(blockchain_conn)
            self.assertIsNone(node._blockchain_liveliness_alarm)
            block_queuing_service = node.block_queuing_service_manager.get_block_queuing_service(blockchain_conn)
            self.assertIsNotNone(block_queuing_service)
            self.assertEqual(
                block_queuing_service,
                node.block_queuing_service_manager.get_designated_block_queuing_service()
            )

        if initialize_relay_conn:
            node.on_connection_added(MockSocketConnection(2, node, ip_address=LOCALHOST, port=8002))
            self.assertEqual(1, len(list(node.connection_pool.get_by_connection_types([ConnectionType.RELAY_ALL]))))
            relay_conn = next(iter(node.connection_pool.get_by_connection_types([ConnectionType.RELAY_ALL])))

            node.on_relay_connection_ready(relay_conn.CONNECTION_TYPE)
            self.assertIsNone(node._relay_liveliness_alarm)

        if initialize_remote_blockchain_conn:
            node.on_connection_added(MockSocketConnection(3, node, ip_address=LOCALHOST, port=8003))
            self.assertEqual(1, len(list(node.connection_pool.get_by_connection_types([ConnectionType.REMOTE_BLOCKCHAIN_NODE]))))
            remote_blockchain_conn = next(iter(node.connection_pool.get_by_connection_types([ConnectionType.REMOTE_BLOCKCHAIN_NODE])))

            node.on_remote_blockchain_connection_ready(remote_blockchain_conn)

        return node
