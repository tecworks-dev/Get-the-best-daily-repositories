from typing import TYPE_CHECKING

from astracommon.network.abstract_socket_connection_protocol import AbstractSocketConnectionProtocol
from astragateway.connections.abstract_relay_connection import AbstractRelayConnection
from astrautils import logging

if TYPE_CHECKING:
    from astragateway.connections.abstract_gateway_node import AbstractGatewayNode

logger = logging.get_logger(__name__)


class OntRelayConnection(AbstractRelayConnection):

    def __init__(self, sock: AbstractSocketConnectionProtocol, node: "AbstractGatewayNode"):
        super(OntRelayConnection, self).__init__(sock, node)

    # TODO: implement msg_tx
