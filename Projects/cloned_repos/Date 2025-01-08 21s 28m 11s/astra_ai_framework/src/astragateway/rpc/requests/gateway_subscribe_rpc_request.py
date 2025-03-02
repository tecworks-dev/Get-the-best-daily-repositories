from typing import TYPE_CHECKING, Callable, Optional

from astracommon.feed.feed import FeedKey
from astracommon.rpc import rpc_constants
from astracommon.rpc.astra_json_rpc_request import BxJsonRpcRequest
from astracommon.rpc.requests.subscribe_rpc_request import SubscribeRpcRequest
from astracommon.feed.feed_manager import FeedManager
from astracommon.feed.subscriber import Subscriber
from astracommon.rpc.rpc_errors import RpcInvalidParams

if TYPE_CHECKING:
    # noinspection PyUnresolvedReferences
    # pylint: disable=ungrouped-imports,cyclic-import
    from astragateway.connections.abstract_gateway_node import AbstractGatewayNode


class GatewaySubscribeRpcRequest(SubscribeRpcRequest):
    def __init__(
        self,
        request: BxJsonRpcRequest,
        node: "AbstractGatewayNode",
        feed_manager: FeedManager,
        subscribe_handler: Callable[[Subscriber, FeedKey, Optional[str]], None],
        feed_network: int = 0
    ) -> None:
        super().__init__(request, node, feed_manager, subscribe_handler, feed_network, node.account_model)

    def validate_params_get_options(self):
        self.feed_network = self.node.network_num
        super().validate_params_get_options()
        if (
            self.feed_name in {rpc_constants.ETH_TRANSACTION_RECEIPTS_FEED_NAME, rpc_constants.ETH_ON_BLOCK_FEED_NAME}
            and (not self.node.opts.ws or not self.node.opts.eth_ws_uri or not self.node.get_ws_server_status())
        ):
            raise RpcInvalidParams(
                self.request_id,
                f"In order to use the {self.feed_name} feed, "
                f"your gateway must be connected to your Ethereum node's websocket server "
                f"and websocket RPC must be enabled on your gateway. "
                f"To do so, start your gateway with the following parameters: "
                f"--ws True "
                f"--ws-host <IP address of client application> "
                f"--ws-port 28333 "
                f"--eth-ws-uri ws://[ip_address]:[port]."
            )
