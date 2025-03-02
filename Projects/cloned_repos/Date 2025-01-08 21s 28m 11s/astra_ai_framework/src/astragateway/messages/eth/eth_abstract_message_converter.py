from typing import Tuple, Optional, Union, List

from astracommon.models.transaction_flag import TransactionFlag
from astracommon.messages.astra.tx_message import TxMessage
from astracommon.utils import convert
from astracommon.utils.blockchain_utils.bdn_tx_to_astra_tx import bdn_tx_to_astra_tx
from astracommon.utils.blockchain_utils.eth import rlp_utils, eth_common_utils, \
    transaction_validation_utils
from astracommon.utils.object_hash import Sha256Hash
from astracommon import constants as common_constants

from astragateway.abstract_message_converter import AbstractMessageConverter, BlockDecompressionResult
from astragateway.messages.eth.internal_eth_block_info import InternalEthBlockInfo
from astragateway.messages.eth.protocol.pooled_transactions_eth_protocol_message import \
    PooledTransactionsEthProtocolMessage
from astragateway.messages.eth.protocol.transactions_eth_protocol_message import TransactionsEthProtocolMessage
from astragateway.utils.block_info import BlockInfo
from astragateway.utils.eth.eth_utils import parse_transaction_bytes

from astrautils import logging

logger = logging.get_logger(__name__)


def parse_block_message(block_msg: InternalEthBlockInfo):
    msg_bytes = memoryview(block_msg.rawbytes())
    _, block_msg_itm_len, block_msg_itm_start = rlp_utils.consume_length_prefix(msg_bytes, 0)

    block_msg_bytes = msg_bytes[block_msg_itm_start:block_msg_itm_start + block_msg_itm_len]

    _, block_hdr_itm_len, block_hdr_itm_start = rlp_utils.consume_length_prefix(block_msg_bytes, 0)
    block_hdr_full_bytes = block_msg_bytes[0:block_hdr_itm_start + block_hdr_itm_len]
    block_hdr_bytes = block_msg_bytes[block_hdr_itm_start:block_hdr_itm_start + block_hdr_itm_len]

    _, prev_block_itm_len, prev_block_itm_start = rlp_utils.consume_length_prefix(block_hdr_bytes, 0)
    prev_block_bytes = block_hdr_bytes[prev_block_itm_start:prev_block_itm_start + prev_block_itm_len]

    _, txs_itm_len, txs_itm_start = rlp_utils.consume_length_prefix(
        block_msg_bytes, block_hdr_itm_start + block_hdr_itm_len
    )
    txs_bytes = block_msg_bytes[txs_itm_start:txs_itm_start + txs_itm_len]

    remaining_bytes = block_msg_bytes[txs_itm_start + txs_itm_len:]

    return txs_bytes, block_hdr_full_bytes, remaining_bytes, prev_block_bytes


class EthAbstractMessageConverter(AbstractMessageConverter):

    def __init__(self):
        self._last_recovery_idx: int = 0

    def tx_to_astra_txs(
        self,
        tx_msg,
        network_num,
        transaction_flag: Optional[TransactionFlag] = None,
        min_tx_network_fee: int = 0,
        account_id: str = common_constants.DECODED_EMPTY_ACCOUNT_ID
    ) -> List[Tuple[TxMessage, Sha256Hash, Union[bytearray, memoryview]]]:
        """
        Converts Ethereum transactions message to array of internal transaction messages

        The code is optimized and does not make copies of bytes

        :param tx_msg: Ethereum transaction message
        :param network_num: blockchain network number
        :param transaction_flag: transaction flag to assign to the BDN transaction.
        :param min_tx_network_fee: minimum transaction fee. excludes transactions with gas price below this value,
        :param account_id: the account id of the gateway
        :return: array of tuples (transaction message, transaction hash, transaction bytes)
        """

        if not isinstance(
            tx_msg,
            (TransactionsEthProtocolMessage, PooledTransactionsEthProtocolMessage)
        ):
            raise TypeError(
                f"(TransactionsEthProtocolMessage, PooledTransactionsEthProtocolMessage) is expected for arg tx_msg but was {type(tx_msg)}"
            )
        astra_tx_msgs = []

        msg_bytes = memoryview(tx_msg.rawbytes())

        _, length, start = rlp_utils.consume_length_prefix(msg_bytes, 0)
        txs_bytes = msg_bytes[start:]

        tx_start_index = 0

        while tx_start_index < len(txs_bytes):
            astra_tx, _, tx_item_length, tx_item_start = eth_common_utils.raw_tx_to_astra_tx(
                txs_bytes, tx_start_index, network_num, transaction_flag, account_id
            )

            gas_price = eth_common_utils.raw_tx_gas_price(txs_bytes, tx_start_index)
            if gas_price >= min_tx_network_fee:
                astra_tx_msgs.append((astra_tx, astra_tx.message_hash(), astra_tx.tx_val()))

            tx_start_index = tx_item_start + tx_item_length

            if tx_start_index == len(txs_bytes):
                break

        return astra_tx_msgs

    def bdn_tx_to_astra_tx(
        self,
        raw_tx: Union[bytes, bytearray, memoryview],
        network_num: int,
        transaction_flag: Optional[TransactionFlag] = None,
        account_id: str = common_constants.DECODED_EMPTY_ACCOUNT_ID
    ) -> TxMessage:
        return bdn_tx_to_astra_tx(raw_tx, network_num, transaction_flag, account_id)

    def astra_tx_to_tx(self, astra_tx_msg):
        """
        Converts internal transaction message to Ethereum transactions message

        The code is optimized and does not make copies of bytes

        :param astra_tx_msg: internal transaction message
        :return: Ethereum transactions message
        """

        if not isinstance(astra_tx_msg, TxMessage):
            raise TypeError("Type TxMessage is expected for astra_tx_msg arg but was {0}"
                            .format(type(astra_tx_msg)))

        return parse_transaction_bytes(astra_tx_msg.tx_val())

    def block_to_astra_block(
        self, block_msg: InternalEthBlockInfo, tx_service, enable_block_compression: bool, min_tx_age_seconds: float
    ) -> Tuple[memoryview, BlockInfo]:
        """
        Convert Ethereum new block message to internal broadcast message with transactions replaced with short ids

        The code is optimized and does not make copies of bytes

        :param block_msg: Ethereum new block message
        :param tx_service: Transactions service
        :param enable_block_compression
        :param min_tx_age_seconds
        :return: Internal broadcast message bytes (bytearray), tuple (txs count, previous block hash)
        """
        raise NotImplementedError

    def astra_block_to_block(self, astra_block_msg, tx_service) -> BlockDecompressionResult:
        """
        Converts internal broadcast message to Ethereum new block message

        The code is optimized and does not make copies of bytes

        :param astra_block_msg: internal broadcast message bytes
        :param tx_service: Transactions service
        :return: tuple (new block message, block hash, unknown transaction short id, unknown transaction hashes)
        """
        raise NotImplementedError

    def encode_raw_msg(self, raw_msg: str) -> bytes:
        msg_bytes = convert.hex_to_bytes(raw_msg)

        return transaction_validation_utils.normalize_typed_transaction(
            memoryview(msg_bytes)
        ).tobytes()
