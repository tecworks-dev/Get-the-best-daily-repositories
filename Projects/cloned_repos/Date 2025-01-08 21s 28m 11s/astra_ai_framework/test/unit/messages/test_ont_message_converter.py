import functools
import os
import random
from argparse import Namespace

from astragateway.testing import gateway_helpers
from astracommon.test_utils.abstract_test_case import AbstractTestCase
from astracommon.constants import DEFAULT_TX_MEM_POOL_BUCKET_SIZE
from astracommon.test_utils import helpers
from astracommon.test_utils.mocks.mock_node import MockNode
from astracommon.utils import crypto, convert
from astracommon.utils.blockchain_utils.ont.ont_object_hash import OntObjectHash
from astracommon.utils.crypto import SHA256_HASH_LEN
from astracommon.services.transaction_service import TransactionService
from astracommon.services.extension_transaction_service import ExtensionTransactionService

import astragateway.messages.ont.ont_message_converter_factory as converter_factory
from astragateway.messages.ont.abstract_ont_message_converter import AbstractOntMessageConverter
from astragateway.messages.ont.block_ont_message import BlockOntMessage
from astragateway.messages.ont import ont_messages_util


def get_sample_block():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root_dir, "samples/ont_sample_block.txt")) as sample_file:
        ont_block = sample_file.read().strip("\n")
    buf = bytearray(convert.hex_to_bytes(ont_block))
    parsed_block = BlockOntMessage(buf=buf)
    return parsed_block


class multi_setup(object):
    def __call__(self, func):
        @functools.wraps(func)
        def run_multi_setup(instance):
            normal_tx_service, normal_converter = instance.init(False)
            instance.ont_message_converter = normal_converter
            instance.tx_service = normal_tx_service
            func(instance)

            extension_tx_service, extension_converter = instance.init(True)
            instance.ont_message_converter = extension_converter
            instance.tx_service = extension_tx_service
            func(instance)
        return run_multi_setup


class OntMessageConverterTests(AbstractTestCase):
    MAGIC = 123
    SAMPLE_BLOCK_PREV_BLOCK_HASH = "68fbf6edf632abf583385ae97ddd0c8cf15f23c6b30c04061a484b904cb42162"
    SAMPLE_BLOCK_BLOCK_HASH = "e51f80e504a667ae79f5ce19d7714476441a70941d37b00023e12d53feb36091"
    SAMPLE_BLOCK_TX_COUNT = 18

    def setUp(self):
        self.ont_message_converter: AbstractOntMessageConverter = None
        self.tx_service = None
        self.short_ids = []
        self.magic = 12345
        self.version = 23456

    @multi_setup()
    def test_plain_compression(self):
        parsed_block = get_sample_block()
        astra_block, astra_block_info = self.ont_message_converter.block_to_astra_block(parsed_block, self.tx_service, True, 0)
        ref_block, block_info, _, _ = self.ont_message_converter.astra_block_to_block(astra_block, self.tx_service)
        self.assertEqual(parsed_block.rawbytes().tobytes(), ref_block.rawbytes().tobytes())
        self.assertEqual(self.SAMPLE_BLOCK_TX_COUNT, parsed_block.txn_count())
        self.assertEqual(self.SAMPLE_BLOCK_TX_COUNT, ref_block.txn_count())
        self.assertEqual(bytearray(convert.hex_to_bytes(self.SAMPLE_BLOCK_PREV_BLOCK_HASH)),
                         parsed_block.prev_block_hash().get_little_endian())
        self.assertEqual(bytearray(convert.hex_to_bytes(self.SAMPLE_BLOCK_PREV_BLOCK_HASH)),
                         ref_block.prev_block_hash().get_little_endian())
        self.assertEqual(bytearray(convert.hex_to_bytes(self.SAMPLE_BLOCK_BLOCK_HASH)),
                         parsed_block.block_hash().get_little_endian())
        self.assertEqual(bytearray(convert.hex_to_bytes(self.SAMPLE_BLOCK_BLOCK_HASH)),
                         ref_block.block_hash().get_little_endian())

    @multi_setup()
    def test_partial_compression(self):
        parsed_block = get_sample_block()
        transactions_short = parsed_block.txns()[:]
        random.shuffle(transactions_short)
        transactions_short = transactions_short[:int(len(transactions_short) * 0.9)]
        for short_id, txn in enumerate(transactions_short):
            astra_tx_hash, _ = ont_messages_util.get_txid(txn)
            transaction_key = self.tx_service.get_transaction_key(astra_tx_hash)
            self.tx_service.assign_short_id_by_key(transaction_key, short_id + 1)
            self.tx_service.set_transaction_contents_by_key(transaction_key, txn)
        astra_block, block_info = self.ont_message_converter.block_to_astra_block(parsed_block, self.tx_service, True, 0)
        ref_block, _, unknown_tx_sids, unknown_tx_hashes = self.ont_message_converter.astra_block_to_block(
            astra_block, self.tx_service
        )
        self.assertEqual(len(unknown_tx_hashes), 0)
        self.assertEqual(len(unknown_tx_sids), 0)
        self.assertEqual(
            parsed_block.rawbytes().tobytes(), ref_block.rawbytes().tobytes()
        )

    @multi_setup()
    def test_full_compression(self):
        parsed_block = get_sample_block()
        transactions = parsed_block.txns()[:]
        random.shuffle(transactions)
        for short_id, txn in enumerate(transactions):
            astra_tx_hash, _ = ont_messages_util.get_txid(txn)
            transaction_key = self.tx_service.get_transaction_key(astra_tx_hash)
            self.tx_service.assign_short_id_by_key(transaction_key, short_id + 1)
            self.tx_service.set_transaction_contents_by_key(transaction_key, txn)
        astra_block, block_info = self.ont_message_converter.block_to_astra_block(parsed_block, self.tx_service, True, 0)
        ref_block, ref_lock_info, unknown_tx_sids, unknown_tx_hashes = self.ont_message_converter.astra_block_to_block(
            astra_block, self.tx_service
        )
        self.assertEqual(len(block_info.short_ids), block_info.txn_count, "all txs were compressed")
        self.assertEqual(len(unknown_tx_hashes), 0)
        self.assertEqual(len(unknown_tx_sids), 0)
        self.assertEqual(
            parsed_block.rawbytes().tobytes(), ref_block.rawbytes().tobytes()
        )

    @multi_setup()
    def test_ont_block_to_astra_block_and_back_sids_found(self):

        prev_block_hash = bytearray(crypto.bitcoin_hash(b"123"))
        prev_block = OntObjectHash(prev_block_hash, length=SHA256_HASH_LEN)
        merkle_root_hash = bytearray(crypto.bitcoin_hash(b"234"))
        merkle_root = OntObjectHash(merkle_root_hash, length=SHA256_HASH_LEN)
        txns_root_hash = bytearray(crypto.bitcoin_hash(b"345"))
        txns_root = OntObjectHash(txns_root_hash, length=SHA256_HASH_LEN)
        block_root_hash = bytearray(crypto.bitcoin_hash(b"456"))
        block_root = OntObjectHash(block_root_hash, length=SHA256_HASH_LEN)
        consensus_payload = bytes(b'111')
        next_bookkeeper = bytes(b'222')
        bookkeepers = [bytes(33)] * 5
        sig_data = [bytes(2)] * 3
        txns = []
        timestamp = 1
        height = 2
        consensus_data = 3

        ont_block = BlockOntMessage(
            self.magic, self.version, prev_block, txns_root, block_root, timestamp, height, consensus_data,
            consensus_payload, next_bookkeeper, bookkeepers, sig_data, txns, merkle_root
        )
        block_hash = ont_block.block_hash()
        astra_block, block_info = self.ont_message_converter.block_to_astra_block(ont_block, self.tx_service, True, 0)
        self.assertEqual(0, block_info.txn_count)
        self.assertEqual(self.short_ids, list(block_info.short_ids))
        self.assertEqual(ont_block.block_hash(), block_info.block_hash)

        parsed_ont_block, block_info, _, _ = self.ont_message_converter.astra_block_to_block(
            astra_block,
            self.tx_service
        )
        self.assertIsNotNone(block_info)
        self.assertEqual(ont_block.rawbytes().tobytes(), parsed_ont_block.rawbytes().tobytes())
        self.assertEqual(self.magic, parsed_ont_block.magic())
        self.assertEqual(prev_block_hash, parsed_ont_block.prev_block_hash().get_little_endian())
        self.assertEqual(ont_block.checksum(), parsed_ont_block.checksum())
        self.assertEqual(block_hash, parsed_ont_block.block_hash())
        self.assertEqual(block_hash.binary, block_info.block_hash.binary)
        self.assertEqual(timestamp, parsed_ont_block.timestamp())

    @multi_setup()
    def test_no_compression(self):
        parsed_block = get_sample_block()
        astra_block, block_info = self.ont_message_converter.block_to_astra_block(parsed_block, self.tx_service, False, 0)
        self.assertTrue(len(parsed_block.rawbytes()) <= len(astra_block))
        self.assertEqual(len(block_info.short_ids), 0)

    def init(self, use_extensions: bool):
        opts = Namespace()
        opts.use_extensions = use_extensions
        opts.import_extensions = use_extensions
        opts.tx_mem_pool_bucket_size = DEFAULT_TX_MEM_POOL_BUCKET_SIZE
        ont_message_converter = converter_factory.create_ont_message_converter(self.MAGIC, opts)
        if use_extensions:
            helpers.set_extensions_parallelism()
            tx_service = ExtensionTransactionService(MockNode(
                gateway_helpers.get_gateway_opts(8999)), 0)
        else:
            tx_service = TransactionService(MockNode(
                gateway_helpers.get_gateway_opts(8999)), 0)
        return tx_service, ont_message_converter
