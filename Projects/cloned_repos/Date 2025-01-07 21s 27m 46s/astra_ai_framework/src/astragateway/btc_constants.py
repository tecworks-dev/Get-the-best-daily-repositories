from astracommon.utils.blockchain_utils.btc import btc_common_constants

BTC_MAGIC_NUMBERS = {
    "main": 0xD9B4BEF9,
    "testnet": 0xDAB5BFFA,
    "testnet3": 0x0709110B,
    "regtest": 0xDAB5BFFA,
    "namecoin": 0xFEB4BEF9
}

# The length of everything in the header minus the checksum
BTC_HEADER_MINUS_CHECKSUM = 20
BTC_HDR_COMMON_OFF = 24  # type: int
BTC_BLOCK_HDR_SIZE = 80
BTC_SHORT_NONCE_SIZE = 8
# Length of a sha256 hash
BTC_SHA_HASH_LEN = btc_common_constants.BTC_SHA_HASH_LEN
BTC_IP_ADDR_PORT_SIZE = 18
BTC_COMPACT_BLOCK_SHORT_ID_LEN = 6
BTC_VARINT_MIN_SIZE = 3

# The services that we provide
# 1: can ask for full blocks.
# 0x20: Node that is compatible with the hard fork.
BTC_CASH_SERVICE_BIT = 0x20  # Bitcoin cash service bit
BTC_NODE_SERVICES = 1
BTC_CASH_SERVICES = 33

BTC_OBJTYPE_TX = 1
BTC_OBJTYPE_BLOCK = 2
BTC_OBJTYPE_FILTERED_BLOCK = 3

BTC_HELLO_MESSAGES = [b"version", b"verack"]

# Indicator byte compressing bitcoin blocks to indicate short id
BTC_SHORT_ID_INDICATOR = 0xFF
BTC_SHORT_ID_INDICATOR_AS_BYTEARRAY = bytearray([BTC_SHORT_ID_INDICATOR])
BTC_SHORT_ID_INDICATOR_LENGTH = 1

TX_VERSION_LEN = btc_common_constants.TX_VERSION_LEN
TX_SEGWIT_FLAG_LEN = btc_common_constants.TX_SEGWIT_FLAG_LEN
TX_LOCK_TIME_LEN = btc_common_constants.TX_LOCK_TIME_LEN
TX_SEGWIT_FLAG_VALUE = btc_common_constants.TX_SEGWIT_FLAG_VALUE

NODE_WITNESS_SERVICE_FLAG = (1 << 3)

BTC_VARINT_SHORT_INDICATOR = btc_common_constants.BTC_VARINT_SHORT_INDICATOR
BTC_VARINT_SHORT_INDICATOR_AS_BYTEARRAY = bytearray([BTC_VARINT_SHORT_INDICATOR])
BTC_VARINT_INT_INDICATOR = btc_common_constants.BTC_VARINT_INT_INDICATOR
BTC_VARINT_INT_INDICATOR_AS_BYTEARRAY = bytearray([BTC_VARINT_INT_INDICATOR])
BTC_VARINT_LONG_INDICATOR = btc_common_constants.BTC_VARINT_LONG_INDICATOR
BTC_VARINT_LONG_INDICATOR_AS_BYTEARRAY = bytearray([BTC_VARINT_LONG_INDICATOR])

BTC_COMPACT_BLOCK_RECOVERY_TIMEOUT_S = 10
BTC_COMPACT_BLOCK_DECOMPRESS_MIN_TX_COUNT = 10000

BTC_DEFAULT_BLOCK_SIZE = 621000
BTC_MINIMAL_SUB_TASK_TX_COUNT = 2500
