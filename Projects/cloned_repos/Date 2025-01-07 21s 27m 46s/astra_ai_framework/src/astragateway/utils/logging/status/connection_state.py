from astracommon.models.serializeable_enum import SerializeableEnum


class ConnectionState(SerializeableEnum):
    DISCONNECTED = "Disconnected"
    ESTABLISHED = "Established"
