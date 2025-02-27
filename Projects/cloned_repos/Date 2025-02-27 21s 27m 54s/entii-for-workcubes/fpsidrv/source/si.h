#pragma once

enum {
	SI_CHANNEL_COUNT = 4,
	SI_POLL_LENGTH = 8
};

typedef void (*SI_TRANSFER_CALLBACK)(ULONG Channel, BOOLEAN Success, PVOID Buffer, ULONG Length);
typedef void (*SI_POLL_CALLBACK)(ULONG Channel, PUCHAR Data);

// Joy Bus device type
typedef union _JOYBUS_DEVICE_TYPE {
	struct {
		UCHAR ValidIfZero;

		union {
			struct {
				UCHAR Mode : 3;
				UCHAR Motor : 2;
				UCHAR GetOrigin : 1;
				UCHAR Unknown : 1;
				UCHAR : 1;
			};
			UCHAR Value;
		} Status;

		USHORT Identifier;
	};
	ULONG Value;
} JOYBUS_DEVICE_TYPE;

static inline BOOLEAN SiDeviceTypeValid(JOYBUS_DEVICE_TYPE Type) {
	return Type.ValidIfZero == 0;
}

NTSTATUS SiInit(void);
JOYBUS_DEVICE_TYPE SiGetDeviceType(ULONG channel);
JOYBUS_DEVICE_TYPE SiGetDeviceTypeReset(ULONG channel);
NTSTATUS SiTransferAsync(ULONG channel, PVOID bufWrite, ULONG lenWrite, PVOID bufRead, ULONG lenRead, SI_TRANSFER_CALLBACK callback);
NTSTATUS SiTransferSync(ULONG channel, PVOID bufWrite, ULONG lenWrite, PVOID bufRead, ULONG lenRead);
NTSTATUS SiPollSetCallback(ULONG channel, SI_POLL_CALLBACK callback);
NTSTATUS SiTransferPoll(ULONG channels, ULONG data, ULONG length);
void SiTogglePoll(BOOLEAN value);

static inline NTSTATUS SiTransferByteSync(ULONG channel, UCHAR command, PVOID bufRead, ULONG lenRead) {
	return SiTransferSync(channel, &command, sizeof(command), bufRead, lenRead);
}