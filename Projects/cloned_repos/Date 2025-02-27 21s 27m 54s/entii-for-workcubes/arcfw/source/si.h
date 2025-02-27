#pragma once
#include "types.h"
#include "runtime.h"

enum {
	SI_CHANNEL_COUNT = 4,
};

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

static inline bool SiDeviceTypeValid(JOYBUS_DEVICE_TYPE Type) {
	return Type.ValidIfZero == 0;
}

void SiInit(ARTX_SYSTEM_TYPE SystemType);
JOYBUS_DEVICE_TYPE SiGetDeviceType(ULONG channel);
JOYBUS_DEVICE_TYPE SiGetDeviceTypeReset(ULONG channel);
bool SiTransferSync(ULONG channel, PVOID bufWrite, ULONG lenWrite, PVOID bufRead, ULONG lenRead);

static inline bool SiTransferByteSync(ULONG channel, UCHAR command, PVOID bufRead, ULONG lenRead) {
	return SiTransferSync(channel, &command, sizeof(command), bufRead, lenRead);
}
