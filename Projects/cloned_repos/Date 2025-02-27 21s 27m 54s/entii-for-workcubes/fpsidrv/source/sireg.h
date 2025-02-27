#pragma once

// SI register definitions.

// Please note: any arrays in these unions will use little endian ordering due to MSR_LE swizzle or optimisation.
// So array[4] => correct big endian index is array[3 - index]

typedef union _SI_CHANNEL_COMMAND_REGISTER {
	struct {
		UCHAR Data2, Data1, Data0, Pad;
	};
	ULONG Value;
} SI_CHANNEL_COMMAND_REGISTER;

typedef union _SI_CHANNEL_RESPONSE_HIGH_REGISTER {
	struct {
		UCHAR Data3, Data2, Data1;

		UCHAR Data0 : 6;
		BOOLEAN ErrorLatch : 1;
		BOOLEAN ErrorStatus : 1;
	};

	ULONG Value;
} SI_CHANNEL_RESPONSE_HIGH_REGISTER;

typedef union _SI_CHANNEL_RESPONSE_LOW_REGISTER {
	struct {
		UCHAR Data3, Data2, Data1, Data0;
	};
	ULONG Value;
} SI_CHANNEL_RESPONSE_LOW_REGISTER;

typedef struct ARC_ALIGNED(4) _SI_CHANNEL_REGISTERS {
	SI_CHANNEL_COMMAND_REGISTER Command;
	SI_CHANNEL_RESPONSE_HIGH_REGISTER ResponseHigh;
	SI_CHANNEL_RESPONSE_LOW_REGISTER ResponseLow;
} SI_CHANNEL_REGISTERS, *PSI_CHANNEL_REGISTERS;

typedef union _SI_POLL_REGISTER {
	struct {
		BOOLEAN
			VblankCopy3 : 1,
			VblankCopy2 : 1,
			VblankCopy1 : 1,
			VblankCopy0 : 1;
		BOOLEAN
			Enable3 : 1,
			Enable2 : 1,
			Enable1 : 1,
			Enable0 : 1;

		UCHAR YTimes;

		USHORT XLines : 10;
		USHORT : 6;
	};
	ULONG Value;
} SI_POLL_REGISTER, *PSI_POLL_REGISTER;

typedef union _SI_COMMUNICATION_CONTROL_STATUS_REGISTER {
	struct {
		BOOLEAN TransferStart : 1;
		UCHAR Channel : 2;
		UCHAR : 5;

		UCHAR InputLength : 7;
		UCHAR : 1;

		UCHAR OutputLength : 7;
		UCHAR : 1;

		UCHAR : 3;
		BOOLEAN ReadStatusInterruptMask : 1;
		BOOLEAN ReadStatusInterruptStatus : 1;
		BOOLEAN CommunicationError : 1;
		BOOLEAN TransferCompleteInterruptMask : 1;
		BOOLEAN TransferCompleteInterruptStatus : 1;
	};
	ULONG Value;
} SI_COMMUNICATION_CONTROL_STATUS_REGISTER, *PSI_COMMUNICATION_CONTROL_STATUS_REGISTER;

typedef union _SI_STATUS_REGISTER_CHANNEL {
	struct {
		BOOLEAN ErrorUnderrun : 1;
		BOOLEAN ErrorOverrun : 1;
		BOOLEAN ErrorCollision : 1;
		BOOLEAN ErrorNoResponse : 1;
		BOOLEAN WriteStatus : 1;
		BOOLEAN ReadStatus : 1;
		UCHAR : 2;
	};
	UCHAR Value;
} SI_STATUS_REGISTER_CHANNEL, *PSI_STATUS_REGISTER_CHANNEL;

typedef union _SI_STATUS_REGISTER {
	struct {
		ULONG : 31;
		BOOLEAN WriteAll : 1;
	};
	struct {
		SI_STATUS_REGISTER_CHANNEL Channel3, Channel2, Channel1, Channel0;
	};
	struct {
		SI_STATUS_REGISTER_CHANNEL Channels[4];
	};
	ULONG Value;
} SI_STATUS_REGISTER, *PSI_STATUS_REGISTER;

typedef union _SI_EXI_CLOCK_LOCK_REGISTER {
	struct {
		ULONG : 31;
		BOOLEAN Enabled : 1;
	};
	ULONG Value;
} SI_EXI_CLOCK_LOCK_REGISTER, *PSI_EXI_CLOCK_LOCK_REGISTER;

typedef struct _SI_REGISTERS {
	// 0x00
	SI_CHANNEL_REGISTERS Channel[SI_CHANNEL_COUNT];
	// 0x30
	SI_POLL_REGISTER Poll;
	SI_COMMUNICATION_CONTROL_STATUS_REGISTER CommunicationControlStatus;
	SI_STATUS_REGISTER Status;
	SI_EXI_CLOCK_LOCK_REGISTER ClockLock;
	// 0x40
	ULONG Reserved[0x40 / sizeof(ULONG)];
	// 0x80
	ULONG IoBuffer[0x80 / sizeof(ULONG)];
	// 0x100
} SI_REGISTERS, *PSI_REGISTERS;

_Static_assert(sizeof(SI_REGISTERS) == 0x100);

enum {
	SI_MAX_TRANSFER_LENGTH = sizeof(((PSI_REGISTERS)NULL)->IoBuffer)
};

enum {
	SI_REGISTER_BASE_FLIPPER = 0x0C006400,
	SI_REGISTER_OFFSET_VEGAS = (0x0D006400 - SI_REGISTER_BASE_FLIPPER)
};
